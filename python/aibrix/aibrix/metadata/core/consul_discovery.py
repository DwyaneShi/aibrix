import asyncio
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from servicediscovery import Client

from aibrix import envs
from aibrix.logger import init_logger

_DEFAULT_LOOKUP_TIMEOUT_SECONDS = 1.0
_DEFAULT_REFRESH_INTERVAL_SECONDS = 3.0
_DEFAULT_PRIORITY_REFRESH_INTERVAL_SECONDS = 0.1
_DEFAULT_FILTER_TAGS = ("aibrix_served_model_name",)
_RAW_SNAPSHOT_KEY: tuple[tuple[str, str], ...] = ()

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class ConsulInferenceEndpoint:
    host: str
    port: int
    tags: dict[str, str]

    @property
    def base_url(self) -> str:
        # IPv6 literals must be bracketed in a URL authority, otherwise the
        # address's own colons get misparsed as the host:port separator.
        host = (
            f"[{self.host}]"
            if ":" in self.host and not self.host.startswith("[")
            else self.host
        )
        return f"http://{host}:{self.port}"


@dataclass(frozen=True, slots=True)
class ConsulInferenceSnapshot:
    version: int
    endpoints: list[ConsulInferenceEndpoint]


class ConsulDiscoveryService:
    def __init__(
        self,
        consul_host: Optional[str] = None,
        consul_port: Optional[str] = None,
        consul_psm: Optional[str] = None,
        debug: bool = False,
        refresh_interval_seconds: float = _DEFAULT_REFRESH_INTERVAL_SECONDS,
    ) -> None:
        self._consul_host = consul_host or envs.CONSUL_HTTP_HOST
        self._consul_port = consul_port or envs.CONSUL_HTTP_PORT
        self._consul_psm = consul_psm or envs.CONSUL_BATCH_DISCOVERY_PSM
        # Required to set for servicediscovery to work locally
        os.environ["CONSUL_HTTP_HOST"] = self._consul_host
        os.environ["CONSUL_HTTP_PORT"] = self._consul_port
        self._client = Client()
        self._debug = debug
        self._refresh_interval_seconds = refresh_interval_seconds
        self._priority_refresh_interval_seconds = min(
            _DEFAULT_PRIORITY_REFRESH_INTERVAL_SECONDS,
            refresh_interval_seconds,
        )
        self._lifecycle_lock = threading.Lock()
        self._cache_lock = threading.RLock()
        self._refresh_condition = threading.Condition(self._cache_lock)
        self._stop_event = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None
        self._started = False
        self._next_regular_refresh_at = (
            time.monotonic() + self._refresh_interval_seconds
        )
        self._next_priority_refresh_at: Optional[float] = None
        self._seen_psms: dict[str, set[tuple[str, ...]]] = {}
        self._priority_psms: set[str] = set()
        self._refresh_round_by_psm: dict[str, int] = {}
        self._version_by_snapshot: dict[
            str, dict[tuple[tuple[str, str], ...], int]
        ] = {}
        self._raw_endpoints: dict[str, list[ConsulInferenceEndpoint]] = {}
        self._categorized_endpoints: dict[
            str,
            dict[
                tuple[str, ...],
                dict[
                    tuple[tuple[str, str], ...],
                    list[ConsulInferenceEndpoint],
                ],
            ],
        ] = {}

    def resolve_psm(self, psm: Optional[str] = None) -> str:
        """Resolve the caller override or fall back to the service default PSM."""
        return psm or self._consul_psm

    def start(self) -> None:
        """Start the background refresh thread once for this service instance."""
        with self._lifecycle_lock:
            if self._started:
                return
            self._stop_event.clear()
            with self._refresh_condition:
                self._next_regular_refresh_at = (
                    time.monotonic() + self._refresh_interval_seconds
                )
                self._next_priority_refresh_at = None
            self._refresh_thread = threading.Thread(
                target=self._refresh_loop,
                name="consul-discovery-refresh",
                daemon=True,
            )
            self._refresh_thread.start()
            self._started = True

    async def stop(self) -> None:
        """Stop the background refresh thread and wait for it to exit."""
        thread = None
        with self._lifecycle_lock:
            if not self._started:
                return
            self._stop_event.set()
            thread = self._refresh_thread
            self._refresh_thread = None
            self._started = False
        with self._refresh_condition:
            self._refresh_condition.notify_all()
        if thread is not None:
            await asyncio.to_thread(thread.join)

    async def lookup(
        self,
        service_id: Optional[str] = None,
        filter_tags: Optional[dict[str, str]] = None,
        lookup_timeout_seconds: Optional[float] = None,
    ) -> ConsulInferenceSnapshot:
        resolved_psm = self.resolve_psm(service_id)
        category_key = self._category_key(filter_tags)
        if lookup_timeout_seconds is None:
            lookup_timeout_seconds = self._lookup_timeout_seconds()
        requires_network_refresh = self._register_filter_category(
            resolved_psm,
            category_key,
        )
        if requires_network_refresh:
            if self._started:
                self._enqueue_priority_refresh(resolved_psm)
            else:
                # No background refreshing enabled, trigger immediate refreshing.
                logger.warning(
                    "background refreshing not enabled, trigger on-demand refreshing, call ConsulDiscoveryService.start() to enable background refreshing."
                )
                await asyncio.to_thread(
                    self._refresh_psm,
                    resolved_psm,
                    lookup_timeout_seconds,
                )
        return self._get_lookup_snapshot(resolved_psm, filter_tags)

    async def discover_model_endpoints(
        self,
        served_model_name: str,
        service_id: Optional[str] = None,
        filter_tags: Optional[dict[str, str]] = None,
        lookup_timeout_seconds: Optional[float] = None,
    ) -> ConsulInferenceSnapshot:
        # Ensure every query includes the default model-name tag filter.
        normalized_filter_tags = self._normalize_filter_tags(
            served_model_name=served_model_name,
            filter_tags=filter_tags,
        )
        resolved_psm = self.resolve_psm(service_id)
        snapshot = await self.lookup(
            service_id=resolved_psm,
            filter_tags=normalized_filter_tags,
            lookup_timeout_seconds=lookup_timeout_seconds,
        )
        if snapshot.endpoints or lookup_timeout_seconds is None:
            return snapshot

        if self._started:
            refresh_round = self._enqueue_priority_refresh(resolved_psm)
            await asyncio.to_thread(
                self._wait_for_refresh_round,
                resolved_psm,
                refresh_round,
                lookup_timeout_seconds,
            )
        else:
            # No background refreshing enabled, trigger immediate refreshing.
            logger.warning(
                "background refreshing not enabled, trigger on-demand refreshing, call ConsulDiscoveryService.start() to enable background refreshing."
            )
            await asyncio.to_thread(
                self._refresh_psm,
                resolved_psm,
                lookup_timeout_seconds,
            )
        return self._get_lookup_snapshot(
            resolved_psm,
            normalized_filter_tags,
        )

    async def wait_for_model_endpoints(
        self,
        served_model_name: str,
        timeout_seconds: float,
        service_id: Optional[str] = None,
        filter_tags: Optional[dict[str, str]] = None,
        lookup_timeout_seconds: Optional[float] = None,
        poll_interval_seconds: float = 1.0,
    ) -> ConsulInferenceSnapshot:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        single_timeout = lookup_timeout_seconds or self._lookup_timeout_seconds()
        while True:
            remaining_timeout_seconds = deadline - loop.time()
            if remaining_timeout_seconds <= 0:
                raise TimeoutError(
                    f"Timed out waiting for Consul endpoints for model "
                    f"'{served_model_name}'"
                )
            lookup_timeout_seconds = max(
                min(remaining_timeout_seconds, single_timeout),
                0.05,  # give 50ms as minimum timeout
            )
            try:
                snapshot = await self.discover_model_endpoints(
                    served_model_name=served_model_name,
                    service_id=service_id,
                    filter_tags=filter_tags,
                    lookup_timeout_seconds=lookup_timeout_seconds,
                )
            except TimeoutError as ex:
                if loop.time() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for Consul endpoints for model "
                        f"'{served_model_name}'"
                    ) from ex
                continue
            if snapshot.endpoints:
                return snapshot
            remaining_sleep_seconds = deadline - loop.time()
            if remaining_sleep_seconds <= 0:
                raise TimeoutError(
                    f"Timed out waiting for Consul endpoints for model "
                    f"'{served_model_name}'"
                )
            await asyncio.sleep(min(poll_interval_seconds, remaining_sleep_seconds))

    def _refresh_loop(self) -> None:
        """Periodically refresh every PSM that has been registered through lookup."""
        while not self._stop_event.is_set():
            with self._refresh_condition:
                refresh_psms = self._collect_refresh_psms()
                while not refresh_psms and not self._stop_event.is_set():
                    self._refresh_condition.wait(self._next_refresh_wait_seconds())
                    if self._stop_event.is_set():
                        return
                    refresh_psms = self._collect_refresh_psms()
            for psm in refresh_psms:
                if self._stop_event.is_set():
                    return
                try:
                    self._refresh_psm(psm, self._lookup_timeout_seconds())
                except Exception:
                    logger.warning(
                        "refresh consul psm failed",
                        psm=psm,
                        exc_info=True,
                    )  # type: ignore[call-arg]

    def _refresh_psm(self, psm: str, lookup_timeout_seconds: float) -> None:
        """Fetch one PSM from Consul and rebuild all cached categories for it."""
        response_payload = self._client.lookup_name(
            psm,
            timeout=max(1, int(lookup_timeout_seconds)),
        )
        endpoints = [self._parse_endpoint(item) for item in response_payload]
        with self._cache_lock:
            previous_signature = self._endpoint_signature(
                self._raw_endpoints.get(psm, [])
            )
            previous_category_signatures = self._category_signatures(
                self._categorized_endpoints.get(psm, {})
            )
            self._raw_endpoints[psm] = endpoints
            category_keys = set(self._seen_psms.get(psm, set()))
        categorized = self._categorize_endpoints(endpoints, category_keys)
        current_signature = self._endpoint_signature(endpoints)
        current_category_signatures = self._category_signatures(categorized)
        with self._refresh_condition:
            snapshot_versions = self._version_by_snapshot.setdefault(psm, {})
            if current_signature != previous_signature:
                snapshot_versions[_RAW_SNAPSHOT_KEY] = (
                    snapshot_versions.get(_RAW_SNAPSHOT_KEY, 0) + 1
                )
            for category_value in set(previous_category_signatures) | set(
                current_category_signatures
            ):
                if current_category_signatures.get(
                    category_value
                ) != previous_category_signatures.get(category_value):
                    snapshot_versions[category_value] = (
                        snapshot_versions.get(category_value, 0) + 1
                    )
            self._categorized_endpoints[psm] = categorized
            self._refresh_round_by_psm[psm] = self._refresh_round_by_psm.get(psm, 0) + 1
            self._refresh_condition.notify_all()

    def _lookup_timeout_seconds(self) -> float:
        """Return the default per-request lookup timeout, widened in debug mode."""
        if self._debug:
            return 5 * _DEFAULT_LOOKUP_TIMEOUT_SECONDS
        return _DEFAULT_LOOKUP_TIMEOUT_SECONDS

    def _get_categorized_endpoints(
        self,
        psm: str,
        category_key: tuple[str, ...],
        category_value: tuple[tuple[str, str], ...],
    ) -> list[ConsulInferenceEndpoint]:
        """Read a snapshot of cached endpoints for one concrete filter category."""
        with self._cache_lock:
            categorized = self._categorized_endpoints.get(psm, {})
            by_key = categorized.get(category_key, {})
            return list(by_key.get(category_value, []))

    def _get_lookup_snapshot(
        self,
        psm: str,
        filter_tags: Optional[dict[str, str]] = None,
    ) -> ConsulInferenceSnapshot:
        with self._cache_lock:
            if filter_tags:
                category_key = self._category_key(filter_tags)
                category_value = self._category_value(filter_tags)
                categorized = self._categorized_endpoints.get(psm, {})
                by_key = categorized.get(category_key, {})
                endpoints = list(by_key.get(category_value, []))
                version = self._version_by_snapshot.get(psm, {}).get(category_value, 0)
                return ConsulInferenceSnapshot(version=version, endpoints=endpoints)
            return ConsulInferenceSnapshot(
                version=self._version_by_snapshot.get(psm, {}).get(
                    _RAW_SNAPSHOT_KEY, 0
                ),
                endpoints=list(self._raw_endpoints.get(psm, [])),
            )

    def _register_filter_category(
        self,
        psm: str,
        category_key: tuple[str, ...],
    ) -> bool:
        """Track one filter-shape for a PSM and report whether Consul refresh is needed."""
        with self._cache_lock:
            seen_filters = self._seen_psms.setdefault(psm, set())
            category_added = category_key not in seen_filters
            if category_added:
                seen_filters.add(category_key)

            raw_endpoints = self._raw_endpoints.get(psm)
            if raw_endpoints is None:
                return True

            if category_added:
                categorized = self._categorize_endpoints(
                    raw_endpoints, set(seen_filters)
                )
                self._categorized_endpoints[psm] = categorized
            return False

    def _enqueue_priority_refresh(self, psm: str) -> int:
        """Schedule an immediate background refresh for one PSM and return the current round."""
        with self._refresh_condition:
            refresh_round = self._refresh_round_by_psm.get(psm, 0)
            self._priority_psms.add(psm)
            next_priority_refresh_at = (
                time.monotonic() + self._priority_refresh_interval_seconds
            )
            if self._next_priority_refresh_at is None:
                self._next_priority_refresh_at = next_priority_refresh_at
            else:
                self._next_priority_refresh_at = min(
                    self._next_priority_refresh_at,
                    next_priority_refresh_at,
                )
            self._refresh_condition.notify_all()
            return refresh_round

    def _wait_for_refresh_round(
        self,
        psm: str,
        refresh_round: int,
        timeout_seconds: float,
    ) -> None:
        """Block until a newer refresh round is observed for the given PSM."""
        with self._refresh_condition:
            refreshed = self._refresh_condition.wait_for(
                lambda: (
                    self._refresh_round_by_psm.get(psm, 0) > refresh_round
                    or self._stop_event.is_set()
                ),
                timeout=timeout_seconds,
            )
            if self._stop_event.is_set():
                return
            if not refreshed:
                raise TimeoutError(
                    f"Timed out refreshing Consul endpoints for psm '{psm}'"
                )

    def _collect_refresh_psms(self) -> tuple[str, ...]:
        """Collect due priority and regular refresh targets into one deduplicated batch."""
        now = time.monotonic()
        refresh_psms: set[str] = set()
        if (
            self._next_priority_refresh_at is not None
            and now >= self._next_priority_refresh_at
        ):
            refresh_psms.update(self._priority_psms)
            self._priority_psms.clear()
            self._next_priority_refresh_at = None
        if now >= self._next_regular_refresh_at:
            refresh_psms.update(self._seen_psms.keys())
            self._next_regular_refresh_at = now + self._refresh_interval_seconds
        return tuple(refresh_psms)

    def _next_refresh_wait_seconds(self) -> float:
        """Return the shortest wait until either priority or regular refresh becomes due."""
        now = time.monotonic()
        next_due_at = self._next_regular_refresh_at
        if self._next_priority_refresh_at is not None:
            next_due_at = min(next_due_at, self._next_priority_refresh_at)
        return max(next_due_at - now, 0.0)

    def _category_signatures(
        self,
        categorized_endpoints: dict[
            tuple[str, ...],
            dict[tuple[tuple[str, str], ...], list[ConsulInferenceEndpoint]],
        ],
    ) -> dict[
        tuple[tuple[str, str], ...],
        tuple[tuple[str, int, tuple[tuple[str, str], ...]], ...],
    ]:
        signatures: dict[
            tuple[tuple[str, str], ...],
            tuple[tuple[str, int, tuple[tuple[str, str], ...]], ...],
        ] = {}
        for category_entries in categorized_endpoints.values():
            for category_value, endpoints in category_entries.items():
                signatures[category_value] = self._endpoint_signature(endpoints)
        return signatures

    def _endpoint_signature(
        self,
        endpoints: list[ConsulInferenceEndpoint],
    ) -> tuple[tuple[str, int, tuple[tuple[str, str], ...]], ...]:
        return tuple(
            sorted(
                (
                    endpoint.host,
                    endpoint.port,
                    tuple(sorted(endpoint.tags.items())),
                )
                for endpoint in endpoints
            )
        )

    def _normalize_filter_tags(
        self,
        served_model_name: str,
        filter_tags: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
        """Ensure every query includes the default model-name tag filter."""
        normalized = dict(filter_tags or {})
        normalized.setdefault("aibrix_served_model_name", served_model_name)
        return normalized

    def _category_key(
        self,
        filter_tags: Optional[dict[str, str]] = None,
    ) -> tuple[str, ...]:
        """Convert filter tag names into a stable cache bucket identifier."""
        if not filter_tags:
            return _DEFAULT_FILTER_TAGS
        return tuple(sorted(filter_tags))

    def _category_value(
        self,
        filter_tags: dict[str, str],
    ) -> tuple[tuple[str, str], ...]:
        """Convert filter tag key/value pairs into a stable lookup key."""
        return tuple((key, filter_tags[key]) for key in sorted(filter_tags))

    def _categorize_endpoints(
        self,
        endpoints: list[ConsulInferenceEndpoint],
        category_keys: set[tuple[str, ...]],
    ) -> dict[
        tuple[str, ...],
        dict[tuple[tuple[str, str], ...], list[ConsulInferenceEndpoint]],
    ]:
        """Group endpoints by every registered filter-shape so cache reads are O(1)."""
        categorized: dict[
            tuple[str, ...],
            dict[tuple[tuple[str, str], ...], list[ConsulInferenceEndpoint]],
        ] = {}
        for category_key in category_keys:
            entries: dict[
                tuple[tuple[str, str], ...],
                list[ConsulInferenceEndpoint],
            ] = {}
            for endpoint in endpoints:
                category_items = []
                missing_value = False
                for key in category_key:
                    value = self._tag_value(endpoint.tags, key)
                    if value == "":
                        missing_value = True
                        break
                    category_items.append((key, value))
                if missing_value:
                    continue
                entries.setdefault(tuple(category_items), []).append(endpoint)
            categorized[category_key] = entries
        return categorized

    def _parse_endpoint(self, payload: object) -> ConsulInferenceEndpoint:
        """Validate one Consul result item and normalize it into our endpoint model."""
        if not isinstance(payload, dict):
            raise ValueError("Consul endpoint payload must be an object")
        host = payload.get("Host")
        port = payload.get("Port")
        if not isinstance(host, str) or not host:
            raise ValueError("Consul endpoint is missing Host")
        if not isinstance(port, int):
            raise ValueError("Consul endpoint is missing Port")
        return ConsulInferenceEndpoint(
            host=host,
            port=port,
            tags=self._normalize_tags(payload.get("Tags")),
        )

    def _normalize_tags(self, raw_tags: Any) -> dict[str, str]:
        """Normalize Consul tags from dict/list forms into a string-to-string map."""
        if isinstance(raw_tags, dict):
            return {
                str(key): "" if value is None else str(value)
                for key, value in raw_tags.items()
            }
        if isinstance(raw_tags, list):
            tags: dict[str, str] = {}
            for item in raw_tags:
                if not isinstance(item, str):
                    continue
                if ":" not in item:
                    tags[item] = ""
                    continue
                key, value = item.split(":", 1)
                tags[key] = value
            return tags
        return {}

    def _tag_value(self, tags: dict[str, str], key: str) -> str:
        """Read one tag value with case-insensitive fallback for inconsistent inputs."""
        if key in tags:
            return tags[key].strip()
        canonical_key = key.lower()
        for current_key, current_value in tags.items():
            if current_key.lower() == canonical_key:
                return current_value.strip()
        return ""

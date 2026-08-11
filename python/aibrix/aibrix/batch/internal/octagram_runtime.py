# Copyright 2026 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Octagram/TCE-backed execution as a Runtime."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import httpx

import aibrix.batch.constant as constant
from aibrix import envs
from aibrix.batch.client import CapacitySignal, EndpointSource
from aibrix.batch.client.sources import DiscoveryEndpointSource, GatewayEndpointSource
from aibrix.batch.internal.config import REGION_DOMAINS
from aibrix.batch.internal.octagram_renderer import OctagramManifestRenderer
from aibrix.batch.internal.octagram_utils import (
    get_job_name,
    get_workload_name,
    parse_endpoint_cluster,
)
from aibrix.batch.job_driver.runtime import (
    RUNTIME_WAIT_MODE_PROVISION,
    RUNTIME_WAIT_MODE_RECONNECT,
    Endpoint,
    RuntimeBase,
    register_runtime,
)
from aibrix.batch.job_entity import (
    BatchJob,
    BatchJobError,
    BatchJobErrorCode,
    JobRuntimeRef,
)
from aibrix.context import InfrastructureContext
from aibrix.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_NAMESPACE = "default"
_MIN_READY_REPLICAS = 1
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 3600.0
_MIN_REQUEST_TIMEOUT_SECONDS = 1.0
_MODEL_DISCOVERY_WAIT_SLICE_SECONDS = 10.0
_GET_RESPONSE_LOG_SUPPRESSION_MIN_WINDOW_SECONDS = 30.0
_PSM_SERVICE_IDC_MAX_SPLITS = 1
_CONSUL_IDC_ALIASES = {
    "aliyun_va": "maliva",
}
_ENDPOINT_CAPACITY_POLL_INTERVAL_SECONDS = 1.0


@dataclass
class OctagramHandle:
    cluster: str
    namespace: str
    workload_name: str
    model_name: str
    psm: Optional[str]
    base_url: Optional[str]
    replicas: int
    request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS
    source: Optional[EndpointSource] = None
    supplemental: list["OctagramHandle"] = field(default_factory=list)


class _CombinedEndpointSource:
    def __init__(self, sources: list[DiscoveryEndpointSource]) -> None:
        self._sources = sources

    async def channels(self) -> list[Any]:
        channel_groups = await asyncio.gather(
            *(source.channels() for source in self._sources)
        )
        return [channel for group in channel_groups for channel in group]

    async def capacity(self) -> CapacitySignal:
        signals = await asyncio.gather(
            *(source.capacity() for source in self._sources)
        )
        return CapacitySignal(
            count=sum(signal.count for signal in signals),
            version=hash(
                tuple((signal.count, signal.version) for signal in signals)
            ),
        )

    async def wait_capacity_change(
        self,
        previous: CapacitySignal,
    ) -> CapacitySignal:
        while True:
            current = await self.capacity()
            if current != previous:
                return current
            await asyncio.sleep(_ENDPOINT_CAPACITY_POLL_INTERVAL_SECONDS)

    async def report_channel_error(
        self,
        channel_id: str,
        error: Exception,
    ) -> None:
        await asyncio.gather(
            *(
                source.report_channel_error(channel_id, error)
                for source in self._sources
            )
        )

    async def aclose(self) -> None:
        await asyncio.gather(*(source.aclose() for source in self._sources))


class OctagramRuntime(RuntimeBase):
    """Provision a TCE DeploymentWorkload, discover endpoints, then tear it down."""

    provisions = True
    session_liveness_failure_threshold = 1

    def __init__(
        self,
        context: InfrastructureContext,
        renderer: Optional[OctagramManifestRenderer] = None,
        ready_timeout_seconds: Optional[int] = None,
    ) -> None:
        super().__init__(
            context,
            ready_timeout_seconds
            or int(getattr(envs, "CONSUL_BATCH_DISCOVERY_TIMEOUT", 900)),
        )
        self._renderer = renderer or self._build_renderer(context)
        self._httpx_client_wrapper = context.httpx_client_wrapper
        self._gateway_domain = self._resolve_gateway_domain()
        # TODO: put env in internal.env, see envs usage convection.
        self._workload_not_found_grace_seconds = (
            envs.OCTAGRAM_WORKLOAD_NOT_FOUND_GRACE_SECONDS
        )
        self._active_handle: Optional[OctagramHandle] = None
        self._last_get_response_log_by_request: dict[
            str, tuple[str, dict[str, Any], float]
        ] = {}

    @staticmethod
    def _build_renderer(context: InfrastructureContext) -> OctagramManifestRenderer:
        return OctagramManifestRenderer(
            context.template_registry, context.profile_registry
        )

    @staticmethod
    def _all_workloads(handle: OctagramHandle) -> list[OctagramHandle]:
        return [handle, *handle.supplemental]

    @staticmethod
    def _handle_payload(handle: OctagramHandle) -> dict[str, Any]:
        return {
            "cluster": handle.cluster,
            "namespace": handle.namespace,
            "workloadName": handle.workload_name,
            "modelName": handle.model_name,
            "psm": handle.psm,
            "baseUrl": handle.base_url,
            "replicas": handle.replicas,
        }

    @staticmethod
    def _handle_from_payload(
        payload: dict[str, Any],
        request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> Optional[OctagramHandle]:
        cluster = payload.get("cluster")
        namespace = payload.get("namespace")
        workload_name = payload.get("workloadName")
        model_name = payload.get("modelName")
        psm = payload.get("psm")
        base_url = payload.get("baseUrl")
        replicas = payload.get("replicas")
        if not (
            isinstance(cluster, str)
            and isinstance(namespace, str)
            and namespace
            and isinstance(workload_name, str)
            and workload_name
            and isinstance(model_name, str)
            and model_name
            and (psm is None or isinstance(psm, str))
            and (base_url is None or isinstance(base_url, str))
            and isinstance(replicas, int)
        ):
            return None
        return OctagramHandle(
            cluster=cluster,
            namespace=namespace,
            workload_name=workload_name,
            model_name=model_name,
            psm=psm,
            base_url=base_url,
            replicas=replicas,
            request_timeout_seconds=request_timeout_seconds,
        )

    def _get_runtime_key(self, job: BatchJob) -> str:
        del job
        return "tce"

    def _get_runtime_owner_ref(self, job: BatchJob) -> Optional[str]:
        del job
        if self._active_handle is None:
            return None
        return (
            f"{self._active_handle.cluster}:{self._active_handle.namespace}:"
            f"{self._active_handle.workload_name}"
        )

    def _get_runtime_reconnect_payload(
        self,
        job: BatchJob,
    ) -> Optional[dict[str, Any]]:
        payload = super()._get_runtime_reconnect_payload(job) or {}
        if self._active_handle is None:
            return None if not payload else payload
        # Keep the primary handle in the legacy top-level shape. Older MDS
        # versions ignore supplementalWorkloads but can still deserialize and
        # reconnect the primary workload after a binary rollback.
        payload.update(self._handle_payload(self._active_handle))
        if self._active_handle.supplemental:
            payload["supplementalWorkloads"] = [
                self._handle_payload(handle)
                for handle in self._active_handle.supplemental
            ]
        return payload

    async def _load_handle(
        self, job: BatchJob, job_id: str, execution: JobRuntimeRef
    ) -> OctagramHandle | None:
        reconnect_payload = execution.reconnect_payload or {}
        request_timeout_seconds = self._resolve_request_timeout_seconds(job)
        handle = self._handle_from_payload(
            reconnect_payload,
            request_timeout_seconds=request_timeout_seconds,
        )
        if handle is None:
            return None

        supplemental_payloads = reconnect_payload.get("supplementalWorkloads", [])
        if not isinstance(supplemental_payloads, list):
            return None
        for payload in supplemental_payloads:
            if not isinstance(payload, dict):
                return None
            supplemental = self._handle_from_payload(
                payload,
                request_timeout_seconds=request_timeout_seconds,
            )
            if supplemental is None:
                return None
            handle.supplemental.append(supplemental)

        self._active_handle = handle
        logger.info(
            "Loaded Octagram runtime handle for batch job",
            job_id=job_id,
            cluster=handle.cluster,
            namespace=handle.namespace,
            workload=handle.workload_name,
            supplemental_workloads=[
                supplemental.workload_name for supplemental in handle.supplemental
            ],
        )  # type: ignore[call-arg]
        return handle

    async def _provision(self, job: BatchJob, job_id: str) -> OctagramHandle:
        if job.job_id is None:
            raise ValueError("job_id is required")
        if job.spec.aibrix is None:
            raise ValueError("OctagramRuntime requires spec.aibrix")

        resource_allocation = job.spec.aibrix.resource_allocation
        resource_details = (
            resource_allocation.resource_details if resource_allocation else None
        )
        if not resource_details:
            raise ValueError(
                "OctagramRuntime requires aibrix.resource_allocation.resource_details"
            )
        if len(resource_details) == 1:
            return await self._provision_single(job, job_id, resource_details[0])

        job_name = get_job_name(job)
        direct_base_url = self._resolve_direct_base_url(job)
        if direct_base_url is not None:
            raise ValueError(
                "OctagramRuntime cannot combine multiple allocations with a direct endpoint"
            )

        rendered_workloads: list[tuple[OctagramHandle, dict[str, Any], Any]] = []
        primary_handle: Optional[OctagramHandle] = None
        served_model_name: Optional[str] = None
        request_timeout_seconds = self._resolve_request_timeout_seconds(job)
        for index, resource_detail in enumerate(resource_details):
            workload_job_name = get_workload_name(job_name, index)
            workload = self._renderer.render(
                job.job_id,
                workload_job_name,
                job.spec,
                resource_detail,
                served_model_name=served_model_name,
            )
            _, idc, physical_cluster, _ = parse_endpoint_cluster(
                resource_detail.endpoint_cluster
            )
            cluster = f"{physical_cluster}-{idc}"
            namespace = workload["metadata"].get("namespace", _DEFAULT_NAMESPACE)
            workload_name = workload["metadata"]["name"]
            model_name = workload["metadata"]["labels"]["model.aibrix.ai/name"]
            handle = OctagramHandle(
                cluster=cluster,
                namespace=namespace,
                workload_name=workload_name,
                model_name=model_name,
                psm=(
                    self._resolve_psm(workload)
                    if index == 0
                    else self._resolve_psm(workload, idc)
                ),
                base_url=direct_base_url if index == 0 else None,
                replicas=int(
                    workload["spec"]["deployStrategy"].get("replicas", 1)
                ),
                request_timeout_seconds=request_timeout_seconds,
            )
            if primary_handle is None:
                primary_handle = handle
                served_model_name = model_name
            else:
                primary_handle.supplemental.append(handle)
            rendered_workloads.append((handle, workload, resource_detail))

        assert primary_handle is not None

        if self._renderer.template is not None:
            self._ready_timeout_seconds = (
                self._renderer.template.spec.engine.ready_timeout_seconds
            )
            logger.debug(
                "ready_timeout_seconds override from template",
                job_id=job.job_id,
                ready_timeout_seconds=self._ready_timeout_seconds,
            )  # type: ignore[call-arg]

        self._active_handle = primary_handle
        applied_workloads: list[OctagramHandle] = []
        try:
            for handle, workload, resource_detail in rendered_workloads:
                await self._apply_workload(
                    handle.cluster,
                    handle.namespace,
                    workload,
                )
                applied_workloads.append(handle)
                logger.info(
                    "Provisioned Octagram workload for batch job",
                    job_id=job_id,
                    cluster=handle.cluster,
                    namespace=handle.namespace,
                    workload=handle.workload_name,
                    model_name=handle.model_name,
                    psm=handle.psm,
                    replicas=handle.replicas,
                    resource_pool_name=resource_detail.resource_pool_name,
                )  # type: ignore[call-arg]
        except BaseException:
            for applied in reversed(applied_workloads):
                try:
                    await self._delete_workload(applied)
                except BaseException as cleanup_error:
                    logger.warning(
                        "Failed to roll back partially provisioned Octagram workload",
                        job_id=job_id,
                        cluster=applied.cluster,
                        namespace=applied.namespace,
                        workload=applied.workload_name,
                        error=str(cleanup_error),
                    )  # type: ignore[call-arg]
            self._active_handle = None
            raise
        return primary_handle

    async def _provision_single(
        self,
        job: BatchJob,
        job_id: str,
        resource_detail: Any,
    ) -> OctagramHandle:
        job_name = get_job_name(job)
        workload = self._renderer.render(
            job.job_id, job_name, job.spec, resource_detail
        )
        _, idc, physical_cluster, _ = parse_endpoint_cluster(
            resource_detail.endpoint_cluster
        )
        cluster = f"{physical_cluster}-{idc}"
        namespace = workload["metadata"].get("namespace", _DEFAULT_NAMESPACE)
        workload_name = workload["metadata"]["name"]
        model_name = workload["metadata"]["labels"]["model.aibrix.ai/name"]
        psm = self._resolve_psm(workload)
        replicas = int(workload["spec"]["deployStrategy"].get("replicas", 1))

        if self._renderer.template is not None:
            self._ready_timeout_seconds = (
                self._renderer.template.spec.engine.ready_timeout_seconds
            )
            logger.debug(
                "ready_timeout_seconds override from template",
                job_id=job.job_id,
                ready_timeout_seconds=self._ready_timeout_seconds,
            )  # type: ignore[call-arg]

        handle = OctagramHandle(
            cluster=cluster,
            namespace=namespace,
            workload_name=workload_name,
            model_name=model_name,
            psm=psm,
            base_url=self._resolve_direct_base_url(job),
            replicas=replicas,
            request_timeout_seconds=self._resolve_request_timeout_seconds(job),
        )
        self._active_handle = handle
        await self._apply_workload(cluster, namespace, workload)
        logger.info(
            "Provisioned Octagram workload for batch job",
            job_id=job_id,
            cluster=cluster,
            namespace=namespace,
            workload=workload_name,
            model_name=model_name,
            psm=psm,
            replicas=replicas,
        )  # type: ignore[call-arg]
        return handle

    async def _wait_ready(
        self,
        handle: OctagramHandle,
        wait_mode: str = RUNTIME_WAIT_MODE_PROVISION,
    ) -> None:
        if not handle.supplemental:
            await self._wait_for_workload(
                cluster=handle.cluster,
                namespace=handle.namespace,
                workload_name=handle.workload_name,
                replicas=handle.replicas,
                wait_mode=wait_mode,
            )
            if handle.base_url is None:
                await self._wait_for_model_discoverable(handle)
            return

        await asyncio.gather(
            *(
                self._wait_for_workload(
                    cluster=workload.cluster,
                    namespace=workload.namespace,
                    workload_name=workload.workload_name,
                    replicas=workload.replicas,
                    wait_mode=wait_mode,
                )
                for workload in self._all_workloads(handle)
            )
        )
        if handle.base_url is None:
            await asyncio.gather(
                *(
                    self._wait_for_model_discoverable(workload)
                    for workload in self._all_workloads(handle)
                )
            )

    async def _check_liveness(
        self, handle: OctagramHandle, reason: str = "unspecified"
    ) -> None:
        if not handle.supplemental:
            await self._wait_for_workload(
                cluster=handle.cluster,
                namespace=handle.namespace,
                workload_name=handle.workload_name,
                replicas=handle.replicas,
                wait_mode=RUNTIME_WAIT_MODE_RECONNECT,
                request_reason=f"check_liveness:{reason}",
            )
            return

        await asyncio.gather(
            *(
                self._wait_for_workload(
                    cluster=workload.cluster,
                    namespace=workload.namespace,
                    workload_name=workload.workload_name,
                    replicas=workload.replicas,
                    wait_mode=RUNTIME_WAIT_MODE_RECONNECT,
                    request_reason=f"check_liveness:{reason}",
                )
                for workload in self._all_workloads(handle)
            )
        )

    async def _connect(self, handle: OctagramHandle) -> Endpoint:
        if handle.base_url is not None:
            handle.source = GatewayEndpointSource(
                handle.base_url,
                capacity=max(handle.replicas, 1),
                timeout=handle.request_timeout_seconds,
            )
        else:
            model_discovery = self._context.model_discovery
            if model_discovery is None:
                raise BatchJobError(
                    code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
                    message="Octagram consul discovery service is not configured",
                )
            if not handle.supplemental:
                handle.source = DiscoveryEndpointSource(
                    model_discovery,
                    handle.model_name,
                    service_id=handle.psm,
                    timeout=handle.request_timeout_seconds,
                )
                return Endpoint(
                    source=cast(Optional[EndpointSource], handle.source),
                    model_name=handle.model_name,
                )

            discovery_sources: list[DiscoveryEndpointSource] = []
            seen_discovery_keys: set[tuple[str, Optional[str]]] = set()
            for workload in self._all_workloads(handle):
                discovery_key = (workload.model_name, workload.psm)
                if discovery_key in seen_discovery_keys:
                    continue
                seen_discovery_keys.add(discovery_key)
                discovery_sources.append(
                    DiscoveryEndpointSource(
                        model_discovery,
                        workload.model_name,
                        service_id=workload.psm,
                        timeout=workload.request_timeout_seconds,
                    )
                )
            handle.source = (
                discovery_sources[0]
                if len(discovery_sources) == 1
                else _CombinedEndpointSource(discovery_sources)
            )
        return Endpoint(
            source=cast(Optional[EndpointSource], handle.source),
            model_name=handle.model_name,
        )

    async def _teardown(self, handle: Optional[OctagramHandle]) -> None:
        if handle is not None and not handle.supplemental:
            try:
                await self._delete_workload(handle)
            finally:
                if handle.source is not None:
                    await handle.source.aclose()
            self._active_handle = None
            return

        if handle is None:
            self._active_handle = None
            return

        teardown_error: Optional[BaseException] = None
        try:
            for workload in reversed(self._all_workloads(handle)):
                try:
                    await self._delete_workload(workload)
                except BaseException as ex:
                    if teardown_error is None:
                        teardown_error = ex
                    logger.warning(
                        "Octagram workload teardown failed; continuing cleanup",
                        job_id=self._active_job_id,
                        cluster=workload.cluster,
                        namespace=workload.namespace,
                        workload=workload.workload_name,
                        error=str(ex),
                    )  # type: ignore[call-arg]
        finally:
            if handle.source is not None:
                await handle.source.aclose()
            self._active_handle = None
        if teardown_error is not None:
            raise teardown_error

    async def _wait_for_model_discoverable(self, handle: OctagramHandle) -> None:
        model_discovery = self._context.model_discovery
        if model_discovery is None:
            raise BatchJobError(
                code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
                message="Octagram consul discovery service is not configured",
            )
        try:
            logger.info(
                "Wait for model endpoints to be discoverable",
                job_id=self._active_job_id,
                model_name=handle.model_name,
                deploy_name=handle.workload_name,
                service_id=handle.psm,
                timeout_seconds=self._ready_timeout_seconds,
            )  # type: ignore[call-arg]
            loop = asyncio.get_running_loop()
            deadline = loop.time() + float(self._ready_timeout_seconds)
            while True:
                if self._stop_requested.is_set():
                    raise asyncio.CancelledError
                await self._check_liveness(
                    handle,
                    reason="wait for model discoverable",
                )
                remaining_timeout_seconds = deadline - loop.time()
                if remaining_timeout_seconds <= 0:
                    raise TimeoutError(
                        "Timed out waiting for Consul endpoints for model "
                        f"'{handle.model_name}'"
                    )
                wait_slice_seconds = min(
                    remaining_timeout_seconds, _MODEL_DISCOVERY_WAIT_SLICE_SECONDS
                )
                try:
                    await model_discovery.wait_for_model_endpoints(
                        served_model_name=handle.model_name,
                        timeout_seconds=wait_slice_seconds,
                        service_id=handle.psm,
                        lookup_timeout_seconds=wait_slice_seconds,
                        poll_interval_seconds=wait_slice_seconds,
                    )
                    return
                except TimeoutError as ex:
                    if loop.time() >= deadline:
                        raise TimeoutError(
                            "Timed out waiting for Consul endpoints for model "
                            f"'{handle.model_name}'"
                        ) from ex
        except TimeoutError as ex:
            raise BatchJobError(
                code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
                message=str(ex),
            ) from ex

    def _resolve_direct_base_url(self, job: BatchJob) -> Optional[str]:
        if job.spec.opts and constant.BATCH_OPTS_RESOURCE_ENDPOINT in job.spec.opts:
            return str(job.spec.opts[constant.BATCH_OPTS_RESOURCE_ENDPOINT])
        return None

    def _resolve_request_timeout_seconds(self, job: BatchJob) -> float:
        client = job.spec.aibrix.client if job.spec.aibrix else None
        client_timeout = getattr(client, "request_timeout_seconds", None)
        if client_timeout is not None:
            return max(
                float(client_timeout),
                _MIN_REQUEST_TIMEOUT_SECONDS,
            )

        try:
            profile_ref = job.spec.aibrix.profile if job.spec.aibrix else None
            profile = self._renderer._resolve_profile(profile_ref)
            return max(
                float(profile.spec.scheduling.request_timeout_seconds),
                _MIN_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning(
                "Failed to resolve Octagram request timeout; using default",
                job_id=job.job_id,
                default_timeout_seconds=_DEFAULT_REQUEST_TIMEOUT_SECONDS,
                error=str(exc),
            )  # type: ignore[call-arg]
            return _DEFAULT_REQUEST_TIMEOUT_SECONDS

    def _resolve_psm(
        self,
        workload: dict[str, Any],
        idc_name: Optional[str] = None,
    ) -> Optional[str]:
        labels = workload["metadata"].get("labels", {})
        psm = labels.get("psm")
        if not psm:
            return None
        return self._normalize_psm_service_idc(
            psm,
            idc_name
            if idc_name is not None
            else getattr(self._renderer, "idc_name", ""),
            replace_existing=idc_name is not None,
        )

    def _normalize_consul_idc_name(self, idc_name: str) -> str:
        normalized = idc_name.lower()
        for source, target in _CONSUL_IDC_ALIASES.items():
            if normalized == source:
                return target
            prefix = f"{source}."
            if normalized.startswith(prefix):
                return f"{target}.{normalized[len(prefix) :]}"
        return normalized

    def _normalize_psm_service_idc(
        self,
        psm: str,
        idc_name: str,
        *,
        replace_existing: bool = False,
    ) -> str:
        if ".service." in psm:
            prefix, existing_idc = psm.rsplit(".service.", _PSM_SERVICE_IDC_MAX_SPLITS)
            normalized_idc = self._normalize_consul_idc_name(
                idc_name if replace_existing and idc_name else existing_idc
            )
            return f"{prefix}.service.{normalized_idc}"

        normalized_idc = self._normalize_consul_idc_name(idc_name)
        if not normalized_idc:
            return psm
        if psm.endswith(".service"):
            return f"{psm}.{normalized_idc}"
        return f"{psm}.service.{normalized_idc}"

    async def _apply_workload(
        self, cluster: str, namespace: str, workload: dict[str, Any]
    ) -> None:
        try:
            response = await self._octagram_request(
                "POST",
                self._workload_path(cluster, namespace, workload["metadata"]["name"]),
                reason="create_workload",
                json=workload,
            )
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code == 409:
                logger.info(
                    "Octagram workload create returned 409; treating as already exists",
                    cluster=cluster,
                    namespace=namespace,
                    workload=workload["metadata"]["name"],
                )  # type: ignore[call-arg]
                return
            raise BatchJobError(
                code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
                message=(
                    "Octagram workload create failed "
                    f"({ex.response.status_code}): {ex.response.text}"
                ),
            ) from ex
        if response.get("error"):
            raise BatchJobError(
                code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
                message=f"Octagram workload create failed: {response['error']}",
            )

    async def _wait_for_workload(
        self,
        cluster: str,
        namespace: str,
        workload_name: str,
        replicas: int,
        wait_mode: str = RUNTIME_WAIT_MODE_RECONNECT,
        request_reason: Optional[str] = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(self._ready_timeout_seconds)
        missing_deadline: Optional[float] = None
        if wait_mode == RUNTIME_WAIT_MODE_PROVISION:
            missing_deadline = min(
                deadline,
                loop.time() + float(self._workload_not_found_grace_seconds),
            )
        while True:
            if self._stop_requested.is_set():
                raise asyncio.CancelledError
            try:
                workload = await self._octagram_request(
                    "GET",
                    self._workload_path(cluster, namespace, workload_name),
                    reason=request_reason or f"wait_for_workload:{wait_mode}",
                )
            except httpx.HTTPStatusError as ex:
                # Octagram has read-after-write inconsistency problem.
                # So we treat 404 as a temporary error and simply ignore for a timeout.
                if (
                    ex.response.status_code == 404
                    and wait_mode == RUNTIME_WAIT_MODE_PROVISION
                    and missing_deadline is not None
                    and loop.time() < missing_deadline
                ):
                    logger.info(
                        "Octagram workload status check returned 404 during provision; retrying",
                        cluster=cluster,
                        namespace=namespace,
                        workload=workload_name,
                        grace_seconds=self._workload_not_found_grace_seconds,
                    )  # type: ignore[call-arg]
                    await asyncio.sleep(1)
                    continue

                error_code = BatchJobErrorCode.RESOURCE_CREATION_ERROR
                if ex.response.status_code == 404:
                    error_code = BatchJobErrorCode.RESOURCE_NOTFOUND_ERROR
                raise BatchJobError(
                    code=error_code,
                    message=(
                        "Octagram workload status check failed "
                        f"({ex.response.status_code}): {ex.response.text}"
                    ),
                ) from ex
            workload_data = self._unwrap_octagram_data(workload)
            status = workload_data.get("status", {})
            phase = status.get("phase")
            if phase == "failed":
                raise BatchJobError(
                    code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
                    message=f"Octagram workload '{workload_name}' failed",
                )
            replicas_statuses = status.get("replicasStatuses") or [{}]
            if phase == "synced":
                # Federation webhook does not ask for actual 0 availability, this works now.
                if replicas == 0:
                    return

                available = max(
                    int(item.get("available") or 0) for item in replicas_statuses
                )
                required_available = min(replicas, _MIN_READY_REPLICAS)
                if replicas > 0 and available >= required_available:
                    # A positive workload is ready once the configured minimum is
                    # available, capped by the requested replica count.
                    return
            if asyncio.get_running_loop().time() >= deadline:
                raise BatchJobError(
                    code=BatchJobErrorCode.RESOURCE_CREATION_ERROR
                    if replicas > 0
                    else BatchJobErrorCode.RESOURCE_DELETION_ERROR,
                    message=(
                        f"Timed out waiting for octagram workload '{workload_name}' "
                        f"to become ready and replica count {replicas}"
                    ),
                )
            await asyncio.sleep(1)

    async def _delete_workload(self, handle: OctagramHandle) -> None:
        workload_path = self._workload_path(
            handle.cluster, handle.namespace, handle.workload_name
        )

        # Try direct deletion first. If Octagram rejects it with a non-404 HTTP
        # error, scale the workload down to zero, wait for it to settle, then
        # retry the delete once.
        for retry_after_scale in (False, True):
            try:
                response = await self._octagram_request(
                    "DELETE",
                    workload_path,
                    reason="delete_workload",
                )
            except httpx.HTTPStatusError as ex:
                if ex.response.status_code == 404:
                    logger.info(
                        "Octagram workload delete returned 404; treating as already deleted",
                        cluster=handle.cluster,
                        namespace=handle.namespace,
                        workload=handle.workload_name,
                        retried_after_scale=retry_after_scale,
                    )  # type: ignore[call-arg]
                    return
                if retry_after_scale:
                    raise BatchJobError(
                        code=BatchJobErrorCode.RESOURCE_DELETION_ERROR,
                        message=(
                            "Octagram workload delete failed after scale to zero "
                            f"({ex.response.status_code}): {ex.response.text}"
                        ),
                    ) from ex
                # Federation have webhook blocks non-zero replica workload deletion, error msg:
                # Error from server: admission webhook "scalabledeletion.kubeguardian.byted.org" denied the request:
                # object's replicas is 1, not safe to delete, please scale to zero before deletion
                # We try _scale_workload_to_zero first.
                await self._scale_workload_to_zero(handle, ex)
                continue

            if response.get("error"):
                if (
                    response.get("code") == 404
                    and "not found" in str(response["error"]).lower()
                ):
                    logger.info(
                        "Octagram workload delete returned payload not found; treating as already deleted",
                        cluster=handle.cluster,
                        namespace=handle.namespace,
                        workload=handle.workload_name,
                        retried_after_scale=retry_after_scale,
                    )  # type: ignore[call-arg]
                    return
                raise BatchJobError(
                    code=BatchJobErrorCode.RESOURCE_DELETION_ERROR,
                    message=f"Octagram workload delete failed: {response['error']}",
                )
            return

    async def _scale_workload_to_zero(
        self, handle: OctagramHandle, delete_error: httpx.HTTPStatusError
    ) -> None:
        delete_context = (
            "Octagram workload delete failed "
            f"({delete_error.response.status_code}): {delete_error.response.text}"
        )

        logger.warning(
            "Octagram workload delete failed, falling back to scale-to-zero",
            cluster=handle.cluster,
            namespace=handle.namespace,
            workload=handle.workload_name,
            status_code=delete_error.response.status_code,
            error=delete_error.response.text,
        )  # type: ignore[call-arg]
        try:
            response = await self._octagram_request(
                "PATCH",
                self._workload_scale_path(
                    handle.cluster,
                    handle.namespace,
                    handle.workload_name,
                    replicas=0,
                ),
                reason="scale_workload_to_zero",
            )
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code == 404:
                return
            raise BatchJobError(
                code=BatchJobErrorCode.RESOURCE_DELETION_ERROR,
                message=(
                    f"{delete_context}; fallback scale to zero failed "
                    f"({ex.response.status_code}): {ex.response.text}"
                ),
            ) from ex
        if response.get("error"):
            raise BatchJobError(
                code=BatchJobErrorCode.RESOURCE_DELETION_ERROR,
                message=(
                    f"{delete_context}; fallback scale to zero failed: "
                    f"{response['error']}"
                ),
            )
        try:
            # The scale endpoint is asynchronous. Wait until Octagram reports the
            # workload as synced at zero replicas before retrying delete.
            await self._wait_for_workload(
                cluster=handle.cluster,
                namespace=handle.namespace,
                workload_name=handle.workload_name,
                replicas=0,
                request_reason="wait for workload synced at zero replicas",
            )
        except BatchJobError as ex:
            raise BatchJobError(
                code=BatchJobErrorCode.RESOURCE_DELETION_ERROR,
                message=(
                    f"{delete_context}; fallback scale to zero did not "
                    f"reach synced: {ex.message}"
                ),
            ) from ex

    def _resolve_gateway_domain(self) -> str:
        configured = getattr(envs, "OCTAGRAM_GATEWAY_DOMAIN", None)
        if isinstance(configured, str) and configured:
            return configured.rstrip("/")
        region = getattr(envs, "REGION", "CN").upper().replace("-", "").replace("_", "")
        if region not in REGION_DOMAINS:
            raise ValueError(
                f"Unsupported REGION '{region}', expected one of {sorted(REGION_DOMAINS)}"
            )
        return REGION_DOMAINS[region].octagram

    def _workload_path(self, cluster: str, namespace: str, workload_name: str) -> str:
        return (
            f"{self._gateway_domain}/api/v1/clusters/{cluster}/namespaces/"
            f"{namespace}/deploymentworkloads/{workload_name}"
        )

    def _workload_scale_path(
        self,
        cluster: str,
        namespace: str,
        workload_name: str,
        replicas: int,
    ) -> str:
        return (
            f"{self._workload_path(cluster, namespace, workload_name)}"
            f"/scale?replicas={replicas}"
        )

    async def _octagram_request(
        self,
        method: str,
        url: str,
        *,
        reason: str = "unspecified",
        response_log_fields: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response: Optional[httpx.Response] = None
        payload: Optional[dict[str, Any]] = None
        error: Optional[Exception] = None
        if response_log_fields is None:
            response_log_fields = (
                ["status.phase", "status.replicasStatuses"]
                if method.upper() == "GET"
                else []
            )
        try:
            if (
                self._httpx_client_wrapper is not None
                and self._httpx_client_wrapper.async_client is not None
            ):
                response = await self._httpx_client_wrapper.async_client.request(
                    method, url, **kwargs
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.request(method, url, **kwargs)

            assert response is not None
            response.raise_for_status()
            payload = response.json()
        except Exception as ex:
            error = ex
            raise
        finally:
            log_data: dict[str, Any] = {
                "method": method,
                "url": url,
                "reason": reason,
                "status_code": response.status_code if response is not None else None,
            }
            if response is not None:
                if payload is None:
                    try:
                        parsed = response.json()
                        if isinstance(parsed, dict):
                            payload = parsed
                    except ValueError:
                        payload = None
            payload_data: dict[str, Any] = {}
            if payload is not None:
                unwrapped_payload = self._unwrap_octagram_data(payload)
                if isinstance(unwrapped_payload, dict):
                    payload_data = unwrapped_payload

            for field in response_log_fields:
                log_data[field.rsplit(".", 1)[-1]] = self._get_response_log_field(
                    payload_data, field
                )

            if error is not None:
                log_data["error"] = str(error)
                if response is not None:
                    log_data["response_text"] = response.text
            should_log = self._should_log_get_response(method, url, log_data)
            if error is None and should_log:
                logger.info("octagram response", **log_data)  # type: ignore[call-arg]
            elif error is not None and should_log:
                logger.error("octagram response", **log_data)  # type: ignore[call-arg]
        return payload or {}

    @staticmethod
    def _unwrap_octagram_data(response: dict[str, Any]) -> dict[str, Any]:
        data = response.get("data")
        if isinstance(data, dict):
            return data
        return response

    @staticmethod
    def _get_response_log_field(payload_data: dict[str, Any], field_path: str) -> Any:
        current: Any = payload_data
        for part in field_path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    def _should_log_get_response(
        self,
        method: str,
        url: str,
        log_data: dict[str, Any],
    ) -> bool:
        if method.upper() != "GET":
            return True
        suppression_window_s = max(
            self.session_liveness_check_interval_s,
            _GET_RESPONSE_LOG_SUPPRESSION_MIN_WINDOW_SECONDS,
        )
        now = time.monotonic()
        reason = str(log_data.get("reason") or "unspecified")
        last_entry = self._last_get_response_log_by_request.get(url)
        self._last_get_response_log_by_request[url] = (reason, dict(log_data), now)
        if last_entry is None:
            return True
        last_reason, last_log_data, last_logged_at = last_entry
        if reason != last_reason:
            return True
        if log_data != last_log_data:
            return True
        return (now - last_logged_at) >= suppression_window_s


register_runtime(
    "tce",
    lambda *, context, **_: OctagramRuntime(context),
)
register_runtime(
    "Octagram",
    lambda *, context, **_: OctagramRuntime(context),
)

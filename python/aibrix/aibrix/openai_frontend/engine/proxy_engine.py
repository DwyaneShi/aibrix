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

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aibrix.openai_frontend.common.artifact_fetcher import (
    ArtifactFetcher,
    DefaultArtifactFetcher,
    FetchResult,
)
from aibrix.openai_frontend.engine.engine import LLMEngine
from aibrix.openai_frontend.proxy.engine_spec import EngineSpec
from aibrix.openai_frontend.proxy.request_inspector import (
    ChatCompletionInspector,
    CompletionInspector,
    EmbeddingInspector,
    RequestInspector,
)
from aibrix.openai_frontend.proxy.request_rewriter import RequestRewriter
from aibrix.openai_frontend.proxy.subprocess_engine import SubprocessEngine
from aibrix.openai_frontend.schemas.openai import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    CreateEmbeddingRequest,
    CreateEmbeddingResponse,
    Model,
)
from aibrix.openai_frontend.utils.utils import (
    ClientError,
    ServerError,
    make_prefix_formatter,
)

HEALTH_CHECK_INTERVAL_SECONDS = 10.0
WATCHDOG_STUCK_THRESHOLD_SECONDS = 300.0
HEARTBEAT_INTERVAL_CYCLES = 360  # ~60s at 10s interval


logger = logging.getLogger(__name__)


class ProxyEngine(LLMEngine):
    def __init__(
        self,
        engine_cmd: List[str],
        engine_port: int = 8001,
        engine_host: str = "127.0.0.1",
        engine_env: Optional[Dict[str, str]] = None,
        spec: Optional[EngineSpec] = None,
        artifact_fetcher: Optional[ArtifactFetcher] = None,
        credentials_map: Optional[Dict[str, Dict]] = None,
        chat_inspector: Optional[RequestInspector] = None,
        completion_inspector: Optional[RequestInspector] = None,
        embedding_inspector: Optional[RequestInspector] = None,
    ):
        self._spec = spec or EngineSpec(name="default")
        self._subprocess = SubprocessEngine(
            cmd=engine_cmd,
            port=engine_port,
            host=engine_host,
            env=engine_env,
            spec=self._spec,
        )

        self._fetcher = artifact_fetcher or DefaultArtifactFetcher()
        self._chat_inspector = chat_inspector or ChatCompletionInspector()
        self._completion_inspector = completion_inspector or CompletionInspector()
        self._embedding_inspector = embedding_inspector or EmbeddingInspector()
        self._credentials_map = credentials_map or {}

        self._ready: bool = False
        self._stopping: bool = False

        # Add [proxy] prefix to all logger output via formatter
        for h in logging.getLogger().handlers:
            h.setFormatter(make_prefix_formatter("[proxy]"))

        # Request tracking: req_id -> start timestamp
        self._in_flight: Dict[str, float] = {}

        # Background health monitor
        self._health_monitor_task: Optional[asyncio.Task] = None

        # Metrics
        self._metrics_registry = CollectorRegistry()
        self._fetch_total = Counter(
            "proxy_fetch_total",
            "Total artifact fetch attempts",
            registry=self._metrics_registry,
        )
        self._fetch_errors = Counter(
            "proxy_fetch_errors",
            "Total artifact fetch errors",
            registry=self._metrics_registry,
        )
        self._fetch_seconds = Gauge(
            "proxy_fetch_seconds_total",
            "Cumulative seconds spent fetching artifacts",
            registry=self._metrics_registry,
        )
        self._requests_total = Counter(
            "proxy_requests_total",
            "Total requests served",
            registry=self._metrics_registry,
        )
        self._request_duration_seconds = Histogram(
            "proxy_request_duration_seconds",
            "Request duration in seconds",
            buckets=(1, 5, 10, 30, 60, 120, 300, 600),
            registry=self._metrics_registry,
        )
        self._in_flight_gauge = Gauge(
            "proxy_in_flight_requests",
            "Number of in-flight requests",
            registry=self._metrics_registry,
        )

    async def start(self) -> None:
        self._stopping = False
        self._subprocess.spawn()
        await self._subprocess.wait_ready()
        await self._subprocess.start_client()
        self._ready = True
        await self._ensure_health_monitor()
        logger.info("ProxyEngine started; inner engine is ready")

    def stop(self) -> None:
        self._stopping = True

        task = self._health_monitor_task
        if task is not None:
            task.cancel()
            self._health_monitor_task = None

        for req_id, start in list(self._in_flight.items()):
            logger.warning(
                "Failing in-flight req_id=%s during shutdown (pending %.1fs)",
                req_id,
                time.monotonic() - start,
            )
        self._in_flight.clear()

        self._subprocess.shutdown()
        logger.info("ProxyEngine stopped")

    # ------------------------------------------------------------------
    # Health monitor
    # ------------------------------------------------------------------

    async def _ensure_health_monitor(self) -> None:
        if self._health_monitor_task is not None:
            return
        self._health_monitor_task = asyncio.create_task(
            self._health_monitor_loop_guarded()
        )
        self._health_monitor_task.add_done_callback(self._on_health_monitor_done)

    def _on_health_monitor_done(self, task: asyncio.Task) -> None:
        self._health_monitor_task = None
        if self._stopping:
            return
        if task.cancelled():
            logger.warning("_health_monitor_loop cancelled unexpectedly")
        else:
            exc = task.exception()
            if exc is not None:
                logger.critical("_health_monitor_loop crashed: %s", exc, exc_info=exc)
            else:
                logger.critical("_health_monitor_loop exited unexpectedly")
        asyncio.ensure_future(self._restart_health_monitor())

    async def _restart_health_monitor(self) -> None:
        if self._stopping:
            return
        await asyncio.sleep(1.0)
        if self._stopping:
            return
        logger.warning("Restarting _health_monitor_loop after crash")
        await self._ensure_health_monitor()

    async def _health_monitor_loop(self) -> None:
        cycle = 0
        while True:
            if self._stopping:
                return

            cycle += 1

            # Watchdog: log in-flight requests stuck > threshold
            now = time.monotonic()
            for req_id, start in list(self._in_flight.items()):
                elapsed = now - start
                if elapsed > WATCHDOG_STUCK_THRESHOLD_SECONDS:
                    logger.warning(
                        "[watchdog] req_id=%s has been pending %.0fs",
                        req_id,
                        elapsed,
                    )

            # Heartbeat logging
            if cycle % HEARTBEAT_INTERVAL_CYCLES == 0:
                pid = self._subprocess.pid
                alive = pid is not None
                logger.info(
                    "[heartbeat] subprocess_pid=%s alive=%s in_flight=%d",
                    pid,
                    alive,
                    len(self._in_flight),
                )

            await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)

    async def _health_monitor_loop_guarded(self) -> None:
        try:
            await self._health_monitor_loop()
        except asyncio.CancelledError:
            if not self._stopping:
                logger.warning("_health_monitor_loop cancelled unexpectedly")
            raise
        except BaseException:
            logger.critical(
                "_health_monitor_loop exiting due to BaseException",
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # LLMEngine protocol
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        if not self._ready:
            return False
        try:
            return await self._subprocess.forward_health()
        except Exception:
            return False

    async def metrics(self) -> str:
        try:
            inner_metrics = await self._subprocess.forward_metrics()
            inner_metrics += "\n"
        except Exception:
            inner_metrics = ""
        return inner_metrics + generate_latest(self._metrics_registry).decode("utf-8")

    async def models(self) -> Union[List[Model], Response]:
        if self._stopping:
            raise ClientError("Proxy engine is stopping")
        try:
            raw_resp = await self._subprocess.forward_models()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ServerError(
                f"Inner engine unreachable: {type(exc).__name__}"
            ) from exc
        except Exception:
            return []
        return Response(
            content=raw_resp.content,
            status_code=raw_resp.status_code,
            media_type=raw_resp.headers.get("content-type", "application/json"),
            headers=self._forward_headers(raw_resp),
        )

    async def chat(
        self, request: CreateChatCompletionRequest
    ) -> Union[
        CreateChatCompletionResponse, Iterator[str], AsyncIterator[str], Response
    ]:
        if self._stopping:
            raise ClientError("Proxy engine is stopping")

        self._requests_total.inc()
        req_id = uuid.uuid4().hex[:8]
        self._in_flight[req_id] = time.monotonic()
        self._in_flight_gauge.set(len(self._in_flight))
        t0 = time.monotonic()

        request_dict = request.model_dump(exclude_unset=True, mode="json")
        rewriter = RequestRewriter(
            self._chat_inspector, self._fetcher, self._credentials_map
        )

        fetch_results: List[FetchResult] = []

        try:
            fetch_t0 = time.monotonic()
            rewritten_dict, fetch_results = await rewriter.rewrite(request_dict)
            fetch_elapsed = time.monotonic() - fetch_t0
            self._fetch_total.inc(len(fetch_results))
            self._fetch_seconds.set(fetch_elapsed)

            try:
                raw_resp = await self._forward_chat_with_retry(rewritten_dict)
            except asyncio.CancelledError:
                logger.warning("[req-%s] Chat request cancelled", req_id)
                raise
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                logger.warning(
                    "[req-%s] Inner engine unreachable: %s", req_id, type(exc).__name__
                )
                raise ServerError(
                    f"Inner engine unreachable: {type(exc).__name__}"
                ) from exc
            except Exception as exc:
                logger.warning(
                    "[req-%s] Chat request failed: %s: %s",
                    req_id,
                    type(exc).__name__,
                    exc,
                )
                raise

            if request.stream:
                return StreamingResponse(
                    self._stream_with_cleanup(
                        raw_resp.aiter_lines(), fetch_results, req_id
                    ),
                    media_type="text/event-stream",
                )

            # Non-streaming: cleanup fetch_results on success too
            try:
                return Response(
                    content=raw_resp.content,
                    status_code=raw_resp.status_code,
                    media_type=raw_resp.headers.get("content-type", "application/json"),
                    headers=self._forward_headers(raw_resp),
                )
            finally:
                await self._fetcher.cleanup(fetch_results)
        except Exception:
            # On error path, also clean up fetch_results if not yet cleaned
            if not request.stream:
                try:
                    await self._fetcher.cleanup(fetch_results)
                except Exception:
                    pass
            raise
        finally:
            self._in_flight.pop(req_id, None)
            self._in_flight_gauge.set(len(self._in_flight))
            elapsed = time.monotonic() - t0
            self._request_duration_seconds.observe(elapsed)
            logger.info("[req-%s] chat completed in %.3fs", req_id, elapsed)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                httpx.ConnectError,
                httpx.TimeoutException,
            )
        ),
        reraise=True,
    )
    async def _forward_chat_with_retry(self, payload: Dict[str, Any]) -> Any:
        return await self._subprocess.forward_chat(payload)

    async def completion(
        self, request: CreateCompletionRequest
    ) -> Union[CreateCompletionResponse, Iterator[str], AsyncIterator[str], Response]:
        if self._stopping:
            raise ClientError("Proxy engine is stopping")

        self._requests_total.inc()
        req_id = uuid.uuid4().hex[:8]
        self._in_flight[req_id] = time.monotonic()
        self._in_flight_gauge.set(len(self._in_flight))
        t0 = time.monotonic()

        request_dict = request.model_dump(exclude_unset=True, mode="json")

        try:
            try:
                raw_resp = await self._forward_completion_with_retry(request_dict)
            except asyncio.CancelledError:
                logger.warning("[req-%s] Completion request cancelled", req_id)
                raise
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                logger.warning(
                    "[req-%s] Inner engine unreachable: %s", req_id, type(exc).__name__
                )
                raise ServerError(
                    f"Inner engine unreachable: {type(exc).__name__}"
                ) from exc
            except Exception as exc:
                logger.warning(
                    "[req-%s] Completion request failed: %s: %s",
                    req_id,
                    type(exc).__name__,
                    exc,
                )
                raise

            if request.stream:
                return StreamingResponse(
                    self._stream_with_cleanup(raw_resp.aiter_lines(), [], req_id),
                    media_type="text/event-stream",
                )

            return Response(
                content=raw_resp.content,
                status_code=raw_resp.status_code,
                media_type=raw_resp.headers.get("content-type", "application/json"),
                headers=self._forward_headers(raw_resp),
            )
        finally:
            self._in_flight.pop(req_id, None)
            self._in_flight_gauge.set(len(self._in_flight))
            elapsed = time.monotonic() - t0
            self._request_duration_seconds.observe(elapsed)
            logger.info("[req-%s] completion completed in %.3fs", req_id, elapsed)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                httpx.ConnectError,
                httpx.TimeoutException,
            )
        ),
        reraise=True,
    )
    async def _forward_completion_with_retry(self, payload: Dict[str, Any]) -> Any:
        return await self._subprocess.forward_completion(payload)

    async def embedding(
        self, request: CreateEmbeddingRequest
    ) -> Union[CreateEmbeddingResponse, Response]:
        if self._stopping:
            raise ClientError("Proxy engine is stopping")

        self._requests_total.inc()
        req_id = uuid.uuid4().hex[:8]
        self._in_flight[req_id] = time.monotonic()
        self._in_flight_gauge.set(len(self._in_flight))
        t0 = time.monotonic()

        request_dict = request.model_dump(exclude_unset=True, mode="json")

        try:
            try:
                raw_resp = await self._forward_embedding_with_retry(request_dict)
            except asyncio.CancelledError:
                logger.warning("[req-%s] Embedding request cancelled", req_id)
                raise
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                logger.warning(
                    "[req-%s] Inner engine unreachable: %s", req_id, type(exc).__name__
                )
                raise ServerError(
                    f"Inner engine unreachable: {type(exc).__name__}"
                ) from exc
            except Exception as exc:
                logger.warning(
                    "[req-%s] Embedding request failed: %s: %s",
                    req_id,
                    type(exc).__name__,
                    exc,
                )
                raise

            return Response(
                content=raw_resp.content,
                status_code=raw_resp.status_code,
                media_type=raw_resp.headers.get("content-type", "application/json"),
                headers=self._forward_headers(raw_resp),
            )
        finally:
            self._in_flight.pop(req_id, None)
            self._in_flight_gauge.set(len(self._in_flight))
            elapsed = time.monotonic() - t0
            self._request_duration_seconds.observe(elapsed)
            logger.info("[req-%s] embedding completed in %.3fs", req_id, elapsed)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (
                httpx.ConnectError,
                httpx.TimeoutException,
            )
        ),
        reraise=True,
    )
    async def _forward_embedding_with_retry(self, payload: Dict[str, Any]) -> Any:
        return await self._subprocess.forward_embedding(payload)

    async def load_model(self, model_name: str) -> Model:
        raise NotImplementedError(
            "ProxyEngine does not support dynamic model loading; "
            "specify models at inner engine startup"
        )

    async def unload_model(self, model_name: str) -> None:
        raise NotImplementedError(
            "ProxyEngine does not support dynamic model unloading"
        )

    async def _stream_with_cleanup(
        self,
        line_iter: AsyncIterator[str],
        fetch_results: List[FetchResult],
        req_id: str,
    ) -> AsyncIterator[str]:
        try:
            async for line in line_iter:
                yield line + "\n"
        finally:
            await self._fetcher.cleanup(fetch_results)
            self._in_flight.pop(req_id, None)
            self._in_flight_gauge.set(len(self._in_flight))

    @staticmethod
    def _forward_headers(resp: Any) -> Dict[str, str]:
        skip = {"content-encoding", "content-length", "transfer-encoding"}
        return {k: v for k, v in resp.headers.items() if k.lower() not in skip}

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
from dataclasses import dataclass
from typing import Any, Optional

import httpx

import aibrix.batch.constant as constant
from aibrix import envs
from aibrix.batch.client import EndpointSource
from aibrix.batch.client.sources import DiscoveryEndpointSource, GatewayEndpointSource
from aibrix.batch.job_driver.runtime import Endpoint, RuntimeBase, register_runtime
from aibrix.batch.job_entity import BatchJob, BatchJobError, BatchJobErrorCode
from aibrix.batch.internal.octagram_renderer import OctagramManifestRenderer
from aibrix.batch.state import JobEntityManager
from aibrix.context import InfrastructureContext
from aibrix.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_NAMESPACE = "default"
_REGION_DOMAINS = {
    "CN": "https://octagram-gateway.byted.org",
    "US": "https://octagram-gateway-us.byted.org",
    "I18N": "https://octagram-gateway-i18n.byted.org",
    "EU": "https://octagram-gateway-eu.tiktoke.org",
    "EUTTP": "https://octagram-gateway-eu.tiktoke.org",
}


@dataclass
class OctagramHandle:
    cluster: str
    namespace: str
    workload_name: str
    model_name: str
    psm: Optional[str]
    base_url: Optional[str]
    replicas: int
    source: Optional[EndpointSource] = None


class OctagramRuntime(RuntimeBase):
    """Provision a TCE DeploymentWorkload, discover endpoints, then tear it down."""

    provisions = True

    def __init__(
        self,
        context: InfrastructureContext,
        entity_manager: JobEntityManager,
        renderer: Optional[OctagramManifestRenderer] = None,
        ready_timeout_seconds: Optional[int] = None,
    ) -> None:
        if entity_manager is None:
            raise BatchJobError(
                BatchJobErrorCode.INVALID_DRIVER,
                "Octagram provider requires a job entity manager",
            )
        self._context = context
        self._entity_manager = entity_manager
        self._renderer = renderer or OctagramManifestRenderer(
            context.template_registry, context.profile_registry
        )
        self._httpx_client_wrapper = context.httpx_client_wrapper
        self._gateway_domain = self._resolve_gateway_domain()
        self._ready_timeout_seconds = ready_timeout_seconds or getattr(
            envs, "CONSUL_BATCH_DISCOVERY_TIMEOUT", 900
        )
        self._mgr_deleted_handler = entity_manager.on_job_deleted(
            self._job_deleted_handler
        )
        self._active_job_id: Optional[str] = None
        self._active_task: Optional[asyncio.Task[Any]] = None
        self._delete_requested = asyncio.Event()

    def cancelled(self) -> bool:
        return self._delete_requested.is_set()

    async def _job_deleted_handler(self, deleted_job: BatchJob) -> bool:
        deleted_job_id = deleted_job.job_id
        if deleted_job_id and deleted_job_id == self._active_job_id:
            self._delete_requested.set()
            if self._active_task is not None and not self._active_task.done():
                self._active_task.cancel()

        if self._mgr_deleted_handler is None:
            return True
        return await self._mgr_deleted_handler(deleted_job)

    async def _provision(self, job: BatchJob, job_id: str) -> OctagramHandle:
        self._active_job_id = job_id
        self._active_task = asyncio.current_task()
        self._delete_requested.clear()

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

        resource_detail = resource_details[0]
        workload = self._renderer.render(job.job_id, job.spec, resource_detail)
        cluster = (resource_detail.endpoint_cluster or "").lower()
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
        )
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

    async def _wait_ready(self, handle: OctagramHandle) -> None:
        await self._wait_for_workload_ready(
            cluster=handle.cluster,
            namespace=handle.namespace,
            workload_name=handle.workload_name,
            replicas=handle.replicas,
        )
        if handle.base_url is None:
            await self._wait_for_model_discoverable(handle)

    async def _connect(self, handle: OctagramHandle) -> Endpoint:
        if handle.base_url is not None:
            handle.source = GatewayEndpointSource(
                handle.base_url,
                capacity=max(handle.replicas, 1),
            )
        else:
            model_discovery = self._context.model_discovery
            if model_discovery is None:
                raise BatchJobError(
                    code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
                    message="Octagram consul discovery service is not configured",
                )
            handle.source = DiscoveryEndpointSource(
                model_discovery,
                handle.model_name,
                service_id=handle.psm,
                timeout=30.0,
            )
        return Endpoint(source=handle.source, model_name=handle.model_name)

    async def _teardown(self, handle: Optional[OctagramHandle]) -> None:
        if handle is not None:
            try:
                await self._delete_workload(handle)
            finally:
                if handle.source is not None:
                    await handle.source.aclose()
        self._active_job_id = None
        self._active_task = None

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
                service_id=handle.psm,
                timeout_seconds=self._ready_timeout_seconds,
            )  # type: ignore[call-arg]
            await model_discovery.wait_for_model_endpoints(
                served_model_name=handle.model_name,
                timeout_seconds=self._ready_timeout_seconds,
                service_id=handle.psm,
            )
        except TimeoutError as ex:
            raise BatchJobError(
                code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
                message=str(ex),
            ) from ex

    def _resolve_direct_base_url(self, job: BatchJob) -> Optional[str]:
        if job.spec.opts and constant.BATCH_OPTS_RESOURCE_ENDPOINT in job.spec.opts:
            return str(job.spec.opts[constant.BATCH_OPTS_RESOURCE_ENDPOINT])
        return None

    def _resolve_psm(self, workload: dict[str, Any]) -> Optional[str]:
        labels = workload["metadata"].get("labels", {})
        psm = labels.get("psm")
        if not psm:
            return None
        idc_name = getattr(self._renderer, "idc_name", "")
        return f"{psm}.service.{idc_name}" if idc_name else psm

    async def _apply_workload(
        self, cluster: str, namespace: str, workload: dict[str, Any]
    ) -> None:
        try:
            response = await self._octagram_request(
                "POST",
                self._workload_path(cluster, namespace, workload["metadata"]["name"]),
                json=workload,
            )
        except httpx.HTTPStatusError as ex:
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

    async def _wait_for_workload_ready(
        self, cluster: str, namespace: str, workload_name: str, replicas: int
    ) -> None:
        deadline = asyncio.get_running_loop().time() + self._ready_timeout_seconds
        while True:
            if self._delete_requested.is_set():
                raise asyncio.CancelledError
            try:
                workload = await self._octagram_request(
                    "GET",
                    self._workload_path(cluster, namespace, workload_name),
                )
            except httpx.HTTPStatusError as ex:
                raise BatchJobError(
                    code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
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
            replicas_statuses = status.get("replicasStatuses") or []
            if phase == "synced":
                if not replicas_statuses:
                    return
                available = max(
                    int(item.get("available") or 0) for item in replicas_statuses
                )
                if available >= replicas:
                    return
            if asyncio.get_running_loop().time() >= deadline:
                raise BatchJobError(
                    code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
                    message=(
                        f"Timed out waiting for octagram workload '{workload_name}' "
                        "to become ready"
                    ),
                )
            await asyncio.sleep(1)

    async def _delete_workload(self, handle: OctagramHandle) -> None:
        try:
            response = await self._octagram_request(
                "DELETE",
                self._workload_path(
                    handle.cluster, handle.namespace, handle.workload_name
                ),
            )
        except httpx.HTTPStatusError as ex:
            if ex.response.status_code != 404:
                raise
            return
        if response.get("error"):
            raise BatchJobError(
                code=BatchJobErrorCode.RESOURCE_DELETION_ERROR,
                message=f"Octagram workload delete failed: {response['error']}",
            )

    def _resolve_gateway_domain(self) -> str:
        configured = getattr(envs, "OCTAGRAM_GATEWAY_DOMAIN", None)
        if isinstance(configured, str) and configured:
            return configured.rstrip("/")
        region = getattr(envs, "REGION", "CN").upper().replace("-", "").replace("_", "")
        if region not in _REGION_DOMAINS:
            raise ValueError(
                f"Unsupported REGION '{region}', expected one of {sorted(_REGION_DOMAINS)}"
            )
        return _REGION_DOMAINS[region]

    def _workload_path(self, cluster: str, namespace: str, workload_name: str) -> str:
        return (
            f"{self._gateway_domain}/api/v1/clusters/{cluster}/namespaces/"
            f"{namespace}/deploymentworkloads/{workload_name}"
        )

    async def _octagram_request(
        self, method: str, url: str, **kwargs: Any
    ) -> dict[str, Any]:
        logger.info(
            "octagram request",
            method=method,
            url=url,
            body=kwargs.get("json"),
        )  # type: ignore[call-arg]
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
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            logger.error(
                "octagram request failed",
                method=method,
                url=url,
                status_code=response.status_code,
                body=response.text,
            )  # type: ignore[call-arg]
            raise
        return response.json()

    @staticmethod
    def _unwrap_octagram_data(response: dict[str, Any]) -> dict[str, Any]:
        data = response.get("data")
        if isinstance(data, dict):
            return data
        return response


register_runtime(
    "tce",
    lambda *, context, entity_manager, **_: OctagramRuntime(context, entity_manager),
)
register_runtime(
    "Octagram",
    lambda *, context, entity_manager, **_: OctagramRuntime(context, entity_manager),
)

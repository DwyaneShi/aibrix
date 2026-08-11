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

"""Deployment detail providers — side-channel queries for batch deployment info."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml

from aibrix import envs
from aibrix.batch.internal.config import AUTHORIZATION_HEADER, REGION_DOMAINS
from aibrix.batch.internal.octagram_utils import (
    get_job_name,
    get_psm,
    get_workload_name,
    parse_endpoint_cluster,
)
from aibrix.batch.internal.utils import async_retry
from aibrix.batch.job_entity import BatchJob
from aibrix.context import InfrastructureContext
from aibrix.context.deployment_detail import register_deployment_detail_provider
from aibrix.logger import init_logger

logger = init_logger(__name__)


class OctagramDeploymentDetailProvider:
    """DeploymentDetailProvider for Octagram/TCE backends."""

    def __init__(self) -> None:
        region = getattr(envs, "REGION", "CN").upper().replace("-", "").replace("_", "")
        if region not in REGION_DOMAINS:
            raise ValueError(
                f"Unsupported REGION '{region}', expected one of {sorted(REGION_DOMAINS)}"
            )

        self._base_urls = REGION_DOMAINS[region]

    @async_retry(Exception, tries=3, delay=0.5, backoff=2, logger=logger)
    async def _request(
        self,
        ctx: InfrastructureContext,
        name: str,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response: Optional[httpx.Response] = None
        error: Optional[Exception] = None
        try:
            if (
                ctx.httpx_client_wrapper is not None
                and ctx.httpx_client_wrapper.async_client is not None
            ):
                response = await ctx.httpx_client_wrapper.async_client.request(
                    method, url, **kwargs
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.request(method, url, **kwargs)
            assert response is not None
            response.raise_for_status()
            return response.json()
        except Exception as ex:
            error = ex
            raise
        finally:
            log_data: dict[str, Any] = {
                "method": method,
                "url": url,
                "status_code": response.status_code if response is not None else None,
            }
            if error is None:
                logger.info(f"{name} response", **log_data)  # type: ignore[call-arg]
            else:
                log_data["error"] = str(error)
                if response is not None:
                    log_data["response_text"] = response.text
                logger.error(f"{name} response", **log_data)  # type: ignore[call-arg]

    async def _octagram_request(
        self,
        ctx: InfrastructureContext,
        method: str,
        api: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        base_url = self._base_urls.octagram.rstrip("/")
        api = api.lstrip("/")
        return await self._request(
            ctx, "octagram", method, f"{base_url}/{api}", **kwargs
        )

    async def _tce_status_request(
        self,
        ctx: InfrastructureContext,
        method: str,
        api: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        base_url = self._base_urls.tce_status.rstrip("/")
        headers = kwargs.pop("headers", {})
        headers[AUTHORIZATION_HEADER] = getattr(envs, "TCE_STATUS_TOKEN", "")
        kwargs["headers"] = headers
        api = api.lstrip("/")
        return await self._request(
            ctx, "tce status", method, f"{base_url}/{api}", **kwargs
        )

    async def get_deployment_detail(
        self, ctx: InfrastructureContext, job: BatchJob
    ) -> Optional[Dict[str, Any]]:
        job_id = job.job_id
        if not job_id:
            logger.warning(
                "Job has no ID, cannot get deployment detail",
                job_id=job_id,
                job_json=job.model_dump_json(),
            )
            return None

        job_name = get_job_name(job)
        psm = get_psm(job)
        if not psm:
            logger.warning(
                "Job has no PSM, cannot get deployment detail",
                job_id=job_id,
                job_json=job.model_dump_json(),
            )
            return None

        if not job.spec.aibrix:
            logger.warning(
                "Job has no AIBrix spec, cannot get deployment detail",
                job_id=job_id,
                job_json=job.model_dump_json(),
            )
            return None

        resource_allocation = job.spec.aibrix.resource_allocation
        resource_details = (
            resource_allocation.resource_details if resource_allocation else None
        )
        if not resource_details:
            logger.warning(
                "Job has no resource details, cannot get deployment detail",
                job_id=job_id,
                job_json=job.model_dump_json(),
            )
            return None

        namespace = "default"
        if len(resource_details) == 1:
            return await self._get_deployment_detail(
                ctx,
                *parse_endpoint_cluster(resource_details[0].endpoint_cluster),
                namespace,
                psm,
                job_id,
                job_name,
            )

        details = await asyncio.gather(
            *(
                self._get_deployment_detail(
                    ctx,
                    *parse_endpoint_cluster(detail.endpoint_cluster),
                    namespace,
                    psm,
                    job_id,
                    get_workload_name(job_name, index),
                )
                for index, detail in enumerate(resource_details)
            )
        )
        available = [detail for detail in details if detail is not None]
        if not available:
            return None

        result = available[0]
        result["workloads"] = [
            workload
            for detail in available
            for workload in detail.get("workloads", [])
        ]
        return result

    async def _get_deployment_detail(
        self,
        ctx: InfrastructureContext,
        zone: str,
        idc: str,
        physical_cluster: str,
        logical_cluster: str,
        namespace: str,
        psm: str,
        job_id: str,
        job_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch Octagram workload detail via Octagram API.

        GET {base_url}/api/v1/clusters/{cluster}/namespaces/{namespace}
            /deploymentworkloads?labelSelector=psm={psm},name={name}

        Response structure:

            {
                "code": 200,
                "data": {
                    "items": [{
                        "metadata": {
                            "name": "batch-mlu590-d91379ce",
                            "uid": "dd036bc9-...",
                            "annotations": {
                                "bytedance.com/main-container-name": "batch-mlu590-d91379ce",
                                "bytedance.quota.salemode": "scheduled",
                                "queue-name": "compute-0-zc-..."
                            }
                        },
                        "spec": {
                            "deployStrategy": {"replicas": 10},
                            "podBase": {
                                "containers": [{
                                    "name": "batch-mlu590-d91379ce",
                                    "image": "hub.byted.org/.../vllm_mlu:1.0.0.9",
                                    "env": [{"name": "TCE_PHYSICAL_CLUSTER", "value": "Federation"}, ...],
                                    "resources": {"cpu": {"request": "32", "limit": "32"}, ...}
                                }],
                                "nodeSelector": {"nodeLevel": "dandelion-ai-mix"}
                            }
                        },
                        "status": {"phase": "synced"}
                    }]
                }
            }
        """
        try:
            cluster = f"{physical_cluster}-{idc}"
            label_selector = f"psm={psm},name={job_name}"
            api = (
                f"/api/v1/clusters/{cluster}/namespaces/{namespace}"
                f"/deploymentworkloads?labelSelector={label_selector}"
            )

            result = await self._octagram_request(ctx, "GET", api)

            code = result.get("code")
            if code not in (200, 0):
                logger.warning(
                    "Octagram API returned non-success code",
                    job_id=job_id,
                    code=code,
                    error=result.get("error", ""),
                )
                return None

            data = result.get("data", {})
            items = data.get("items") if isinstance(data, dict) else None
            if not isinstance(items, list) or len(items) == 0:
                logger.warning("Octagram API returned empty items", job_id=job_id)
                return None

            cluster_detail = {
                "zone": zone,
                "idc": idc,
                "physical_cluster": physical_cluster,
                "logical_cluster": logical_cluster,
                "namespace": "default",
            }

            workload_details: List[Dict[str, Any]] = []
            for item in items:
                wl_yaml = yaml.dump(item, default_flow_style=False, allow_unicode=True)

                metadata = item.get("metadata", {})
                spec = item.get("spec", {})
                workload_name = metadata.get("name", "")
                deployment_meta_annotations = spec.get("deploymentMeta", {}).get(
                    "annotations", {}
                )
                pod_base_annotations = spec.get("podBase", {}).get("annotations", {})
                deploy_strategy = spec.get("deployStrategy", {})
                pod_base = spec.get("podBase", {})
                containers = pod_base.get("containers", [])

                main_container_name = pod_base_annotations.get(
                    "bytedance.com/main-container-name", ""
                )
                primary_container = {}
                if main_container_name:
                    for container in containers:
                        if container.get("name", "") == main_container_name:
                            primary_container = container
                            break
                if not primary_container and containers:
                    primary_container = containers[0]

                if not primary_container:
                    continue

                image = primary_container.get("image", "")
                replicas = deploy_strategy.get("replicas", 0)
                resources = primary_container.get("resources", {})
                sale_mode = deployment_meta_annotations.get(
                    "bytedance.quota.salemode", ""
                )
                phase = item.get("status", {}).get("phase", "")

                pods, workload_healthy = await self._get_pod_detail(
                    ctx=ctx,
                    zone=zone,
                    idc=idc,
                    physical_cluster=physical_cluster,
                    logical_cluster=logical_cluster,
                    namespace=namespace,
                    psm=psm,
                    service_id=job_name,
                    workload_name=workload_name,
                )

                workload_details.append(
                    {
                        "id": metadata.get("uid", ""),
                        "name": workload_name,
                        "type": "DeploymentWorkload",
                        "sale_mode": sale_mode,
                        "role": "default",
                        "phase": phase,
                        "cluster": cluster_detail,
                        "replicas": replicas,
                        "image": image,
                        "resources": resources,
                        "queue_name": deployment_meta_annotations.get("queue-name", ""),
                        "pods": pods,
                        "healthy": workload_healthy,
                        "yaml": wl_yaml,
                    }
                )

            return {
                "type": "octagram",
                "job_id": job_id,
                "job_name": job_name,
                "psm": psm,
                "cluster": cluster_detail,
                "workloads": workload_details,
                "monitoring": self._get_argos_url(psm, job_name),
                "grafana": self._get_grafana_url(job_name),
            }
        except Exception as e:
            logger.warning(
                "_get_deployment_detail failed",
                job_id=job_id,
                error=str(e),
            )
            return None

    def _get_argos_url(
        self,
        psm: str,
        service_id: str,
        pod_name: Optional[str] = None,
    ) -> str:
        """https://cloud.bytedance.net/argos/bernard/server?bernard_service_id={service_id}&from=now-1h&pod={pod_name}&psm={psm}&to=now"""

        base_url = self._base_urls.argos.rstrip("/")
        query = f"{base_url}/bernard/server?from=now-1h&to=now&bernard_service_id={service_id}&psm={psm}"
        if pod_name:
            query += f"&pod={pod_name}"

        return query

    def _get_grafana_url(self, service_id: str) -> str:
        """https://grafana.byted.org/d/bM6pc4si5/merlin-jian-kong-bernard-service-monitoring-dashboard?orgId=1&refresh=1m&from=now-3h&to=now&var-idc=*&var-service={psm}"""

        base_url = self._base_urls.grafana.rstrip("/")
        query = f"{base_url}/d/bM6pc4si5/merlin-jian-kong-bernard-service-monitoring-dashboard?orgId=1&refresh=1m&from=now-3h&to=now&var-idc=*&var-service={service_id}"
        return query

    async def _get_pod_detail(
        self,
        ctx: InfrastructureContext,
        zone: str,
        idc: str,
        physical_cluster: str,
        logical_cluster: str,
        namespace: str,
        psm: str,
        service_id: str,
        workload_name: str,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Fetch pod details for a workload via TCE Status API.

        GET {tce_status}/api/v1/pods/deploy?zoneName=...&clusterName=...&idc=...
            &namespace=...&deployName=...&extensions=port|ipv6|containerStatus
            &logicalCluster=...&includeTerminating=true

        For Federation clusters ("fed" in clusterName), zoneName is omitted.

        Response structure:
        {
            "data": [
                {
                    "zoneName": "China-North-LF",
                    "clusterName": "Gallipoli",
                    "idc": "HL",
                    "logicalCluster": "default",
                    "psm": "...",
                    "podName": "workload-name-abc123-1",
                    "podPhase": "Running",
                    "node": "10.0.1.2",
                    "podIp": "10.0.1.2",
                    "creatTime": "2026-07-01T21:05:46+08:00",
                    "deleted": null,
                    "labels": {
                        "name": "deploy-name",
                        "psm": "xxx.service.xxx"
                    },
                    "containerInfos": {
                        "container-name": {
                            "containerID": "containerd://...",
                            "containerStatus": "Running",
                            "nodePorts": [{"nodePort": 8080, "port": 8080}]
                        }
                    },
                    "extraInfos": {
                        "hostIPv6": "...",
                        "podIPv6": "..."
                    }
                }
            ]
        }
        """
        try:
            extensions = "port|ipv6|containerStatus"
            params = (
                f"clusterName={physical_cluster}"
                f"&idc={idc}"
                f"&namespace={namespace}"
                f"&deployName={workload_name}"
                f"&extensions={extensions}"
                f"&logicalCluster={logical_cluster}"
                f"&includeTerminating=true"
            )
            if "fed" not in physical_cluster.lower():
                params = f"zoneName={zone}&" + params

            api = f"/api/v1/pods/deploy?{params}"

            result = await self._tce_status_request(ctx, "GET", api)

            pods_raw = result.get("data", [])
            if not isinstance(pods_raw, list):
                return [], False

            def _pod_ready(pod_phase: str, raw_item: dict) -> bool:
                if pod_phase.lower() != "running":
                    return False
                for cinfo in (raw_item.get("containerInfos") or {}).values():
                    if cinfo.get("containerStatus", "").lower() != "running":
                        return False
                return True

            pods: List[Dict[str, Any]] = []
            for item in pods_raw:
                labels = item.get("labels") or {}
                container_infos = item.get("containerInfos") or {}
                extra_infos = item.get("extraInfos") or {}
                primary_name = labels.get("name", "")
                primary_container = container_infos.get(primary_name, {})
                pod_name = item.get("podName", "")
                pod_phase = item.get("podPhase", "")

                cluster_detail = {
                    "zone": item.get("zoneName", ""),
                    "idc": item.get("idc", ""),
                    "physical_cluster": item.get("clusterName", ""),
                    "logical_cluster": item.get("logicalCluster", ""),
                    "namespace": "default",
                }

                pod: Dict[str, Any] = {
                    "name": pod_name,
                    "phase": pod_phase,
                    "cluster": cluster_detail,
                    "pod_ip": item.get("podIp", ""),
                    "pod_ipv6": extra_infos.get("podIPv6", ""),
                    "host_ipv4": extra_infos.get("hostIPv4", ""),
                    "host_ipv6": extra_infos.get("hostIPv6", ""),
                    "node": item.get("node", ""),
                    "node_name": item.get("nodeName", ""),
                    "created_at": item.get("creatTime", ""),
                    "container_status": primary_container.get("containerStatus", ""),
                    "ports": primary_container.get("nodePorts", []),
                    "healthy": _pod_ready(pod_phase, item),
                    "monitoring": self._get_argos_url(psm, service_id, pod_name),
                }
                pods.append(pod)

            workload_healthy = len(pods) > 0 and all(
                pods[i]["healthy"] for i in range(len(pods))
            )
            return pods, workload_healthy
        except Exception as e:
            logger.warning(
                "_get_pod_detail failed",
                workload_name=workload_name,
                error=str(e),
            )
            return [], False


# Self-register on import — one key per runtime_target.
_provider = OctagramDeploymentDetailProvider()
register_deployment_detail_provider("tce", _provider)
register_deployment_detail_provider("Octagram", _provider)

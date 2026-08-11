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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, cast

import httpx
import pytest

from aibrix import envs
from aibrix.batch.internal.octagram_runtime import (
    _MODEL_DISCOVERY_WAIT_SLICE_SECONDS,
    OctagramHandle,
    OctagramRuntime,
    _CombinedEndpointSource,
)
from aibrix.batch.job_driver.runtime import (
    RUNTIME_WAIT_MODE_PROVISION,
    RUNTIME_WAIT_MODE_RECONNECT,
)
from aibrix.batch.job_entity import (
    BatchJob,
    BatchJobEndpoint,
    BatchJobError,
    BatchJobErrorCode,
    BatchJobSpec,
    BatchJobState,
    BatchJobStatus,
    JobRuntimeRef,
    ObjectMeta,
    ResourceDetail,
    ResourceRequirement,
    TypeMeta,
)
from aibrix.batch.state import JobEntityManager
from aibrix.context import (
    InfrastructureContext,
    ModelDiscovery,
    ModelEndpoint,
    ModelLookupSnapshot,
)
from tests.batch.internal.octagram_backend import FakeOctagramRenderer


class FakeEntityManager(JobEntityManager):
    def __init__(self) -> None:
        super().__init__()

    async def submit_job(
        self, session_id: str, job: BatchJobSpec, request_count: int = 0
    ) -> None:
        return None

    async def update_job_ready(self, job: BatchJob) -> None:
        return None

    async def update_job_status(self, job: BatchJob) -> None:
        return None

    async def cancel_job(self, job: BatchJob) -> None:
        return None

    async def delete_job(self, job: BatchJob) -> None:
        return None

    async def get_job(
        self, job_id: str, force_reload: bool = False
    ) -> Optional[BatchJob]:
        return None

    async def list_jobs(
        self, after: Optional[str] = None, limit: int = 20
    ) -> list[BatchJob]:
        return []


class FakeAsyncClient:
    def __init__(self, responses: dict[str, list[httpx.Response | Exception]]) -> None:
        self._responses = {method: list(items) for method, items in responses.items()}
        self.calls: list[tuple[str, str]] = []

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        del kwargs
        self.calls.append((method, url))
        try:
            response_or_exc = self._responses[method].pop(0)
        except (KeyError, IndexError) as exc:
            raise AssertionError(f"unexpected {method} {url}") from exc
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc


class FakeHttpxClientWrapper:
    def __init__(self, responses: dict[str, list[httpx.Response | Exception]]) -> None:
        self.async_client = FakeAsyncClient(responses)


@dataclass
class FakeModelEndpoint(ModelEndpoint):
    base_url: str = "http://model-endpoint.example.test"


@dataclass
class FakeModelLookupSnapshot(ModelLookupSnapshot):
    version: int = 1
    endpoints: list[ModelEndpoint] = field(
        default_factory=lambda: [FakeModelEndpoint()]
    )


class FakeModelDiscovery:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def lookup(
        self,
        service_id: Optional[str] = None,
        filter_tags: Optional[dict[str, str]] = None,
        lookup_timeout_seconds: Optional[float] = None,
    ) -> ModelLookupSnapshot:
        del service_id, filter_tags, lookup_timeout_seconds
        return FakeModelLookupSnapshot()

    async def discover_model_endpoints(
        self,
        served_model_name: str,
        service_id: Optional[str] = None,
        filter_tags: Optional[dict[str, str]] = None,
        lookup_timeout_seconds: Optional[float] = None,
    ) -> ModelLookupSnapshot:
        del served_model_name, service_id, filter_tags, lookup_timeout_seconds
        return FakeModelLookupSnapshot()

    async def wait_for_model_endpoints(
        self,
        served_model_name: str,
        timeout_seconds: float,
        service_id: Optional[str] = None,
        filter_tags: Optional[dict[str, str]] = None,
        lookup_timeout_seconds: Optional[float] = None,
        poll_interval_seconds: float = 1.0,
    ) -> object:
        self.calls.append(
            {
                "served_model_name": served_model_name,
                "timeout_seconds": timeout_seconds,
                "service_id": service_id,
                "filter_tags": filter_tags,
                "lookup_timeout_seconds": lookup_timeout_seconds,
                "poll_interval_seconds": poll_interval_seconds,
            }
        )
        return FakeModelLookupSnapshot()


def _response(
    method: str,
    url: str,
    status_code: int,
    *,
    payload: Optional[dict] = None,
    text: str = "",
) -> httpx.Response:
    request = httpx.Request(method, url)
    if payload is not None:
        return httpx.Response(status_code, request=request, json=payload)
    return httpx.Response(status_code, request=request, text=text)


def _handle() -> OctagramHandle:
    return OctagramHandle(
        cluster="cluster-a",
        namespace="default",
        workload_name="batch-job-abcd1234",
        model_name="served-model",
        psm=None,
        base_url=None,
        replicas=1,
    )


class LegacySingleQueueRenderer(FakeOctagramRenderer):
    def render(
        self,
        job_id: str,
        job_name: str,
        spec: Any,
        provider_spec: Any,
        namespace: str = "default",
    ) -> dict[str, Any]:
        return super().render(
            job_id,
            job_name,
            spec,
            provider_spec,
            namespace=namespace,
        )


@pytest.mark.asyncio
async def test_connect_uses_handle_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _handle()
    handle.base_url = "http://engine.example.test:8000"
    handle.request_timeout_seconds = 123.0
    runtime = _runtime(FakeHttpxClientWrapper({}), monkeypatch)

    endpoint = await runtime._connect(handle)

    assert endpoint.source is not None
    channels = await endpoint.source.channels()
    assert channels[0]._timeout == 123.0


@pytest.mark.asyncio
async def test_octagram_runtime_manages_multiple_queue_workloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _make_job()
    allocation = job.spec.aibrix.resource_allocation
    allocation.resource_details.append(
        ResourceDetail(
            provider="tce",
            endpoint_cluster="zone/LQ/cluster-b/default",
            resource_pool_name="compute-lq",
            salemode="scheduled",
            resources=[
                ResourceRequirement(
                    accelerator_type="NVIDIA-A10",
                    accelerator_count=1,
                    replica=1,
                )
            ],
        )
    )

    gateway = "https://octagram-gateway.example.test"
    primary_url = (
        f"{gateway}/api/v1/clusters/cluster-a-HL/namespaces/default/"
        "deploymentworkloads/batch-job-1234"
    )
    supplemental_name = "batch-mock-template-job-1234-allocation-1"
    supplemental_url = (
        f"{gateway}/api/v1/clusters/cluster-b-LQ/namespaces/default/"
        f"deploymentworkloads/{supplemental_name}"
    )
    runtime = _runtime(
        FakeHttpxClientWrapper(
            {
                "POST": [
                    _response("POST", primary_url, 200, payload={"data": {}}),
                    _response("POST", supplemental_url, 200, payload={"data": {}}),
                ]
            }
        ),
        monkeypatch,
        renderer=FakeOctagramRenderer(),
    )

    handle = await runtime._provision(job, job.job_id)

    assert handle.workload_name == "batch-job-1234"
    assert len(handle.supplemental) == 1
    assert handle.supplemental[0].workload_name == supplemental_name
    assert handle.supplemental[0].model_name == handle.model_name
    assert handle.supplemental[0].psm == "fake-psm.service.lq"

    execution = runtime._build_runtime_ref(job)
    assert execution is not None
    persisted_execution = JobRuntimeRef.model_validate_json(
        execution.model_dump_json(by_alias=True, exclude_none=True)
    )
    assert persisted_execution.reconnect_payload is not None
    assert persisted_execution.reconnect_payload["cluster"] == "cluster-a-HL"
    assert persisted_execution.reconnect_payload["workloadName"] == "batch-job-1234"
    assert persisted_execution.reconnect_payload["modelName"] == "batch-job-1234"
    assert persisted_execution.reconnect_payload["replicas"] == 1
    assert len(persisted_execution.reconnect_payload["supplementalWorkloads"]) == 1

    reloaded_runtime = _runtime(FakeHttpxClientWrapper({}), monkeypatch)
    reloaded = await reloaded_runtime._load_handle(job, job.job_id, execution)
    assert reloaded is not None
    assert [item.workload_name for item in reloaded.supplemental] == [
        supplemental_name
    ]

    runtime._context.model_discovery = cast(ModelDiscovery, FakeModelDiscovery())
    endpoint = await runtime._connect(handle)
    assert isinstance(endpoint.source, _CombinedEndpointSource)

    deleted: list[str] = []

    async def _delete_workload(workload: OctagramHandle) -> None:
        deleted.append(workload.workload_name)

    monkeypatch.setattr(runtime, "_delete_workload", _delete_workload)
    await runtime._teardown(handle)
    assert deleted == [supplemental_name, "batch-job-1234"]


def _runtime(
    wrapper: FakeHttpxClientWrapper,
    monkeypatch: pytest.MonkeyPatch,
    renderer: Optional[FakeOctagramRenderer] = None,
) -> OctagramRuntime:
    monkeypatch.setattr(
        envs,
        "OCTAGRAM_GATEWAY_DOMAIN",
        "https://octagram-gateway.example.test",
    )
    return OctagramRuntime(
        InfrastructureContext(httpx_client_wrapper=wrapper),
        renderer=cast(Any, renderer),
    )


def _make_job(job_id: str = "job-123456789abc") -> BatchJob:
    spec = BatchJobSpec.from_strings(
        input_file_id="input-file-1",
        endpoint=BatchJobEndpoint.CHAT_COMPLETIONS.value,
        completion_window="24h",
        aibrix={
            "model_template": {"name": "mock-template"},
            "runtime": {"target": "tce"},
            "resource_allocation": {
                "provision_id": "reservation-1",
                "provision_resource_deadline": 3600,
                "resource_details": [
                    {
                        "endpoint_cluster": "zone/HL/cluster-a/default",
                        "gpu_type": "H100",
                        "replica": 1,
                    }
                ],
            },
        },
    )
    status = BatchJobStatus.model_validate(
        {
            "jobID": job_id,
            "state": BatchJobState.IN_PROGRESS,
            "createdAt": datetime.now(timezone.utc),
            "inProgressAt": datetime.now(timezone.utc),
        }
    )
    return BatchJob(
        sessionID="session-1",
        typeMeta=TypeMeta(apiVersion="batch/v1", kind="Job"),
        metadata=ObjectMeta.model_validate({"name": "job", "namespace": "default"}),
        spec=spec,
        status=status,
    )


def test_resolve_psm_appends_idc_for_worker_psm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        FakeHttpxClientWrapper({}),
        monkeypatch,
        renderer=FakeOctagramRenderer(psm="inf.aibrix.inference_workers"),
    )
    workload = {
        "metadata": {
            "labels": {
                "psm": "inf.aibrix.inference_workers",
            }
        }
    }

    assert runtime._resolve_psm(workload) == "inf.aibrix.inference_workers.service.hl"


def test_resolve_psm_does_not_double_append_idc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        FakeHttpxClientWrapper({}),
        monkeypatch,
        renderer=FakeOctagramRenderer(psm="inf.aibrix.inference_workers.service.my2"),
    )
    workload = {
        "metadata": {
            "labels": {
                "psm": "inf.aibrix.inference_workers.service.my2",
            }
        }
    }

    assert runtime._resolve_psm(workload) == "inf.aibrix.inference_workers.service.my2"


def test_resolve_psm_normalizes_aliyun_va_idc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = FakeOctagramRenderer(psm="inf.aibrix.inference_workers")
    renderer.idc_name = "aliyun_va"
    runtime = _runtime(
        FakeHttpxClientWrapper({}),
        monkeypatch,
        renderer=renderer,
    )
    workload = {
        "metadata": {
            "labels": {
                "psm": "inf.aibrix.inference_workers",
            }
        }
    }

    assert (
        runtime._resolve_psm(workload) == "inf.aibrix.inference_workers.service.maliva"
    )


def test_resolve_psm_normalizes_existing_aliyun_va_service_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(
        FakeHttpxClientWrapper({}),
        monkeypatch,
        renderer=FakeOctagramRenderer(
            psm="inf.aibrix.inference_workers.service.aliyun_va"
        ),
    )
    workload = {
        "metadata": {
            "labels": {
                "psm": "inf.aibrix.inference_workers.service.aliyun_va",
            }
        }
    }

    assert (
        runtime._resolve_psm(workload) == "inf.aibrix.inference_workers.service.maliva"
    )


@pytest.mark.parametrize(
    ("psm", "idc_name", "expected"),
    [
        (
            "inf.aibrix.inference_workers",
            "",
            "inf.aibrix.inference_workers",
        ),
        (
            "inf.aibrix.inference_workers.service",
            "HL",
            "inf.aibrix.inference_workers.service.hl",
        ),
    ],
)
def test_normalize_psm_service_idc_handles_incomplete_service_ids(
    monkeypatch: pytest.MonkeyPatch,
    psm: str,
    idc_name: str,
    expected: str,
) -> None:
    runtime = _runtime(FakeHttpxClientWrapper({}), monkeypatch)

    assert runtime._normalize_psm_service_idc(psm, idc_name) == expected


_PROVISIONED_CLUSTER = "cluster-a-HL"


def _scripted_monotonic(values: tuple[float, ...]) -> Any:
    scripted_values = iter(values)
    original_monotonic = time.monotonic
    last_value = values[-1]

    def _monotonic() -> float:
        nonlocal last_value
        try:
            last_value = next(scripted_values)
            return last_value
        except StopIteration:
            # asyncio and pytest may call monotonic more often than the
            # runtime code path under test; keep returning a stable value.
            return max(last_value, original_monotonic())

    return _monotonic


@pytest.mark.asyncio
async def test_octagram_runtime_uses_immediate_liveness_failure_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(FakeHttpxClientWrapper({"GET": []}), monkeypatch)

    assert runtime.session_liveness_failure_threshold == 1


@pytest.mark.asyncio
async def test_delete_workload_falls_back_to_scale_zero_on_non_404_delete_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _handle()
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "DELETE": [
                _response("DELETE", base_url, 500, text="delete boom"),
                _response("DELETE", base_url, 200, payload={}),
            ],
            "PATCH": [
                _response("PATCH", f"{base_url}/scale?replicas=0", 200, payload={})
            ],
        }
    )
    runtime = _runtime(wrapper, monkeypatch)
    wait_calls: list[tuple[str, str, str, int, Optional[str]]] = []

    async def _wait_ready(
        cluster: str,
        namespace: str,
        workload_name: str,
        replicas: int,
        wait_mode: str = RUNTIME_WAIT_MODE_RECONNECT,
        request_reason: Optional[str] = None,
    ) -> None:
        del wait_mode
        wait_calls.append((cluster, namespace, workload_name, replicas, request_reason))

    monkeypatch.setattr(runtime, "_wait_for_workload", _wait_ready)

    await runtime._delete_workload(handle)

    assert wait_calls == [
        (
            "cluster-a",
            "default",
            "batch-job-abcd1234",
            0,
            "wait for workload synced at zero replicas",
        )
    ]
    assert wrapper.async_client.calls == [
        ("DELETE", base_url),
        ("PATCH", f"{base_url}/scale?replicas=0"),
        ("DELETE", base_url),
    ]


@pytest.mark.asyncio
async def test_delete_workload_404_returns_without_scale_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _handle()
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {"DELETE": [_response("DELETE", base_url, 404, text="gone")]}
    )
    runtime = _runtime(wrapper, monkeypatch)

    await runtime._delete_workload(handle)

    assert wrapper.async_client.calls == [("DELETE", base_url)]


@pytest.mark.asyncio
async def test_delete_workload_payload_not_found_returns_without_scale_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _handle()
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "DELETE": [
                _response(
                    "DELETE",
                    base_url,
                    200,
                    payload={
                        "data": None,
                        "error": (
                            "deploymentworkloads.core.tce.byted.org "
                            '"batch-job-abcd1234" not found'
                        ),
                        "code": 404,
                    },
                )
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)

    await runtime._delete_workload(handle)

    assert wrapper.async_client.calls == [("DELETE", base_url)]


@pytest.mark.asyncio
async def test_delete_workload_raises_when_scale_zero_fallback_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _handle()
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "DELETE": [_response("DELETE", base_url, 500, text="delete boom")],
            "PATCH": [
                _response(
                    "PATCH", f"{base_url}/scale?replicas=0", 502, text="scale boom"
                )
            ],
        }
    )
    runtime = _runtime(wrapper, monkeypatch)

    with pytest.raises(BatchJobError) as exc_info:
        await runtime._delete_workload(handle)

    assert exc_info.value.code == BatchJobErrorCode.RESOURCE_DELETION_ERROR.value
    assert "delete failed (500): delete boom" in exc_info.value.message
    assert "fallback scale to zero failed (502): scale boom" in exc_info.value.message


@pytest.mark.asyncio
async def test_delete_workload_raises_when_scaled_workload_never_resyncs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _handle()
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "DELETE": [_response("DELETE", base_url, 500, text="delete boom")],
            "PATCH": [
                _response("PATCH", f"{base_url}/scale?replicas=0", 200, payload={})
            ],
        }
    )
    runtime = _runtime(wrapper, monkeypatch)

    async def _wait_ready(
        cluster: str,
        namespace: str,
        workload_name: str,
        replicas: int,
        wait_mode: str = RUNTIME_WAIT_MODE_RECONNECT,
        request_reason: Optional[str] = None,
    ) -> None:
        del cluster, namespace, workload_name, replicas, wait_mode, request_reason
        raise BatchJobError(
            code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
            message="Timed out waiting for octagram workload to become ready",
        )

    monkeypatch.setattr(runtime, "_wait_for_workload", _wait_ready)

    with pytest.raises(BatchJobError) as exc_info:
        await runtime._delete_workload(handle)

    assert exc_info.value.code == BatchJobErrorCode.RESOURCE_DELETION_ERROR.value
    assert "delete failed (500): delete boom" in exc_info.value.message
    assert "fallback scale to zero did not reach synced" in exc_info.value.message


@pytest.mark.asyncio
async def test_wait_for_workload_returns_when_zero_replica_is_synced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "GET": [
                _response(
                    "GET",
                    base_url,
                    200,
                    payload={
                        "data": {
                            "status": {
                                "phase": "synced",
                                "replicasStatuses": [
                                    {
                                        "name": "batch-job-abcd1234",
                                        "type": "workload",
                                    }
                                ],
                            }
                        }
                    },
                )
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)

    await runtime._wait_for_workload(
        cluster="cluster-a",
        namespace="default",
        workload_name="batch-job-abcd1234",
        replicas=0,
    )

    assert wrapper.async_client.calls == [("GET", base_url)]


@pytest.mark.asyncio
async def test_wait_for_workload_404_raises_not_found_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {"GET": [_response("GET", base_url, 404, text="gone")]}
    )
    runtime = _runtime(wrapper, monkeypatch)

    with pytest.raises(BatchJobError) as exc_info:
        await runtime._wait_for_workload(
            cluster="cluster-a",
            namespace="default",
            workload_name="batch-job-abcd1234",
            replicas=1,
            wait_mode=RUNTIME_WAIT_MODE_RECONNECT,
        )

    assert exc_info.value.code == BatchJobErrorCode.RESOURCE_NOTFOUND_ERROR.value
    assert "Octagram workload status check failed (404): gone" in exc_info.value.message


@pytest.mark.asyncio
async def test_wait_for_workload_provision_retries_404_until_workload_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "GET": [
                _response("GET", base_url, 404, text="gone"),
                _response(
                    "GET",
                    base_url,
                    200,
                    payload={
                        "data": {
                            "status": {
                                "phase": "synced",
                                "replicasStatuses": [{"available": 1}],
                            }
                        }
                    },
                ),
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)

    async def _sleep(_: float) -> None:
        return None

    monkeypatch.setattr("aibrix.batch.internal.octagram_runtime.asyncio.sleep", _sleep)

    await runtime._wait_for_workload(
        cluster="cluster-a",
        namespace="default",
        workload_name="batch-job-abcd1234",
        replicas=1,
        wait_mode=RUNTIME_WAIT_MODE_PROVISION,
    )

    assert wrapper.async_client.calls == [("GET", base_url), ("GET", base_url)]


@pytest.mark.asyncio
async def test_wait_for_workload_caps_minimum_by_requested_replicas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_replicas = 1
    configured_minimum = 2
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "GET": [
                _response(
                    "GET",
                    base_url,
                    200,
                    payload={
                        "data": {
                            "status": {
                                "phase": "synced",
                                "replicasStatuses": [{"available": requested_replicas}],
                            }
                        }
                    },
                )
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)
    monkeypatch.setattr(
        "aibrix.batch.internal.octagram_runtime._MIN_READY_REPLICAS",
        configured_minimum,
    )

    await runtime._wait_for_workload(
        cluster="cluster-a",
        namespace="default",
        workload_name="batch-job-abcd1234",
        replicas=requested_replicas,
    )

    assert wrapper.async_client.calls == [("GET", base_url)]


@pytest.mark.asyncio
async def test_wait_for_workload_provision_404_respects_grace_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "GET": [
                _response("GET", base_url, 404, text="gone"),
                _response("GET", base_url, 404, text="gone"),
                _response("GET", base_url, 404, text="gone"),
                _response("GET", base_url, 404, text="gone"),
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)
    cast(Any, runtime)._workload_not_found_grace_seconds = 0.2
    original_sleep = asyncio.sleep

    async def _sleep(_: float) -> None:
        await original_sleep(0.1)

    monkeypatch.setattr("aibrix.batch.internal.octagram_runtime.asyncio.sleep", _sleep)

    with pytest.raises(BatchJobError) as exc_info:
        await runtime._wait_for_workload(
            cluster="cluster-a",
            namespace="default",
            workload_name="batch-job-abcd1234",
            replicas=1,
            wait_mode=RUNTIME_WAIT_MODE_PROVISION,
        )

    assert exc_info.value.code == BatchJobErrorCode.RESOURCE_NOTFOUND_ERROR.value
    assert "Octagram workload status check failed (404): gone" in exc_info.value.message
    assert len(wrapper.async_client.calls) >= 2


@pytest.mark.asyncio
async def test_wait_for_model_discoverable_checks_workload_existence_each_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(FakeHttpxClientWrapper({"GET": []}), monkeypatch)
    model_discovery = FakeModelDiscovery()
    runtime._context.model_discovery = cast(ModelDiscovery, model_discovery)
    liveness_calls: list[str] = []

    async def _check_liveness(
        handle: OctagramHandle, reason: str = "unspecified"
    ) -> None:
        del handle
        liveness_calls.append(reason)

    runtime._check_liveness = _check_liveness  # type: ignore[method-assign]

    await runtime._wait_for_model_discoverable(_handle())

    assert liveness_calls == ["wait for model discoverable"]
    assert len(model_discovery.calls) == 1
    assert model_discovery.calls[0]["served_model_name"] == "served-model"
    assert (
        model_discovery.calls[0]["timeout_seconds"]
        == _MODEL_DISCOVERY_WAIT_SLICE_SECONDS
    )
    assert (
        model_discovery.calls[0]["lookup_timeout_seconds"]
        == _MODEL_DISCOVERY_WAIT_SLICE_SECONDS
    )
    assert (
        model_discovery.calls[0]["poll_interval_seconds"]
        == _MODEL_DISCOVERY_WAIT_SLICE_SECONDS
    )


@pytest.mark.asyncio
async def test_wait_ready_forwards_wait_mode_to_workload_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(FakeHttpxClientWrapper({"GET": []}), monkeypatch)
    readiness_wait_modes: list[str] = []

    async def _wait_for_workload(
        cluster: str,
        namespace: str,
        workload_name: str,
        replicas: int,
        wait_mode: str = RUNTIME_WAIT_MODE_RECONNECT,
        request_reason: Optional[str] = None,
    ) -> None:
        del cluster, namespace, workload_name, replicas, request_reason
        readiness_wait_modes.append(wait_mode)

    async def _wait_for_model_discoverable(handle: OctagramHandle) -> None:
        del handle

    runtime._wait_for_workload = _wait_for_workload  # type: ignore[method-assign]
    runtime._wait_for_model_discoverable = _wait_for_model_discoverable  # type: ignore[method-assign]

    await runtime._wait_ready(_handle(), wait_mode=RUNTIME_WAIT_MODE_RECONNECT)

    assert readiness_wait_modes == [RUNTIME_WAIT_MODE_RECONNECT]


@pytest.mark.asyncio
async def test_check_liveness_forwards_reason_to_workload_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(FakeHttpxClientWrapper({"GET": []}), monkeypatch)
    liveness_calls: list[tuple[str, Optional[str]]] = []

    async def _wait_for_workload(
        cluster: str,
        namespace: str,
        workload_name: str,
        replicas: int,
        wait_mode: str = RUNTIME_WAIT_MODE_RECONNECT,
        request_reason: Optional[str] = None,
    ) -> None:
        del cluster, namespace, workload_name, replicas
        liveness_calls.append((wait_mode, request_reason))

    runtime._wait_for_workload = _wait_for_workload  # type: ignore[method-assign]

    await runtime._check_liveness(_handle(), reason="session_liveness_loop")

    assert liveness_calls == [
        (RUNTIME_WAIT_MODE_RECONNECT, "check_liveness:session_liveness_loop")
    ]


@pytest.mark.asyncio
async def test_apply_workload_409_is_treated_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {"POST": [_response("POST", base_url, 409, text="already exists")]}
    )
    runtime = _runtime(wrapper, monkeypatch)

    await runtime._apply_workload(
        "cluster-a",
        "default",
        {"metadata": {"name": "batch-job-abcd1234"}},
    )

    assert wrapper.async_client.calls == [("POST", base_url)]


@pytest.mark.asyncio
async def test_octagram_runtime_builds_execution_ref_with_current_payload_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _make_job()
    base_url = (
        f"https://octagram-gateway.example.test/api/v1/clusters/{_PROVISIONED_CLUSTER}/"
        "namespaces/default/deploymentworkloads/batch-job-1234"
    )
    runtime = _runtime(
        FakeHttpxClientWrapper(
            {"POST": [_response("POST", base_url, 200, payload={"data": {}})]}
        ),
        monkeypatch,
        renderer=LegacySingleQueueRenderer(psm="fake-psm.service.my2"),
    )
    await runtime._provision(job, job.job_id)

    execution = runtime._build_runtime_ref(job)

    assert execution is not None
    assert execution.driver_type == "tce"
    assert execution.owner_ref == f"{_PROVISIONED_CLUSTER}:default:batch-job-1234"
    assert execution.reconnect_payload == {
        "cluster": _PROVISIONED_CLUSTER,
        "namespace": "default",
        "workloadName": "batch-job-1234",
        "modelName": "batch-job-1234",
        "psm": "fake-psm.service.my2",
        "baseUrl": None,
        "replicas": 1,
    }


@pytest.mark.asyncio
async def test_octagram_runtime_reconnect_accepts_current_payload_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _make_job()
    runtime = _runtime(FakeHttpxClientWrapper({"POST": []}), monkeypatch)
    monkeypatch.setattr(runtime, "_resolve_request_timeout_seconds", lambda _: 45.0)

    handle = await runtime._load_handle(
        job,
        job.job_id,
        JobRuntimeRef(
            driverType="tce",
            ownerRef=f"{_PROVISIONED_CLUSTER}:default:batch-job-1234",
            reconnectPayload={
                "cluster": _PROVISIONED_CLUSTER,
                "namespace": "default",
                "workloadName": "batch-job-1234",
                "modelName": "batch-job-1234",
                "psm": "fake-psm.service.hl",
                "baseUrl": None,
                "replicas": 1,
            },
        ),
    )

    assert handle is not None
    assert handle.cluster == _PROVISIONED_CLUSTER
    assert handle.namespace == "default"
    assert handle.workload_name == "batch-job-1234"
    assert handle.model_name == "batch-job-1234"
    assert handle.psm == "fake-psm.service.hl"
    assert handle.request_timeout_seconds == 45.0
    assert runtime._active_handle == handle


@pytest.mark.asyncio
async def test_octagram_runtime_reconnect_ignores_legacy_payload_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _make_job()
    runtime = _runtime(FakeHttpxClientWrapper({"POST": []}), monkeypatch)
    monkeypatch.setattr(runtime, "_resolve_request_timeout_seconds", lambda _: 45.0)

    handle = await runtime._load_handle(
        job,
        job.job_id,
        JobRuntimeRef(
            driverType="tce",
            ownerRef=f"{_PROVISIONED_CLUSTER}:default:batch-job-1234",
            reconnectPayload={
                "cluster": _PROVISIONED_CLUSTER,
                "namespace": "default",
                "workloadName": "batch-job-1234",
                "modelName": "batch-job-1234",
                "psm": "fake-psm.service.hl",
                "baseUrl": None,
                "replicas": 1,
                "requestTimeoutSeconds": 123.0,
            },
        ),
    )

    assert handle is not None
    assert handle.request_timeout_seconds == 45.0


@pytest.mark.asyncio
async def test_octagram_request_get_logs_default_response_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "GET": [
                _response(
                    "GET",
                    base_url,
                    200,
                    payload={
                        "data": {
                            "status": {
                                "phase": "synced",
                                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
                            }
                        }
                    },
                )
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)
    logs: list[tuple[str, str, dict]] = []

    def _info(event: str, **kwargs) -> None:
        logs.append(("info", event, kwargs))

    monkeypatch.setattr("aibrix.batch.internal.octagram_runtime.logger.info", _info)

    payload = await runtime._octagram_request("GET", base_url, reason="test_reason")

    assert payload == {
        "data": {
            "status": {
                "phase": "synced",
                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
            }
        }
    }
    assert logs == [
        (
            "info",
            "octagram response",
            {
                "method": "GET",
                "url": base_url,
                "reason": "test_reason",
                "status_code": 200,
                "phase": "synced",
                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
            },
        )
    ]


@pytest.mark.asyncio
async def test_octagram_request_get_suppresses_repeated_response_logs_within_default_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    payload = {
        "data": {
            "status": {
                "phase": "synced",
                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
            }
        }
    }
    wrapper = FakeHttpxClientWrapper(
        {
            "GET": [
                _response("GET", base_url, 200, payload=payload),
                _response("GET", base_url, 200, payload=payload),
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)
    runtime.session_liveness_check_interval_s = 5
    logs: list[tuple[str, str, dict]] = []

    def _info(event: str, **kwargs) -> None:
        logs.append(("info", event, kwargs))

    monkeypatch.setattr("aibrix.batch.internal.octagram_runtime.logger.info", _info)
    monkeypatch.setattr(
        "aibrix.batch.internal.octagram_runtime.time.monotonic",
        _scripted_monotonic((100.0, 120.0)),
    )

    await runtime._octagram_request("GET", base_url)
    await runtime._octagram_request("GET", base_url)

    assert len(logs) == 1


@pytest.mark.asyncio
async def test_octagram_request_get_logs_repeated_response_after_liveness_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    payload = {
        "data": {
            "status": {
                "phase": "synced",
                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
            }
        }
    }
    wrapper = FakeHttpxClientWrapper(
        {
            "GET": [
                _response("GET", base_url, 200, payload=payload),
                _response("GET", base_url, 200, payload=payload),
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)
    runtime.session_liveness_check_interval_s = 45
    logs: list[tuple[str, str, dict]] = []

    def _info(event: str, **kwargs) -> None:
        logs.append(("info", event, kwargs))

    monkeypatch.setattr("aibrix.batch.internal.octagram_runtime.logger.info", _info)
    monkeypatch.setattr(
        "aibrix.batch.internal.octagram_runtime.time.monotonic",
        _scripted_monotonic((100.0, 146.0)),
    )

    await runtime._octagram_request("GET", base_url)
    await runtime._octagram_request("GET", base_url)

    assert len(logs) == 2


@pytest.mark.asyncio
async def test_octagram_request_get_dedup_tracks_last_reason_per_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    payload = {
        "data": {
            "status": {
                "phase": "synced",
                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
            }
        }
    }
    wrapper = FakeHttpxClientWrapper(
        {
            "GET": [
                _response("GET", base_url, 200, payload=payload),
                _response("GET", base_url, 200, payload=payload),
                _response("GET", base_url, 200, payload=payload),
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)
    runtime.session_liveness_check_interval_s = 30
    logs: list[tuple[str, str, dict]] = []

    def _info(event: str, **kwargs) -> None:
        logs.append(("info", event, kwargs))

    monkeypatch.setattr("aibrix.batch.internal.octagram_runtime.logger.info", _info)
    monkeypatch.setattr(
        "aibrix.batch.internal.octagram_runtime.time.monotonic",
        _scripted_monotonic((100.0, 110.0, 120.0)),
    )

    await runtime._octagram_request("GET", base_url, reason="reason_a")
    await runtime._octagram_request("GET", base_url, reason="reason_b")
    await runtime._octagram_request("GET", base_url, reason="reason_a")

    assert [entry[2]["reason"] for entry in logs] == [
        "reason_a",
        "reason_b",
        "reason_a",
    ]


@pytest.mark.asyncio
async def test_octagram_request_get_suppresses_repeated_error_logs_within_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "GET": [
                _response(
                    "GET",
                    base_url,
                    404,
                    payload={
                        "data": {
                            "status": {
                                "phase": "failed",
                                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
                            }
                        }
                    },
                    text="gone",
                ),
                _response(
                    "GET",
                    base_url,
                    404,
                    payload={
                        "data": {
                            "status": {
                                "phase": "failed",
                                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
                            }
                        }
                    },
                    text="gone",
                ),
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)
    logs: list[tuple[str, str, dict]] = []

    def _error(event: str, **kwargs) -> None:
        logs.append(("error", event, kwargs))

    monkeypatch.setattr("aibrix.batch.internal.octagram_runtime.logger.error", _error)
    monkeypatch.setattr(
        "aibrix.batch.internal.octagram_runtime.time.monotonic",
        _scripted_monotonic((100.0, 120.0)),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await runtime._octagram_request("GET", base_url)
    with pytest.raises(httpx.HTTPStatusError):
        await runtime._octagram_request("GET", base_url)

    assert len(logs) == 1


@pytest.mark.asyncio
async def test_octagram_request_post_skips_default_response_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "POST": [
                _response(
                    "POST",
                    base_url,
                    200,
                    payload={"data": {"status": {"phase": "synced"}}},
                )
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)
    logs: list[tuple[str, str, dict]] = []

    def _info(event: str, **kwargs) -> None:
        logs.append(("info", event, kwargs))

    monkeypatch.setattr("aibrix.batch.internal.octagram_runtime.logger.info", _info)

    await runtime._octagram_request(
        "POST", base_url, reason="test_reason", json={"name": "demo"}
    )

    assert logs == [
        (
            "info",
            "octagram response",
            {
                "method": "POST",
                "url": base_url,
                "reason": "test_reason",
                "status_code": 200,
            },
        )
    ]


@pytest.mark.asyncio
async def test_octagram_request_http_error_logs_response_text_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper(
        {
            "GET": [
                _response(
                    "GET",
                    base_url,
                    404,
                    payload={
                        "data": {
                            "status": {
                                "phase": "failed",
                                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
                            }
                        }
                    },
                    text="gone",
                )
            ]
        }
    )
    runtime = _runtime(wrapper, monkeypatch)
    logs: list[tuple[str, str, dict]] = []

    def _error(event: str, **kwargs) -> None:
        logs.append(("error", event, kwargs))

    monkeypatch.setattr("aibrix.batch.internal.octagram_runtime.logger.error", _error)

    with pytest.raises(httpx.HTTPStatusError):
        await runtime._octagram_request("GET", base_url, reason="test_reason")

    assert logs == [
        (
            "error",
            "octagram response",
            {
                "method": "GET",
                "url": base_url,
                "reason": "test_reason",
                "status_code": 404,
                "phase": "failed",
                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
                "error": "Client error '404 Not Found' for url "
                f"'{base_url}'\nFor more information check: "
                "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404",
                "response_text": '{"data": {"status": {"phase": "failed", '
                '"replicasStatuses": [{"name": "batch-job-abcd1234"}]}}}',
            },
        )
    ]


@pytest.mark.asyncio
async def test_octagram_request_non_http_error_logs_without_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = (
        "https://octagram-gateway.example.test/api/v1/clusters/cluster-a/"
        "namespaces/default/deploymentworkloads/batch-job-abcd1234"
    )
    wrapper = FakeHttpxClientWrapper({"GET": [RuntimeError("boom")]})
    runtime = _runtime(wrapper, monkeypatch)
    logs: list[tuple[str, str, dict]] = []

    def _error(event: str, **kwargs) -> None:
        logs.append(("error", event, kwargs))

    monkeypatch.setattr("aibrix.batch.internal.octagram_runtime.logger.error", _error)

    with pytest.raises(RuntimeError, match="boom"):
        await runtime._octagram_request("GET", base_url, reason="test_reason")

    assert logs == [
        (
            "error",
            "octagram response",
            {
                "method": "GET",
                "url": base_url,
                "reason": "test_reason",
                "status_code": None,
                "phase": None,
                "replicasStatuses": None,
                "error": "boom",
            },
        )
    ]

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, cast

import httpx
import pytest

from aibrix import envs
from aibrix.batch.internal.octagram_runtime import OctagramHandle, OctagramRuntime
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


_PROVISIONED_CLUSTER = "cluster-a-HL"


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
    wait_calls: list[tuple[str, str, str, int]] = []

    async def _wait_ready(
        cluster: str,
        namespace: str,
        workload_name: str,
        replicas: int,
        wait_mode: str = "reconnect",
    ) -> None:
        del wait_mode
        wait_calls.append((cluster, namespace, workload_name, replicas))

    monkeypatch.setattr(runtime, "_wait_for_workload_ready", _wait_ready)

    await runtime._delete_workload(handle)

    assert wait_calls == [("cluster-a", "default", "batch-job-abcd1234", 0)]
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
        wait_mode: str = "reconnect",
    ) -> None:
        del cluster, namespace, workload_name, replicas, wait_mode
        raise BatchJobError(
            code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
            message="Timed out waiting for octagram workload to become ready",
        )

    monkeypatch.setattr(runtime, "_wait_for_workload_ready", _wait_ready)

    with pytest.raises(BatchJobError) as exc_info:
        await runtime._delete_workload(handle)

    assert exc_info.value.code == BatchJobErrorCode.RESOURCE_DELETION_ERROR.value
    assert "delete failed (500): delete boom" in exc_info.value.message
    assert "fallback scale to zero did not reach synced" in exc_info.value.message


@pytest.mark.asyncio
async def test_wait_for_workload_ready_returns_when_zero_replica_is_synced(
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

    await runtime._wait_for_workload_ready(
        cluster="cluster-a",
        namespace="default",
        workload_name="batch-job-abcd1234",
        replicas=0,
    )

    assert wrapper.async_client.calls == [("GET", base_url)]


@pytest.mark.asyncio
async def test_wait_for_workload_ready_404_raises_not_found_error(
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
        await runtime._wait_for_workload_ready(
            cluster="cluster-a",
            namespace="default",
            workload_name="batch-job-abcd1234",
            replicas=1,
            wait_mode="reconnect",
        )

    assert exc_info.value.code == BatchJobErrorCode.RESOURCE_NOTFOUND_ERROR.value
    assert "Octagram workload status check failed (404): gone" in exc_info.value.message


@pytest.mark.asyncio
async def test_wait_for_workload_ready_provision_retries_404_until_workload_exists(
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

    await runtime._wait_for_workload_ready(
        cluster="cluster-a",
        namespace="default",
        workload_name="batch-job-abcd1234",
        replicas=1,
        wait_mode="provision",
    )

    assert wrapper.async_client.calls == [("GET", base_url), ("GET", base_url)]


@pytest.mark.asyncio
async def test_wait_for_workload_ready_provision_404_respects_grace_window(
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
        await runtime._wait_for_workload_ready(
            cluster="cluster-a",
            namespace="default",
            workload_name="batch-job-abcd1234",
            replicas=1,
            wait_mode="provision",
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
    workload_wait_calls: list[tuple[str, str, str, int, str]] = []

    async def _wait_for_workload_ready(
        cluster: str,
        namespace: str,
        workload_name: str,
        replicas: int,
        wait_mode: str = "reconnect",
    ) -> None:
        workload_wait_calls.append(
            (cluster, namespace, workload_name, replicas, wait_mode)
        )

    runtime._wait_for_workload_ready = _wait_for_workload_ready  # type: ignore[method-assign]

    await runtime._wait_for_model_discoverable(_handle())

    assert workload_wait_calls == [
        ("cluster-a", "default", "batch-job-abcd1234", 1, "reconnect")
    ]
    assert len(model_discovery.calls) == 1
    assert model_discovery.calls[0]["served_model_name"] == "served-model"


@pytest.mark.asyncio
async def test_wait_ready_forwards_wait_mode_to_workload_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(FakeHttpxClientWrapper({"GET": []}), monkeypatch)
    readiness_wait_modes: list[str] = []

    async def _wait_for_workload_ready(
        cluster: str,
        namespace: str,
        workload_name: str,
        replicas: int,
        wait_mode: str = "reconnect",
    ) -> None:
        del cluster, namespace, workload_name, replicas
        readiness_wait_modes.append(wait_mode)

    async def _wait_for_model_discoverable(handle: OctagramHandle) -> None:
        del handle

    runtime._wait_for_workload_ready = _wait_for_workload_ready  # type: ignore[method-assign]
    runtime._wait_for_model_discoverable = _wait_for_model_discoverable  # type: ignore[method-assign]

    await runtime._wait_ready(_handle(), wait_mode="reconnect")

    assert readiness_wait_modes == ["reconnect"]


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
        renderer=FakeOctagramRenderer(),
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
        "psm": "fake-psm.service.hl",
        "baseUrl": None,
        "replicas": 1,
    }


@pytest.mark.asyncio
async def test_octagram_runtime_reconnect_accepts_current_payload_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _make_job()
    runtime = _runtime(FakeHttpxClientWrapper({"POST": []}), monkeypatch)

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
    assert runtime._active_handle == handle


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

    payload = await runtime._octagram_request("GET", base_url)

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
                "status_code": 200,
                "phase": "synced",
                "replicasStatuses": [{"name": "batch-job-abcd1234"}],
            },
        )
    ]


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

    await runtime._octagram_request("POST", base_url, json={"name": "demo"})

    assert logs == [
        (
            "info",
            "octagram response",
            {
                "method": "POST",
                "url": base_url,
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
        await runtime._octagram_request("GET", base_url)

    assert logs == [
        (
            "error",
            "octagram response",
            {
                "method": "GET",
                "url": base_url,
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
        await runtime._octagram_request("GET", base_url)

    assert logs == [
        (
            "error",
            "octagram response",
            {
                "method": "GET",
                "url": base_url,
                "status_code": None,
                "phase": None,
                "replicasStatuses": None,
                "error": "boom",
            },
        )
    ]

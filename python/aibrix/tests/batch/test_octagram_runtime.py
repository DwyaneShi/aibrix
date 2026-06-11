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

from typing import Optional

import httpx
import pytest

from aibrix import envs
from aibrix.batch.internal.octagram_runtime import OctagramHandle, OctagramRuntime
from aibrix.batch.job_entity import (
    BatchJob,
    BatchJobError,
    BatchJobErrorCode,
    BatchJobSpec,
)
from aibrix.batch.state import JobEntityManager
from aibrix.context import InfrastructureContext


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

    async def get_job(self, job_id: str) -> Optional[BatchJob]:
        return None

    async def list_jobs(self) -> list[BatchJob]:
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
    wrapper: FakeHttpxClientWrapper, monkeypatch: pytest.MonkeyPatch
) -> OctagramRuntime:
    monkeypatch.setattr(
        envs,
        "OCTAGRAM_GATEWAY_DOMAIN",
        "https://octagram-gateway.example.test",
    )
    return OctagramRuntime(
        InfrastructureContext(httpx_client_wrapper=wrapper),
        FakeEntityManager(),
    )


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
        cluster: str, namespace: str, workload_name: str, replicas: int
    ) -> None:
        wait_calls.append((cluster, namespace, workload_name, replicas))

    runtime._wait_for_workload_ready = _wait_ready  # type: ignore[method-assign]

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
        cluster: str, namespace: str, workload_name: str, replicas: int
    ) -> None:
        del cluster, namespace, workload_name, replicas
        raise BatchJobError(
            code=BatchJobErrorCode.RESOURCE_CREATION_ERROR,
            message="Timed out waiting for octagram workload to become ready",
        )

    runtime._wait_for_workload_ready = _wait_ready  # type: ignore[method-assign]

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
        )

    assert exc_info.value.code == BatchJobErrorCode.RESOURCE_NOTFOUND_ERROR.value
    assert "Octagram workload status check failed (404): gone" in exc_info.value.message


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

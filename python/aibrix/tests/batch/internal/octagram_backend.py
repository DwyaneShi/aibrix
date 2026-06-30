from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx

import aibrix.batch.internal.octagram_runtime as octagram_runtime_module
from aibrix import envs
from aibrix.batch.client.sources import NoopEndpointSource
from aibrix.batch.job_driver.runtime import Endpoint

ORIGINAL_OCTAGRAM_TEARDOWN = octagram_runtime_module.OctagramRuntime._teardown


class FastOctagramEndpointSource(NoopEndpointSource):
    def __init__(self, context):
        self._context = context
        super().__init__(delay=context.values.get("endpoint_source_delay_seconds", 0.0))


class FakeHttpxClientWrapper:
    def __init__(self, async_client: "FakeOctagramHttpClient"):
        self.async_client = async_client


class FakeOctagramHttpClient:
    def __init__(self) -> None:
        self.posts: list[str] = []
        self.gets: list[str] = []
        self.patches: list[str] = []
        self.deletes: list[str] = []
        self._existing_workloads: set[str] = set()

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        del kwargs
        request = httpx.Request(method, url)
        method = method.upper()
        if method == "POST":
            self.posts.append(url)
            self._existing_workloads.add(url)
            return httpx.Response(
                200,
                request=request,
                json={"data": {}, "error": "", "code": 0},
            )
        if method == "GET":
            self.gets.append(url)
            if url not in self._existing_workloads:
                return httpx.Response(404, request=request, text="gone")
            return httpx.Response(
                200,
                request=request,
                json={
                    "data": {
                        "metadata": {"name": url.rsplit("/", 1)[-1]},
                        "status": {
                            "phase": "synced",
                            "replicasStatuses": [{"available": 1}],
                        },
                    },
                    "error": "",
                    "code": 0,
                },
            )
        if method == "PATCH":
            self.patches.append(url)
            return httpx.Response(
                200,
                request=request,
                json={"data": {}, "error": "", "code": 0},
            )
        if method == "DELETE":
            self.deletes.append(url)
            if url not in self._existing_workloads:
                return httpx.Response(404, request=request, text="gone")
            self._existing_workloads.remove(url)
            return httpx.Response(
                200,
                request=request,
                json={"data": {}, "error": "", "code": 0},
            )
        raise AssertionError(f"unexpected method {method}")


@dataclass
class _FakeModelEndpoint:
    base_url: str


@dataclass
class _FakeLookupSnapshot:
    version: int
    endpoints: list[_FakeModelEndpoint]


class FakeModelDiscovery:
    def __init__(self) -> None:
        self.wait_calls: list[tuple[str, Optional[str], float]] = []

    async def lookup(
        self,
        service_id: Optional[str] = None,
        filter_tags: Optional[dict[str, str]] = None,
        lookup_timeout_seconds: Optional[float] = None,
    ) -> _FakeLookupSnapshot:
        del service_id, filter_tags, lookup_timeout_seconds
        return _FakeLookupSnapshot(
            version=1,
            endpoints=[_FakeModelEndpoint(base_url="http://octagram.local.test:8000")],
        )

    async def discover_model_endpoints(
        self,
        served_model_name: str,
        service_id: Optional[str] = None,
        filter_tags: Optional[dict[str, str]] = None,
        lookup_timeout_seconds: Optional[float] = None,
    ) -> _FakeLookupSnapshot:
        del served_model_name
        return await self.lookup(
            service_id=service_id,
            filter_tags=filter_tags,
            lookup_timeout_seconds=lookup_timeout_seconds,
        )

    async def wait_for_model_endpoints(
        self,
        served_model_name: str,
        timeout_seconds: float,
        service_id: Optional[str] = None,
        filter_tags: Optional[dict[str, str]] = None,
        lookup_timeout_seconds: Optional[float] = None,
        poll_interval_seconds: float = 1.0,
    ) -> _FakeLookupSnapshot:
        del filter_tags, lookup_timeout_seconds, poll_interval_seconds
        self.wait_calls.append((served_model_name, service_id, timeout_seconds))
        return await self.discover_model_endpoints(
            served_model_name,
            service_id=service_id,
        )


class FakeOctagramRenderer:
    def __init__(self, psm: str = "fake-psm") -> None:
        self.psm = psm
        self.template = None
        self.idc_name = "HL"

    def render(
        self,
        job_id: str,
        spec,
        provider_spec,
        namespace: str = "default",
    ) -> dict[str, Any]:
        del spec, provider_spec
        workload_name = f"batch-{job_id[:8]}"
        return {
            "apiVersion": "core.tce.byted.org/v1alpha1",
            "kind": "DeploymentWorkload",
            "metadata": {
                "name": workload_name,
                "namespace": namespace,
                "labels": {
                    "name": workload_name,
                    "batch.aibrix.ai/job_id": job_id,
                    "model.aibrix.ai/name": workload_name,
                    "psm": self.psm,
                },
            },
            "spec": {"deployStrategy": {"replicas": 1}},
        }


def configure_local_metastore_octagram_backend(app, monkeypatch) -> None:
    context = app.state.batch_driver._context
    shared_state = getattr(monkeypatch, "_aibrix_fake_octagram_backend_state", None)
    if shared_state is None:
        shared_state = {
            "http_client": FakeOctagramHttpClient(),
            "model_discovery": FakeModelDiscovery(),
            "octagram_teardown_calls": [],
            "octagram_endpoint_source_builds": [],
        }
        setattr(monkeypatch, "_aibrix_fake_octagram_backend_state", shared_state)

    http_client = shared_state["http_client"]
    model_discovery = shared_state["model_discovery"]
    context.httpx_client_wrapper = FakeHttpxClientWrapper(http_client)
    context.model_discovery = model_discovery
    context.values["service_http_client"] = http_client
    context.values["service_model_discovery"] = model_discovery
    context.values["runtime_teardown_calls"] = shared_state["octagram_teardown_calls"]
    context.values["octagram_endpoint_source_builds"] = shared_state[
        "octagram_endpoint_source_builds"
    ]

    monkeypatch.setattr(
        envs,
        "OCTAGRAM_GATEWAY_DOMAIN",
        "https://octagram-gateway.example.test",
    )
    monkeypatch.setattr(
        octagram_runtime_module.OctagramRuntime,
        "_build_renderer",
        staticmethod(lambda context: FakeOctagramRenderer()),
    )

    async def _connect_with_test_source(self, handle):
        self._context.values["octagram_endpoint_source_builds"].append(
            handle.workload_name
        )
        handle.source = FastOctagramEndpointSource(self._context)
        return Endpoint(source=handle.source, model_name=handle.model_name)

    async def _recording_teardown(self, handle):
        if handle is not None:
            self._context.values["runtime_teardown_calls"].append(
                {
                    "job_id": self._active_job_id,
                    "workload_name": handle.workload_name,
                }
            )
        return await ORIGINAL_OCTAGRAM_TEARDOWN(self, handle)

    monkeypatch.setattr(
        octagram_runtime_module.OctagramRuntime,
        "_connect",
        _connect_with_test_source,
    )
    monkeypatch.setattr(
        octagram_runtime_module.OctagramRuntime,
        "_teardown",
        _recording_teardown,
    )

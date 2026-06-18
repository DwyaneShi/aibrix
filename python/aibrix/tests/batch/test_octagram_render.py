import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    import redis.asyncio as _redis_asyncio
except ModuleNotFoundError:
    redis_module = sys.modules.setdefault("redis", types.ModuleType("redis"))
    redis_asyncio_module = types.ModuleType("redis.asyncio")

    class _RedisStub:
        def __init__(self, *_args, **_kwargs):
            self._init_args = (_args, _kwargs)

    setattr(redis_asyncio_module, "Redis", _RedisStub)
    setattr(redis_module, "asyncio", redis_asyncio_module)
    sys.modules["redis.asyncio"] = redis_asyncio_module
else:
    _redis_asyncio

from aibrix import envs
from aibrix.batch.internal.octagram_renderer import OctagramManifestRenderer
from aibrix.batch.job_entity import BatchJobSpec, ResourceDetail, ResourceRequirement

_TESTDATA_DIR = Path(__file__).parent / "testdata"
_FIXED_NOW = datetime(2026, 5, 22, 18, 0, tzinfo=timezone.utc)


def _template_spec(service_id: str | None = "inf.aibrix.platform"):
    spec = {
        "engine": {
            "type": "vllm",
            "version": "0.89",
            "image": "hub.byted.org/aibrix/inf.aibrix.vllm:1.0.0.150",
            "invocation": "http_server",
        },
        "model_source": {
            "type": "hdfs",
            "uri": (
                "hdfs://haruna/home/byte_mlsys_bernard/ssd/models/"
                "Qwen3.5-0.8B-Merlin-HF/1.0.c84b5056/model_store_open_source_model"
            ),
        },
        "accelerator": {
            "type": "NVIDIA-A10",
            "count": 1,
            "interconnect": "pcie",
            "vram_gb": 24,
        },
        "parallelism": {"tp": 1, "pp": 1, "dp": 1},
        "engine_args": {"gpu_memory_utilization": 0.85},
        "supported_endpoints": ["/v1/chat/completions"],
        "deployment_mode": "dedicated",
    }
    if service_id is not None:
        spec["service_id"] = service_id
    return spec


def _request_snapshot_template_spec():
    spec = _template_spec()
    spec["engine"]["image"] = "hub.byted.org/aibrix/inf.aibrix.vllm:1.0.0.135_nydus"
    return spec


def _resource_detail(
    *,
    endpoint_cluster: str = "Echo-HL",
    resource_pool_name: str = ("compute-3530-hl-echo-ai-default"),
    salemode: str = "scheduled",
    qos_level: str = "shared_cores",
    logical_cluster: str = "ai",
    accelerator_type: str = "NVIDIA-H20",
    accelerator_category: str = "gpu",
    cpu: str | None = "11",
    memory: str | None = "125Gi",
    accelerator_count: int = 1,
    replica: int = 1,
) -> ResourceDetail:
    return ResourceDetail(
        provider="tce",
        endpoint_cluster=endpoint_cluster,
        resource_pool_name=resource_pool_name,
        salemode=salemode,
        qos_level=qos_level,
        logical_cluster=logical_cluster,
        resources=[
            ResourceRequirement(
                accelerator_type=accelerator_type,
                accelerator_category=accelerator_category,
                cpu=cpu,
                memory=memory,
                accelerator_count=accelerator_count,
                replica=replica,
            )
        ],
    )


def _request_snapshot_resource_detail() -> ResourceDetail:
    return _resource_detail(
        endpoint_cluster="Federation-YG",
        resource_pool_name=(
            "compute-3530-yg-federationgpu-non.standard.g19-default-guarantee"
        ),
        logical_cluster="non-standard-g19",
        accelerator_type="NVIDIA-A30",
    )


def _spec(
    *,
    template_name: str = "tce-vllm",
    template_spec: dict | None = None,
    completion_window: str = "24h",
    resource_allocation: dict | None = None,
):
    aibrix = {
        "model_template": {
            "name": template_name,
            "spec": template_spec or _template_spec(),
        }
    }
    if resource_allocation is not None:
        aibrix["resource_allocation"] = resource_allocation
    return BatchJobSpec.from_strings(
        input_file_id="file-1",
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        aibrix=aibrix,
    )


def _rendered_env(rendered: dict[str, object]) -> dict[str, str]:
    container = rendered["spec"]["podBase"]["containers"][0]
    return {item["name"]: item["value"] for item in container["env"]}


def _render(
    *,
    job_id: str = "6281d2a8-6281ds2a8-a2",
    spec: BatchJobSpec | None = None,
    detail: ResourceDetail | None = None,
    now_provider=None,
):
    renderer = OctagramManifestRenderer(
        now_provider=now_provider or (lambda: _FIXED_NOW)
    )
    return renderer.render(
        job_id=job_id,
        spec=spec or _spec(),
        prividerSpec=detail or _resource_detail(),
    )


def _load_testdata_yaml(name: str):
    return yaml.safe_load((_TESTDATA_DIR / name).read_text())


def _normalized_request_snapshot_yaml():
    expected = _load_testdata_yaml("deploymentworkload_example_request.yaml")
    return expected


def test_parse_endpoint_cluster_preserves_cluster_prefix_and_idc_suffix():
    renderer = OctagramManifestRenderer()

    cluster, idc = renderer._parse_endpoint_cluster("Bernard-Prod-HL")

    assert cluster == "Bernard-Prod"
    assert idc == "HL"


def test_octagram_manifest_renderer_matches_example_request_yaml():
    rendered = _render(
        spec=_spec(template_spec=_request_snapshot_template_spec()),
        detail=_request_snapshot_resource_detail(),
    )

    assert rendered == _normalized_request_snapshot_yaml()


def test_octagram_manifest_renderer_renders_expected_tce_fields():
    rendered = _render(
        job_id="6281d2a8-6281ds2a8-a2",
        spec=_spec(),
        detail=_resource_detail(),
    )

    assert rendered["metadata"]["name"] == "batch-tce-vllm-6281d2a8"
    assert rendered["metadata"]["labels"]["model.aibrix.ai/name"] == (
        "batch-tce-vllm-6281d2a8-model_store_open_source_model"
    )
    assert rendered["metadata"]["labels"]["psm"] == "inf.aibrix.platform"

    feature_gates = rendered["spec"]["featureGates"]
    assert {"name": "UseScheduledResource", "value": True} in feature_gates
    assert {"name": "KatalystQosLevel", "value": "shared_cores"} in feature_gates

    deployment_annotations = rendered["spec"]["deploymentMeta"]["annotations"]
    assert deployment_annotations["bytedance.quota.salemode"] == "scheduled"
    assert (
        deployment_annotations["deployment.tce.kubernetes.io/requestCpuUserDemand"]
        == "11"
    )
    assert (
        deployment_annotations["deployment.tce.kubernetes.io/requestMemUserDemand"]
        == "125Gi"
    )

    resources = rendered["spec"]["podBase"]["containers"][0]["resources"]
    assert resources["gpu"] == {"request": "1", "limit": "1"}
    assert resources["cpu"] == {"request": "11", "limit": "11"}
    assert resources["memory"] == {"request": "125Gi", "limit": "125Gi"}

    env = _rendered_env(rendered)
    assert env["MODEL_NAME"] == "model_store_open_source_model"
    assert env["EXTRA_PARAMETERS"] == "--gpu-memory-utilization 0.85"
    assert env["TCE_CLUSTER"] == "Echo"
    assert env["TCE_INTERNAL_IDC"] == "HL"
    assert env["LOAD_SERVICE_PSM"] == "inf.aibrix.platform"

    treatments = rendered["spec"]["treatments"]
    assert treatments[0] == {
        "type": "PreTreatment",
        "data": {
            "apiVersion": "core.tce.byted.org/v1alpha1",
            "kind": "IdentityTreatment",
            "metadata": {
                "name": "batch-tce-vllm-6281d2a8",
                "namespace": "default",
            },
            "spec": {
                "authType": "psm",
                "user": "jingyuan.zhang0929",
            },
            "status": {},
        },
    }


def test_octagram_manifest_renderer_uses_workload_psm_for_log_host_paths():
    rendered = _render(spec=_spec())
    volumes = rendered["spec"]["podBase"]["volumes"]
    log_volume_paths = {
        volume["name"]: volume["hostPath"]["path"]
        for volume in volumes
        if volume["name"]
        in {"opt-tiger-data-log", "opt-tiger-toutiao-log", "var-log-tiger"}
    }

    assert log_volume_paths == {
        "opt-tiger-data-log": "/opt/tiger/tce/containers/inf.aibrix.platform",
        "opt-tiger-toutiao-log": "/opt/tiger/tce/containers/inf.aibrix.platform",
        "var-log-tiger": "/opt/tiger/tce/containers/inf.aibrix.platform",
    }


def test_octagram_manifest_renderer_clamps_window_to_resource_allocation_deadline():
    detail = _resource_detail(
        endpoint_cluster="Federation-LQ",
        resource_pool_name=("compute-3530-lq-federationgpu-default-default-guarantee"),
        logical_cluster="default",
        accelerator_type="NVIDIA-A10",
        cpu="8",
        memory="32Gi",
    )
    deadline = int(datetime(2026, 5, 22, 19, 0, tzinfo=timezone.utc).timestamp())
    spec = _spec(
        template_name="tce-vllm-test",
        resource_allocation={
            "provision_resource_deadline": deadline,
            "resource_details": [detail.model_dump(exclude_none=True)],
        },
    )

    rendered = _render(
        job_id="6281d2a8-6281ds2a8-a2",
        spec=spec,
        detail=spec.aibrix.resource_allocation.resource_details[0],
    )

    rules = rendered["spec"]["treatments"][1]["data"]["spec"]["hpaExtensionConfig"][
        "resourceUtilizationConfig"
    ]["rules"]
    assert rules[0]["effectiveWindow"] == {
        "startTime": "202605221800",
        "lastMinutes": 60,
        "location": "utc",
    }
    assert rules[1] == {
        "maxReplica": 0,
        "minReplica": 0,
        "resourcePercentage": {"gpu": 1, "cpu": 1, "memory": 1},
    }


def test_octagram_manifest_renderer_maps_xpu_category_to_xpu_annotation():
    rendered = _render(
        detail=_resource_detail(
            accelerator_type="BI-XPU",
            accelerator_category="xpu",
            accelerator_count=2,
        )
    )

    deployment_annotations = rendered["spec"]["deploymentMeta"]["annotations"]
    resources = rendered["spec"]["podBase"]["containers"][0]["resources"]
    rules = rendered["spec"]["treatments"][1]["data"]["spec"]["hpaExtensionConfig"][
        "resourceUtilizationConfig"
    ]["rules"]

    assert "deployment.tce.kubernetes.io/gpu-type" not in deployment_annotations
    assert deployment_annotations["deployment.tce.kubernetes.io/xpu-type"] == "BI-XPU"
    assert resources["xpu"] == {"request": "2", "limit": "2"}
    assert rules[0]["resourcePercentage"] == {"xpu": 1, "cpu": 1, "memory": 1}
    assert rules[1]["resourcePercentage"] == {"xpu": 1, "cpu": 1, "memory": 1}


def test_octagram_manifest_renderer_maps_npu_category_to_habana_annotation():
    rendered = _render(
        detail=_resource_detail(
            accelerator_type="HL-325",
            accelerator_category="npu",
        )
    )

    deployment_annotations = rendered["spec"]["deploymentMeta"]["annotations"]
    resources = rendered["spec"]["podBase"]["containers"][0]["resources"]
    rules = rendered["spec"]["treatments"][1]["data"]["spec"]["hpaExtensionConfig"][
        "resourceUtilizationConfig"
    ]["rules"]

    assert "deployment.tce.kubernetes.io/gpu-type" not in deployment_annotations
    assert (
        deployment_annotations["deployment.tce.kubernetes.io/habana-type"] == "HL-325"
    )
    assert resources["npu"] == {"request": "1", "limit": "1"}
    assert rules[0]["resourcePercentage"] == {"npu": 1, "cpu": 1, "memory": 1}
    assert rules[1]["resourcePercentage"] == {"npu": 1, "cpu": 1, "memory": 1}


def test_octagram_manifest_renderer_includes_cpu_and_memory_in_hpa_resource_percentage():
    rendered = _render(detail=_resource_detail(accelerator_count=4, replica=3))

    rules = rendered["spec"]["treatments"][1]["data"]["spec"]["hpaExtensionConfig"][
        "resourceUtilizationConfig"
    ]["rules"]

    assert rules[0]["maxReplica"] == 3
    assert rules[0]["minReplica"] == 3
    assert rules[0]["resourcePercentage"] == {"gpu": 1, "cpu": 1, "memory": 1}
    assert rules[1]["resourcePercentage"] == {"gpu": 1, "cpu": 1, "memory": 1}


def test_octagram_manifest_renderer_omits_empty_cpu_and_memory_fields():
    rendered = _render(
        job_id="6281d2a8-6281ds2a8-a2",
        spec=_spec(),
        detail=_resource_detail(cpu=None, memory=None, accelerator_count=0),
    )

    deployment_annotations = rendered["spec"]["deploymentMeta"]["annotations"]
    pod_annotations = rendered["spec"]["podBase"]["annotations"]
    resources = rendered["spec"]["podBase"]["containers"][0]["resources"]
    rules = rendered["spec"]["treatments"][1]["data"]["spec"]["hpaExtensionConfig"][
        "resourceUtilizationConfig"
    ]["rules"]

    assert (
        "deployment.tce.kubernetes.io/requestCpuUserDemand"
        not in deployment_annotations
    )
    assert (
        "deployment.tce.kubernetes.io/requestMemUserDemand"
        not in deployment_annotations
    )
    assert "pod.tce.kubernetes.io/requestCpuUserDemand" not in pod_annotations
    assert "pod.tce.kubernetes.io/requestMemUserDemand" not in pod_annotations
    assert "cpu" not in resources
    assert "memory" not in resources
    assert rules[0]["resourcePercentage"] == {"gpu": 1}
    assert rules[1]["resourcePercentage"] == {"gpu": 1}


def test_octagram_manifest_renderer_adds_scheduled_feature_gate_conditionally():
    scheduled = _render(
        job_id="6281d2a8-6281ds2a8-a2",
        spec=_spec(),
        detail=_resource_detail(salemode="scheduled"),
    )
    unscheduled = _render(
        job_id="7281d2a8-6281ds2a8-a2",
        spec=_spec(),
        detail=_resource_detail(salemode=""),
    )

    assert {"name": "UseScheduledResource", "value": True} in scheduled["spec"][
        "featureGates"
    ]
    assert {"name": "UseScheduledResource", "value": True} not in unscheduled["spec"][
        "featureGates"
    ]


def test_octagram_manifest_renderer_defaults_psm_from_env_when_template_missing(
    monkeypatch,
):
    monkeypatch.setattr(envs, "CONSUL_BATCH_DISCOVERY_PSM", "env.default.psm")

    spec = _spec(
        template_name="ning-a10",
        template_spec=_template_spec(service_id=None),
    )

    rendered = _render(
        job_id="089890d4-1113-450f-b1c7-9b2d5cd55510",
        spec=spec,
        detail=_resource_detail(
            resource_pool_name="compute-3530-yg-federationgpu-default-default",
            qos_level="shared",
            logical_cluster="default",
            accelerator_type="NVIDIA-A10",
            cpu=None,
            memory=None,
        ),
    )

    assert rendered["metadata"]["labels"]["psm"] == "env.default.psm"
    assert rendered["spec"]["deploymentMeta"]["labels"]["psm"] == "env.default.psm"
    assert rendered["spec"]["deployStrategy"]["selector"]["matchLabels"]["psm"] == (
        "env.default.psm"
    )


def test_octagram_manifest_renderer_writes_json_request_for_test_template(tmp_path):
    rendered = _render(
        job_id="089890d4-1113-450f-b1c7-9b2d5cd55510",
        spec=_spec(),
        detail=_resource_detail(),
        now_provider=lambda: datetime.now(timezone.utc),
    )
    json_path = tmp_path / "deploymentworkload_example_request_test.json"

    json_path.write_text(json.dumps(rendered, indent=2) + "\n")

    assert json.loads(json_path.read_text()) == rendered

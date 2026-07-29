from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any, Callable, Dict, List, Optional

from aibrix import envs
from aibrix.batch.internal.octagram_utils import parse_endpoint_cluster
from aibrix.batch.job_entity import BatchJobSpec, ResourceDetail, ResourceRequirement
from aibrix.batch.manifest.engine_adapter import build_engine_args
from aibrix.batch.manifest.renderer import _RendererSupport
from aibrix.batch.template import ModelDeploymentTemplate
from aibrix.downloader.utils import infer_model_name
from aibrix.logger import init_logger

logger = init_logger(__name__)

# Renderer defaults.
_DEFAULT_NAMESPACE = "default"
_DEFAULT_TCE_ENV = "prod"
_DEFAULT_TCE_STAGE = "all_dc"
_DEFAULT_TCE_PRIMARY_PORT = "fake_port"
_DEFAULT_IDENTITY_TREATMENT_USER = getattr(
    envs, "OCTAGRAM_IDENTITY_USER", "jingyuan.zhang0929"
)

# Compatibility fallbacks for callers that bypass the Console planner.
_FALLBACK_CPU_CORES_PER_GPU = 16
_FALLBACK_MEMORY_GIB_PER_GPU = 96
_MAX_MODEL_IDENTITY_LENGTH = 63
_INVALID_MODEL_IDENTITY_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

# TCE filesystem and container lifecycle contract.
_BERNARD_HOST_PATH = "/opt/tiger/bernard"
_BERNARD_TOOLS_HOST_PATH = f"{_BERNARD_HOST_PATH}/bernard_tools"
_PRE_STOP_COMMAND = (
    "bash",
    "-c",
    (
        f"{_BERNARD_TOOLS_HOST_PATH}/bin/pre_stop; "
        "/home/tiger/.op/docker_pre_stop.sh;"
    ),
)


@dataclass(frozen=True)
class _ExecProbeConfig:
    command: tuple[str, ...]
    failure_threshold: int
    period_seconds: int
    initial_delay_seconds: Optional[int] = None
    success_threshold: int = 1
    timeout_seconds: int = 30

    def to_manifest(self) -> Dict[str, Any]:
        probe: Dict[str, Any] = {"exec": {"command": list(self.command)}}
        if self.initial_delay_seconds is not None:
            probe["initialDelaySeconds"] = self.initial_delay_seconds
        probe.update(
            {
                "failureThreshold": self.failure_threshold,
                "periodSeconds": self.period_seconds,
                "successThreshold": self.success_threshold,
                "timeoutSeconds": self.timeout_seconds,
            }
        )
        return probe


_LIVENESS_PROBE = _ExecProbeConfig(
    command=(
        "bash",
        "-c",
        f"{_BERNARD_TOOLS_HOST_PATH}/bin/liveness_check.sh",
    ),
    initial_delay_seconds=180,
    failure_threshold=5,
    period_seconds=60,
)
_READINESS_PROBE = _ExecProbeConfig(
    command=(
        "bash",
        "-c",
        f"{_BERNARD_TOOLS_HOST_PATH}/bin/readiness_check.sh",
    ),
    failure_threshold=2,
    period_seconds=20,
)


# The matching/planner layer uses fully-qualified accelerator SKU names while
# Octagram nodes are labeled with vendor-stripped names. Translate on dispatch;
# unmapped values pass through unchanged.
_OCTAGRAM_ACCELERATOR_TYPE_MAPPING = {
    "NVIDIA-A100-SXM4-80GB": "A100-SXM-80GB",
}

_OCTAGRAM_XPU_RESOURCE_NAME = "bytedance.com/xpu"
_OCTAGRAM_NPU_RESOURCE_NAME = "habana.ai/goya"


def _sanitize_model_identity(value: str) -> str:
    normalized = _INVALID_MODEL_IDENTITY_CHARS.sub("-", value.lower()).strip("-_.")
    return normalized[:_MAX_MODEL_IDENTITY_LENGTH].rstrip("-_.")


def _build_served_model_name(job_name: str, model_name: str) -> str:
    return _sanitize_model_identity(f"{job_name}-{model_name}")


def _map_accelerator_type(accelerator_type: Optional[str]) -> str:
    if not accelerator_type:
        return ""
    return _OCTAGRAM_ACCELERATOR_TYPE_MAPPING.get(accelerator_type, accelerator_type)


def _accelerator_category(resource: ResourceRequirement) -> str:
    return (resource.accelerator_category or "gpu").lower()


def _deployment_accelerator_type_annotation(resource: ResourceRequirement) -> str:
    category = _accelerator_category(resource)
    if category == "xpu":
        return "deployment.tce.kubernetes.io/xpu-type"
    if category == "npu":
        # Octagram uses habana-type for NPU-class accelerators.
        return "deployment.tce.kubernetes.io/habana-type"
    return "deployment.tce.kubernetes.io/gpu-type"


_BASE_FEATURE_GATES = [
    {"name": "ContainerResourceView", "value": "container"},
    {"name": "ContainerShmSize", "value": 100000},
    {"name": "IPv6OnlyCompatible", "value": True},
]

_VOLUME_MOUNTS = [
    {"name": "bernard", "mountPath": _BERNARD_HOST_PATH, "readOnly": True},
    {
        "name": "opt-tiger-data-log",
        "mountPath": "/opt/tiger/data/log",
        "subPath": "$(MY_POD_NAME)/data/log",
    },
    {
        "name": "opt-tiger-toutiao-log",
        "mountPath": "/opt/tiger/toutiao/log",
        "subPath": "$(MY_POD_NAME)/toutiao/log",
    },
    {"name": "run", "mountPath": "/run"},
    {
        "name": "var-log-tiger",
        "mountPath": "/var/log/tiger",
        "subPath": "$(MY_POD_NAME)/var/log/tiger",
    },
    {"name": "yarn-deploy", "mountPath": "/opt/tiger/yarn_deploy", "readOnly": True},
]

_VOLUMES = [
    {
        "name": "bernard",
        "hostPath": {"path": _BERNARD_HOST_PATH, "type": ""},
    },
    {"name": "run", "emptyDir": {"medium": "Memory", "sizeLimit": "64Mi"}},
    {"name": "yarn-deploy", "hostPath": {"path": "/opt/tiger/yarn_deploy", "type": ""}},
]


def _log_host_path(psm: str) -> Dict[str, str]:
    return {
        "path": f"/opt/tiger/tce/containers/{psm}",
        "type": "DirectoryOrCreate",
    }


def _volumes(psm: Optional[str]) -> List[Dict[str, Any]]:
    volumes = deepcopy(_VOLUMES)
    if not psm:
        return volumes

    log_volumes = [
        {"name": "opt-tiger-data-log", "hostPath": _log_host_path(psm)},
        {"name": "opt-tiger-toutiao-log", "hostPath": _log_host_path(psm)},
        {"name": "var-log-tiger", "hostPath": _log_host_path(psm)},
    ]
    volumes.extend(log_volumes)
    return volumes


class OctagramManifestRenderer(_RendererSupport):
    def __init__(
        self,
        *args,
        now_provider: Optional[Callable[[], datetime]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self.template: Optional[ModelDeploymentTemplate] = None

    def render(
        self,
        job_id: str,
        job_name: str,
        spec: BatchJobSpec,
        providerSpec: ResourceDetail,
        namespace: str = _DEFAULT_NAMESPACE,
        tce_env: str = _DEFAULT_TCE_ENV,
        tce_stage: str = _DEFAULT_TCE_STAGE,
        tce_primary_port: str = _DEFAULT_TCE_PRIMARY_PORT,
    ) -> Dict[str, Any]:
        template, _ = self._resolve(spec)
        self.template = template

        # Validate the supportable value space up-front so downstream
        # layers can assume they're working with k8s + dedicated + supported endpoint.
        self._validate_template(template, spec.endpoint)

        container_name = job_name
        resource = providerSpec.resource
        # Keep default resource sizing at the renderer boundary instead of
        # duplicating backend-specific defaults in the Console planner.
        # Explicit resource allocation values still take precedence.
        gpu_count = resource.accelerator_count or 0
        if gpu_count > 0:
            if not resource.cpu:
                resource.cpu = str(gpu_count * _FALLBACK_CPU_CORES_PER_GPU)
            if not resource.memory:
                resource.memory = f"{gpu_count * _FALLBACK_MEMORY_GIB_PER_GPU}Gi"
        model_name = (
            spec.model or infer_model_name(template.spec.model_source.uri)
        ).lower()
        served_model_name = _build_served_model_name(job_name, model_name)
        console_job_id = spec.aibrix.job_id if spec.aibrix else None
        _, idc, physical_cluster, _ = parse_endpoint_cluster(
            providerSpec.endpoint_cluster
        )
        self.cluster_name = physical_cluster
        self.idc_name = idc

        # Layered composition.
        manifest = self._system_base(
            job_id=job_id,
            console_job_id=console_job_id,
            job_name=job_name,
            served_model_name=served_model_name,
            namespace=namespace,
        )
        self._apply_template_psm(manifest, template)
        self._apply_feature_gates(manifest, providerSpec)
        self._apply_deployment_meta(manifest, providerSpec, resource)
        self._apply_strategy(manifest, resource)
        self._apply_pod_base(
            job_name=job_name,
            manifest=manifest,
            template=template,
            detail=providerSpec,
            resource=resource,
            container_name=container_name,
            cluster_name=physical_cluster,
            idc_name=idc,
            tce_env=tce_env,
            tce_stage=tce_stage,
            tce_primary_port=tce_primary_port,
        )
        self._apply_treatments(
            manifest,
            spec=spec,
            resource=resource,
            job_name=job_name,
            namespace=namespace,
        )
        return manifest

    def _system_base(
        self,
        job_id: str,
        console_job_id: Optional[str],
        job_name: str,
        served_model_name: str,
        namespace: str,
    ) -> Dict[str, Any]:
        labels = {
            "name": job_name,
            "batch.aibrix.ai/job_id": job_id,
            "model.aibrix.ai/name": served_model_name,
            # malachite-reporter relies on this label to determine whether
            # to enable bernard metrics reporting
            "cloudnative-application-platform": "mlsys.bernard",
        }
        if console_job_id:
            labels["batch.aibrix.ai/console_job_id"] = console_job_id
        return {
            "apiVersion": "core.tce.byted.org/v1alpha1",
            "kind": "DeploymentWorkload",
            "metadata": {
                "name": job_name,
                "namespace": namespace,
                "annotations": {
                    # This is flag deprecated, use full-sync-trigger later to trigger after synced
                    "core.tce.byted.org/updatetrigger": "true"
                    # "controller.octagram.io/full-sync-trigger": self._now().isoformat(
                    #     timespec="seconds"
                    # ).replace("+00:00", "Z")
                },
                "labels": labels,
            },
            "spec": {},
        }

    def _apply_template_psm(
        self, manifest: Dict[str, Any], template: ModelDeploymentTemplate
    ) -> None:
        psm = template.spec.service_id or envs.CONSUL_BATCH_DISCOVERY_PSM
        if psm:
            manifest["metadata"]["labels"]["psm"] = psm

    def _apply_feature_gates(
        self, manifest: Dict[str, Any], provider: ResourceDetail
    ) -> None:
        feature_gates = deepcopy(_BASE_FEATURE_GATES)
        if provider.salemode == "scheduled":
            feature_gates.append({"name": "UseScheduledResource", "value": True})
        if provider.qos_level == "shared_cores":
            feature_gates.extend(
                [
                    {
                        "name": "KatalystMemoryEnhancement",
                        "value": {"numa_binding": "false"},
                    },
                    {
                        "name": "KatalystQosEnhancement",
                        "value": {
                            "katalyst.kubewharf.io/memory_enhancement": '{"numa_binding":"false","numa_exclusive":"false"}'
                        },
                    },
                ]
            )

        feature_gates.append({"name": "KatalystQosLevel", "value": provider.qos_level})
        manifest["spec"]["featureGates"] = feature_gates

    def _apply_deployment_meta(
        self,
        manifest: Dict[str, Any],
        detail: ResourceDetail,
        resource: ResourceRequirement,
    ) -> None:
        annotations = {
            "bytedance.quota.salemode": detail.salemode or "",
            _deployment_accelerator_type_annotation(resource): _map_accelerator_type(
                resource.accelerator_type
            ),
            "deployment.tce.kubernetes.io/requestGpuUserDemand": str(
                resource.accelerator_count or 0
            ),
            "queue-name": detail.resource_pool_name or "",
            "vpa.katalyst.kubewharf.io/enable": "true",
        }
        if resource.cpu:
            annotations["deployment.tce.kubernetes.io/requestCpuUserDemand"] = (
                resource.cpu
            )
        if resource.memory:
            annotations["deployment.tce.kubernetes.io/requestMemUserDemand"] = (
                resource.memory
            )
        manifest["spec"]["deploymentMeta"] = {
            "labels": {
                "name": manifest["metadata"]["name"],
                "batch.aibrix.ai/job_id": manifest["metadata"]["labels"][
                    "batch.aibrix.ai/job_id"
                ],
                "model.aibrix.ai/name": manifest["metadata"]["labels"][
                    "model.aibrix.ai/name"
                ],
                # malachite-reporter relies on this label to determine whether
                # to enable bernard metrics reporting
                "cloudnative-application-platform": "mlsys.bernard",
            },
            "annotations": annotations,
        }
        psm = manifest["metadata"]["labels"].get("psm")
        if psm:
            manifest["spec"]["deploymentMeta"]["labels"]["psm"] = psm
        console_job_id = manifest["metadata"]["labels"].get(
            "batch.aibrix.ai/console_job_id"
        )
        if console_job_id:
            manifest["spec"]["deploymentMeta"]["labels"][
                "batch.aibrix.ai/console_job_id"
            ] = console_job_id

    def _apply_strategy(
        self,
        manifest: Dict[str, Any],
        resource: ResourceRequirement,
    ) -> None:
        replicas = resource.replica or 1
        manifest["spec"]["deployStrategy"] = {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "name": manifest["metadata"]["name"],
                }
            },
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {"maxSurge": "25%", "maxUnavailable": "25%"},
            },
            "minReadySeconds": 10,
            "revisionHistoryLimit": 5,
            "progressDeadlineSeconds": 2147483647,
        }

    def _apply_pod_base(
        self,
        job_name: str,
        manifest: Dict[str, Any],
        template: ModelDeploymentTemplate,
        detail: ResourceDetail,
        resource: ResourceRequirement,
        container_name: str,
        cluster_name: str,
        idc_name: str,
        tce_env: str,
        tce_stage: str,
        tce_primary_port: str,
    ) -> None:
        annotations = {
            "AvailableZoneRebalance": "true",
            "bytedance.com/main-container-name": container_name,
            "bytedance.quota.salemode": detail.salemode or "",
            "godel.bytedance.com/without-preemption-protection": "true",
            "pod.kubernetes.io/explicit.deletion": "true",
            "pod.tce.kubernetes.io/autoport": "1",
            "pod.tce.kubernetes.io/requestGpuUserDemand": str(
                resource.accelerator_count or 0
            ),
            "queue-name": detail.resource_pool_name or "",
        }
        if resource.cpu:
            annotations["pod.tce.kubernetes.io/requestCpuUserDemand"] = resource.cpu
        if resource.memory:
            annotations["pod.tce.kubernetes.io/requestMemUserDemand"] = resource.memory
        manifest["spec"]["podBase"] = {
            "annotations": annotations,
            "containers": [
                {
                    "name": container_name,
                    "image": template.spec.engine.image,
                    "imagePullPolicy": "IfNotPresent",
                    "isMainContainer": True,
                    "env": self._container_env(
                        job_name=job_name,
                        manifest=manifest,
                        template=template,
                        cluster_name=cluster_name,
                        idc_name=idc_name,
                        tce_env=tce_env,
                        tce_stage=tce_stage,
                        tce_primary_port=tce_primary_port,
                        accelerator_count=resource.accelerator_count or 0,
                    ),
                    "ports": [
                        {"containerPort": 0, "protocol": "TCP"} for _ in range(10)
                    ],
                    "resources": self._container_resources(resource),
                    "volumeMounts": deepcopy(_VOLUME_MOUNTS),
                    "livenessProbe": _LIVENESS_PROBE.to_manifest(),
                    "readinessProbe": _READINESS_PROBE.to_manifest(),
                    "lifecycle": {
                        "preStop": {
                            "exec": {"command": list(_PRE_STOP_COMMAND)},
                        }
                    },
                }
            ],
            "nodeSelector": {
                "accelerator": _map_accelerator_type(resource.accelerator_type),
                "nodeLevel": detail.logical_cluster or "",
            },
            "hostNetwork": True,
            "terminationGracePeriodSeconds": 30,
            "volumes": _volumes(manifest["metadata"]["labels"].get("psm")),
        }

    def _apply_treatments(
        self,
        manifest: Dict[str, Any],
        spec: BatchJobSpec,
        resource: ResourceRequirement,
        job_name: str,
        namespace: str,
    ) -> None:
        treatments: List[Dict[str, Any]] = [
            self._build_identity_treatment(job_name=job_name, namespace=namespace)
        ]
        treatment = self._build_hpa_time_window_treatment(
            manifest=manifest,
            spec=spec,
            resource=resource,
            job_name=job_name,
            namespace=namespace,
        )
        if treatment is not None:
            treatments.append(treatment)
        if treatments:
            manifest["spec"]["treatments"] = treatments
        return None

    def _build_identity_treatment(
        self,
        job_name: str,
        namespace: str,
    ) -> Dict[str, Any]:
        return {
            "type": "PreTreatment",
            "data": {
                "apiVersion": "core.tce.byted.org/v1alpha1",
                "kind": "IdentityTreatment",
                "metadata": {
                    "name": job_name,
                    "namespace": namespace,
                },
                "spec": {
                    "authType": "psm",
                    "user": _DEFAULT_IDENTITY_TREATMENT_USER,
                },
                "status": {},
            },
        }

    def _build_hpa_time_window_treatment(
        self,
        manifest: Dict[str, Any],
        spec: BatchJobSpec,
        resource: ResourceRequirement,
        job_name: str,
        namespace: str,
    ) -> Optional[Dict[str, Any]]:
        now = self._now()
        window_end = self._resolve_window_end(spec, now)
        if window_end is None:
            return None

        # Octagram only materializes resourceUtilization rules when a metric
        # target is present, even if fixed min/max replicas drive the behavior.
        resource_percentage = self._build_hpa_resource_percentage(resource)
        active_replicas = resource.replica or 1
        rules: List[Dict[str, Any]] = []

        if window_end > now:
            rules.append(
                {
                    "effectiveWindow": {
                        "startTime": now.strftime("%Y%m%d%H%M"),
                        "lastMinutes": max(
                            1,
                            ceil((window_end - now).total_seconds() / 60),
                        ),
                        "location": "utc",
                    },
                    "maxReplica": active_replicas,
                    "minReplica": active_replicas,
                    "resourcePercentage": resource_percentage,
                }
            )

        rules.append(
            {
                "maxReplica": 0,
                "minReplica": 0,
                "resourcePercentage": resource_percentage,
            }
        )

        return {
            "type": "PostTreatment",
            "data": {
                "apiVersion": "core.tce.byted.org/v1alpha1",
                "kind": "HPATreatment",
                "metadata": {
                    "name": job_name,
                    "namespace": namespace,
                },
                "spec": {
                    "hpaExtensionConfig": {
                        "resourceUtilizationConfig": {
                            "rules": rules,
                        }
                    }
                },
                "status": {},
            },
        }

    def _resolve_window_end(
        self,
        spec: BatchJobSpec,
        now: datetime,
    ) -> Optional[datetime]:
        candidates = [now + timedelta(seconds=spec.completion_window)]
        resource_allocation = spec.aibrix.resource_allocation if spec.aibrix else None
        provision_deadline = (
            resource_allocation.provision_resource_deadline
            if resource_allocation
            else None
        )
        if provision_deadline is not None and provision_deadline > 0:
            candidates.append(
                datetime.fromtimestamp(provision_deadline, tz=timezone.utc)
            )
        return min(candidates) if candidates else None

    def _now(self) -> datetime:
        current = self._now_provider()
        if current.tzinfo is None:
            return current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc)

    @staticmethod
    def _build_hpa_resource_percentage(
        resource: ResourceRequirement,
    ) -> Dict[str, int]:
        metrics = {OctagramManifestRenderer._resolve_hpa_metric(resource): 100}
        if resource.cpu:
            metrics["cpu"] = 100
        if resource.memory:
            metrics["memory"] = 100
        return metrics

    @staticmethod
    def _resolve_hpa_metric(resource: ResourceRequirement) -> str:
        metric = _accelerator_category(resource)
        if metric in {"cpu", "memory", "gpu", "npu", "xpu"}:
            return metric
        return "gpu"

    def _container_env(
        self,
        job_name: str,
        manifest: Dict[str, Any],
        template: ModelDeploymentTemplate,
        cluster_name: str,
        idc_name: str,
        tce_env: str,
        tce_stage: str,
        tce_primary_port: str,
        accelerator_count: int,
    ) -> List[Dict[str, Any]]:
        model_name = _sanitize_model_identity(
            infer_model_name(template.spec.model_source.uri)
        )
        env = [
            {"name": "AIBRIX_LLM_ENGINE", "value": template.spec.engine.type.value},
            {"name": "HDFS_MODEL_PATH", "value": template.spec.model_source.uri},
            {"name": "MODEL_NAME", "value": model_name},
            {
                "name": "AIBRIX_SERVED_MODEL_NAME",
                "value": manifest["metadata"]["labels"]["model.aibrix.ai/name"],
            },
            {
                "name": "EXTRA_PARAMETERS",
                "value": " ".join(build_engine_args(template.spec, True)),
            },
            {"name": "MY_GPU_REQUEST", "value": str(accelerator_count)},
            {"name": "TCE_CLUSTER", "value": cluster_name},
            {"name": "TCE_ENV", "value": tce_env},
            {"name": "TCE_INTERNAL_IDC", "value": idc_name},
            {"name": "TCE_PHYSICAL_CLUSTER", "value": cluster_name},
            {"name": "TCE_PRIMARY_PORT", "value": tce_primary_port},
            {"name": "TCE_STAGE", "value": tce_stage},
            # we also include BERNARD_SERVICE_ID to leverage bernard facilities
            {"name": "BERNARD_SERVICE_ID", "value": job_name},
        ]
        psm = manifest["metadata"]["labels"].get("psm")
        if psm:
            env.extend(
                [
                    {"name": "LOAD_SERVICE_PSM", "value": psm},
                    {"name": "TCE_PSM", "value": psm},
                    {"name": "AIBRIX_PSM", "value": psm},
                    {"name": "BERNARD_ID", "value": psm},
                ]
            )
        return env

    def _container_resources(self, resource: ResourceRequirement) -> Dict[str, Any]:
        accelerator_key = _accelerator_category(resource)
        accelerator_count = str(resource.accelerator_count or 0)
        request_limit = {
            "request": accelerator_count,
            "limit": accelerator_count,
        }
        resources: Dict[str, Any] = {}
        if accelerator_key == "xpu":
            resources["extended"] = [
                {
                    "name": _OCTAGRAM_XPU_RESOURCE_NAME,
                    "requestLimit": request_limit,
                }
            ]
        elif accelerator_key == "npu":
            resources["extended"] = [
                {
                    "name": _OCTAGRAM_NPU_RESOURCE_NAME,
                    "requestLimit": request_limit,
                }
            ]
        else:
            resources[accelerator_key] = request_limit
        if resource.cpu:
            resources["cpu"] = {"request": resource.cpu, "limit": resource.cpu}
        if resource.memory:
            resources["memory"] = {
                "request": resource.memory,
                "limit": resource.memory,
            }
        return resources

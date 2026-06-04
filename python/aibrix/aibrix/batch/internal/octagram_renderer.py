from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple

from aibrix import envs
from aibrix.batch.job_entity import BatchJobSpec, ResourceDetail, ResourceRequirement
from aibrix.batch.template import ModelDeploymentTemplate
from aibrix.downloader.utils import infer_model_name
from aibrix.logger import init_logger

from aibrix.batch.manifest.engine_adapter import build_engine_args
from aibrix.batch.manifest.renderer import _RendererSupport

logger = init_logger(__name__)

_DEFAULT_NAMESPACE = "default"
_DEFAULT_CONTAINER_NAME = "llm-engine"
_DEFAULT_TCE_ENV = "prod"
_DEFAULT_TCE_STAGE = "all_dc"
_DEFAULT_TCE_PRIMARY_PORT = "fake_port"
_DEFAULT_VOLUME_CAPACITY = "10000Gi"
_DEFAULT_IDENTITY_TREATMENT_USER = "jingyuan.zhang0929"

_BASE_FEATURE_GATES = [
    {"name": "ContainerResourceView", "value": "container"},
    {"name": "ContainerShmSize", "value": 100000},
    {"name": "IPv6OnlyCompatible", "value": True},
]

_VOLUME_MOUNTS = [
    {"name": "bernard", "mountPath": "/opt/tiger/bernard", "readOnly": True},
    {"name": "bernard-pre-download", "mountPath": "/opt/tiger/bernard_pre_download"},
    {
        "name": "bernard-tce-tools",
        "mountPath": "/opt/tiger/tce/tce_tools",
        "readOnly": True,
    },
    {"name": "cgroups", "mountPath": "/sys/fs/cgroup", "readOnly": True},
    {"name": "chadc", "mountPath": "/opt/tiger/chadc", "readOnly": True},
    {
        "name": "consul-deploy",
        "mountPath": "/opt/tiger/consul_deploy",
        "readOnly": True,
    },
    {"name": "core", "mountPath": "/opt/tiger/cores", "readOnly": True},
    {"name": "databus", "mountPath": "/tmp"},
    {"name": "databus-new", "mountPath": "/opt/tmp", "readOnly": True},
    {"name": "dist", "mountPath": "/opt/tiger/dist/", "readOnly": True},
    {
        "name": "host-usr-local-tao-agent-modules-bvc",
        "mountPath": "/usr/local/tao/agent/modules/bvc/",
        "readOnly": True,
    },
    {"name": "jdk", "mountPath": "/opt/tiger/jdk", "readOnly": True},
    {
        "name": "linux-gnu",
        "mountPath": "/opt/tiger/x86_64-linux-gnu",
        "readOnly": True,
    },
    {"name": "network", "mountPath": "/opt/tiger/tce/network", "readOnly": True},
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
    {"name": "pyutil", "mountPath": "/opt/tiger/pyutil", "readOnly": True},
    {"name": "run", "mountPath": "/run"},
    {"name": "run-lock", "mountPath": "/run/lock", "readOnly": True},
    {"name": "ss-bin", "mountPath": "/opt/tiger/ss_bin", "readOnly": True},
    {"name": "sys", "mountPath": "/sys", "readOnly": True},
    {
        "name": "sys-resolv-conf",
        "mountPath": "/etc/resolv.conf",
        "readOnly": True,
    },
    {
        "name": "tce-tools-binary",
        "mountPath": "/opt/tiger/tce/tce_tools/bin/binary",
        "readOnly": True,
    },
    {
        "name": "var-log-tiger",
        "mountPath": "/var/log/tiger",
        "subPath": "$(MY_POD_NAME)/var/log/tiger",
    },
    {"name": "yarn-deploy", "mountPath": "/opt/tiger/yarn_deploy", "readOnly": True},
]

_VOLUMES = [
    {"name": "bernard", "hostPath": {"path": "/opt/tiger/bernard", "type": ""}},
    {
        "name": "bernard-pre-download",
        "hostPath": {"path": "/opt/tiger/bernard_pre_download", "type": ""},
    },
    {
        "name": "bernard-tce-tools",
        "hostPath": {"path": "/opt/tiger/bernard/bernard_tools", "type": ""},
    },
    {"name": "cgroups", "hostPath": {"path": "/sys/fs/cgroup", "type": ""}},
    {"name": "chadc", "hostPath": {"path": "/opt/tiger/chadc", "type": ""}},
    {
        "name": "consul-deploy",
        "hostPath": {"path": "/opt/tiger/consul_deploy", "type": ""},
    },
    {
        "name": "core",
        "hostPath": {"path": "/opt/tiger/cores", "type": "DirectoryOrCreate"},
    },
    {"name": "databus", "hostPath": {"path": "/tmp", "type": ""}},
    {"name": "databus-new", "hostPath": {"path": "/opt/tmp", "type": ""}},
    {"name": "dist", "hostPath": {"path": "/opt/tiger/dist/", "type": ""}},
    {
        "name": "host-usr-local-tao-agent-modules-bvc",
        "hostPath": {"path": "/usr/local/tao/agent/modules/bvc/", "type": ""},
    },
    {"name": "jdk", "hostPath": {"path": "/opt/tiger/jdk", "type": ""}},
    {
        "name": "linux-gnu",
        "hostPath": {"path": "/usr/lib/x86_64-linux-gnu", "type": ""},
    },
    {
        "name": "network",
        "hostPath": {"path": "/opt/tiger/tce/network", "type": ""},
    },
    {
        "name": "opt-tiger-data-log",
        "hostPath": {
            "path": "/opt/tiger/tce/containers/inf.aibrix.kvcache",
            "type": "DirectoryOrCreate",
        },
    },
    {
        "name": "opt-tiger-toutiao-log",
        "hostPath": {
            "path": "/opt/tiger/tce/containers/inf.aibrix.kvcache",
            "type": "DirectoryOrCreate",
        },
    },
    {"name": "pyutil", "hostPath": {"path": "/opt/tiger/pyutil", "type": ""}},
    {"name": "run", "emptyDir": {"medium": "Memory", "sizeLimit": "64Mi"}},
    {"name": "run-lock", "hostPath": {"path": "/run/lock", "type": ""}},
    {"name": "ss-bin", "hostPath": {"path": "/opt/tiger/ss_bin", "type": ""}},
    {"name": "sys", "hostPath": {"path": "/sys", "type": ""}},
    {"name": "sys-resolv-conf", "hostPath": {"path": "/etc/resolv.conf", "type": ""}},
    {
        "name": "tce-tools-binary",
        "hostPath": {"path": "/opt/tiger/tce/tce_tools/bin/binary", "type": ""},
    },
    {
        "name": "var-log-tiger",
        "hostPath": {
            "path": "/opt/tiger/tce/containers/inf.aibrix.kvcache",
            "type": "DirectoryOrCreate",
        },
    },
    {"name": "yarn-deploy", "hostPath": {"path": "/opt/tiger/yarn_deploy", "type": ""}},
]


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
        spec: BatchJobSpec,
        prividerSpec: ResourceDetail,
        namespace: str = _DEFAULT_NAMESPACE,
        container_name: str = _DEFAULT_CONTAINER_NAME,
        tce_env: str = _DEFAULT_TCE_ENV,
        tce_stage: str = _DEFAULT_TCE_STAGE,
        tce_primary_port: str = _DEFAULT_TCE_PRIMARY_PORT,
    ) -> Dict[str, Any]:
        template, _ = self._resolve(spec)
        self.template = template

        # Validate the supportable value space up-front so downstream
        # layers can assume they're working with k8s + dedicated + supported endpoint.
        self._validate_template(template, spec.endpoint)

        # Construct dynamic values
        job_name = f"batch-{template.name}-{job_id[:8]}".lower()
        resource = prividerSpec.resource
        model_name = infer_model_name(template.spec.model_source.uri).lower()
        served_model_name = f"{job_name}-{model_name}"
        cluster_name, idc_name = self._parse_endpoint_cluster(
            prividerSpec.endpoint_cluster
        )
        self.cluster_name = cluster_name
        self.idc_name = idc_name

        # Layered composition.
        manifest = self._system_base(
            job_id=job_id,
            job_name=job_name,
            served_model_name=served_model_name,
            namespace=namespace,
        )
        self._apply_template_psm(manifest, template)
        self._apply_feature_gates(manifest, prividerSpec)
        self._apply_deployment_meta(manifest, prividerSpec, resource)
        self._apply_strategy(manifest, resource)
        self._apply_pod_base(
            manifest=manifest,
            template=template,
            detail=prividerSpec,
            resource=resource,
            container_name=container_name,
            cluster_name=cluster_name,
            idc_name=idc_name,
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
        job_name: str,
        served_model_name: str,
        namespace: str,
    ) -> Dict[str, Any]:
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
                "labels": {
                    "name": job_name,
                    "batch.aibrix.ai/job_id": job_id,
                    "model.aibrix.ai/name": served_model_name,
                },
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
            "deployment.tce.kubernetes.io/gpu-type": resource.accelerator_type or "",
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
            },
            "annotations": annotations,
        }
        psm = manifest["metadata"]["labels"].get("psm")
        if psm:
            manifest["spec"]["deploymentMeta"]["labels"]["psm"] = psm

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
                    "batch.aibrix.ai/job_id": manifest["metadata"]["labels"][
                        "batch.aibrix.ai/job_id"
                    ],
                    "model.aibrix.ai/name": manifest["metadata"]["labels"][
                        "model.aibrix.ai/name"
                    ],
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
        psm = manifest["metadata"]["labels"].get("psm")
        if psm:
            manifest["spec"]["deployStrategy"]["selector"]["matchLabels"]["psm"] = psm

    def _apply_pod_base(
        self,
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
                    "livenessProbe": {
                        "exec": {
                            "command": [
                                "bash",
                                "-c",
                                "/opt/tiger/bernard/bernard_tools/bin/liveness_check.sh",
                            ]
                        },
                        "initialDelaySeconds": 180,
                        "failureThreshold": 5,
                        "periodSeconds": 60,
                        "successThreshold": 1,
                        "timeoutSeconds": 30,
                    },
                    "readinessProbe": {
                        "exec": {
                            "command": [
                                "bash",
                                "-c",
                                "/opt/tiger/tce/tce_tools/bin/readiness_check.sh",
                            ]
                        },
                        "failureThreshold": 2,
                        "periodSeconds": 20,
                        "successThreshold": 1,
                        "timeoutSeconds": 30,
                    },
                    "lifecycle": {
                        "preStop": {
                            "exec": {
                                "command": [
                                    "bash",
                                    "-c",
                                    "/opt/tiger/tce/tce_tools/bin/pre_stop; /home/tiger/.op/docker_pre_stop.sh;",
                                ]
                            }
                        }
                    },
                }
            ],
            "nodeSelector": {
                "accelerator": resource.accelerator_type or "",
                "nodeLevel": detail.logical_cluster or "",
            },
            "hostNetwork": True,
            "terminationGracePeriodSeconds": 30,
            "volumes": deepcopy(_VOLUMES),
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

        resource_metric = self._resolve_hpa_metric(resource)
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
                    "resourcePercentage": {resource_metric: 1},
                }
            )

        rules.append(
            {
                "maxReplica": 0,
                "minReplica": 0,
                "resourcePercentage": {resource_metric: 1},
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
    def _resolve_hpa_metric(resource: ResourceRequirement) -> str:
        metric = (resource.accelerator_category or "gpu").lower()
        if metric in {"cpu", "memory", "gpu", "npu", "xpu"}:
            return metric
        return "gpu"

    def _container_env(
        self,
        manifest: Dict[str, Any],
        template: ModelDeploymentTemplate,
        cluster_name: str,
        idc_name: str,
        tce_env: str,
        tce_stage: str,
        tce_primary_port: str,
        accelerator_count: int,
    ) -> List[Dict[str, Any]]:
        model_name = infer_model_name(template.spec.model_source.uri)  # must be same
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
            {"name": "TP_SIZE", "value": str(template.spec.parallelism.tp)},
            {"name": "PP_SIZE", "value": str(template.spec.parallelism.pp)},
            {"name": "DP_SIZE", "value": str(template.spec.parallelism.dp)},
            {"name": "MY_GPU_REQUEST", "value": str(accelerator_count)},
            {"name": "TCE_CLUSTER", "value": cluster_name},
            {"name": "TCE_ENV", "value": tce_env},
            {"name": "TCE_INTERNAL_IDC", "value": idc_name},
            {"name": "TCE_PHYSICAL_CLUSTER", "value": cluster_name},
            {"name": "TCE_PRIMARY_PORT", "value": tce_primary_port},
            {"name": "TCE_STAGE", "value": tce_stage},
        ]
        psm = manifest["metadata"]["labels"].get("psm")
        if psm:
            env.extend(
                [
                    {"name": "LOAD_SERVICE_PSM", "value": psm},
                    {"name": "TCE_PSM", "value": psm},
                ]
            )
        return env

    def _container_resources(
        self, resource: ResourceRequirement
    ) -> Dict[str, Dict[str, str]]:
        accelerator_key = (resource.accelerator_category or "gpu").lower()
        accelerator_count = str(resource.accelerator_count or 0)
        resources = {
            accelerator_key: {
                "request": accelerator_count,
                "limit": accelerator_count,
            },
        }
        if resource.cpu:
            resources["cpu"] = {"request": resource.cpu, "limit": resource.cpu}
        if resource.memory:
            resources["memory"] = {
                "request": resource.memory,
                "limit": resource.memory,
            }
        return resources

    def _parse_endpoint_cluster(
        self, endpoint_cluster: Optional[str]
    ) -> Tuple[str, str]:
        if not endpoint_cluster:
            return "", ""
        # Lower-case before splitting: consul cluster lookup is case-sensitive
        # and the upstream endpoint_cluster may arrive mixed-case (e.g. Echo-HL).
        parts = endpoint_cluster.lower().rsplit("-", 1)
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]

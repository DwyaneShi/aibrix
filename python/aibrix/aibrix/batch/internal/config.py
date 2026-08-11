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

import re
from dataclasses import dataclass

AUTHORIZATION_HEADER = "Authorization"


@dataclass
class RegionDomain:
    argos: str
    grafana: str
    octagram: str
    tce_status: str


REGION_DOMAINS = {
    "CN": RegionDomain(
        argos="https://cloud.bytedance.net/argos",
        grafana="https://grafana.byted.org",
        octagram="https://octagram-gateway.byted.org",
        tce_status="http://tce-status.byted.org",
    ),
    "US": RegionDomain(
        argos="https://cloud.tiktok-row.net/argos",
        grafana="https://grafana-i18n.byted.org",
        octagram="https://octagram-gateway-us.byted.org",
        tce_status="http://tce-status-us.byted.org",
    ),
}


# ---------------------------------------------------------------------------
# GPU resource specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuSpec:
    """Per-GPU resource specification for a GPU card type."""

    cpu_per_gpu: int
    mem_per_gpu: int


_GPU_TYPE_PREFIXES = ("NVIDIA-GeForce-", "NVIDIA-", "Tesla-")
_GPU_INTERCONNECT_RE = re.compile(r"-(?:PCIE|SXM\d*)-", re.IGNORECASE)
_GPU_HBM_RE = re.compile(r"-\d+GB$", re.IGNORECASE)
_GPU_TYPE_ALIASES: dict[str, str] = {}

# Per-GPU resource specs
_GPU_SPECS: dict[str, GpuSpec] = {
    "A100": GpuSpec(cpu_per_gpu=15750, mem_per_gpu=251),
    "RTX-6000D": GpuSpec(cpu_per_gpu=14000, mem_per_gpu=88),
}

_DEFAULT_CPU_MILLICORES_PER_GPU = 16000
_DEFAULT_MEM_PER_GPU = 96


def normalize_gpu_type(gpu_type: str) -> str:
    """Normalize a gpu_type string to its canonical short form.

    1. Strip vendor prefixes (NVIDIA-GeForce-, NVIDIA-, Tesla-).
    2. Remove interconnect suffixes (-PCIE-, -SXM-, -SXM2-, etc.).
    3. Remove HBM capacity suffixes (-40GB, -80GB, etc.).
    4. Resolve known aliases.
    """
    stripped = gpu_type
    for prefix in _GPU_TYPE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break
    stripped = _GPU_INTERCONNECT_RE.sub("-", stripped)
    stripped = _GPU_HBM_RE.sub("", stripped)
    return _GPU_TYPE_ALIASES.get(stripped, stripped)


def get_gpu_spec(gpu_type: str) -> GpuSpec:
    """Look up per-GPU resource spec by gpu_type.

    Returns a default spec if the gpu_type is unknown.
    """
    canonical = normalize_gpu_type(gpu_type)
    return _GPU_SPECS.get(
        canonical,
        GpuSpec(
            cpu_per_gpu=_DEFAULT_CPU_MILLICORES_PER_GPU,
            mem_per_gpu=_DEFAULT_MEM_PER_GPU,
        ),
    )

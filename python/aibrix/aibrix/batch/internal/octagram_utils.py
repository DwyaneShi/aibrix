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

"""Shared helpers for Octagram/TCE internals."""

from typing import Optional, Tuple

from aibrix import envs
from aibrix.batch.job_entity import BatchJob


def get_job_name(job: BatchJob) -> str:
    """Derive the Octagram workload job_name from a BatchJob.

    ``batch-{template_name}-{job_id[:8]}`` when a template is referenced,
    otherwise ``batch-{job_id[:8]}``. Always lowercased.
    """
    job_id = job.job_id or ""
    aibrix = job.spec.aibrix if job.spec else None
    template_name = aibrix.model_template_name if aibrix else None

    if template_name:
        return f"batch-{template_name}-{job_id[:8]}".lower()
    return f"batch-{job_id[:8]}".lower()


def get_psm(job: BatchJob) -> Optional[str]:
    """Derive the Octagram PSM from a BatchJob.

    ``template.spec.service_id`` when available, falling back to
    ``envs.CONSUL_BATCH_DISCOVERY_PSM``.
    """
    aibrix = job.spec.aibrix if job.spec else None
    if aibrix and aibrix.model_template and aibrix.model_template.spec:
        psm = aibrix.model_template.spec.get("service_id")
        if psm:
            return psm
    return getattr(envs, "CONSUL_BATCH_DISCOVERY_PSM", None) or None


def parse_endpoint_cluster(
    endpoint_cluster: Optional[str],
) -> Tuple[str, str, str, str]:
    """Parse endpoint cluster string into (zone, idc, physical_cluster,
    logical_cluster) tuple."""
    if not endpoint_cluster:
        return "", "", "", ""
    parts = endpoint_cluster.rsplit("/", 4)
    return parts[0], parts[1], parts[2], parts[3]

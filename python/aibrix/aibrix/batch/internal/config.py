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
        argos="https://cloud-tx.tiktokd.net/argos",
        grafana="https://grafana-us.byted.org",
        octagram="https://octagram-gateway-us.byted.org",
        tce_status="http://tce-status-us.byted.org",
    ),
}

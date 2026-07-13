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

"""Internal ByteTOS PSM environment helpers.

Minimal env combinations by region:

- Overseas internal TOS, e.g. REGION=US: use PSM mode.
  Required:
  - STORAGE_TYPE
  - STORAGE_TOS_ACCESS_KEY
  - STORAGE_TOS_SECRET_KEY
  - STORAGE_TOS_BUCKET
  - STORAGE_TOS_IDC
  Optional:
  - STORAGE_TOS_SERVICE
  - STORAGE_TOS_CLUSTER
  - STORAGE_TOS_REMOTE_PSM
  Do not require STORAGE_TOS_ENDPOINT or STORAGE_TOS_REGION; endpoint mode can
  fail bytedtos endpoint checks before internal PSM routing is used.

- CN internal TOS: endpoint mode is sufficient.
  Required:
  - STORAGE_TYPE
  - STORAGE_TOS_ACCESS_KEY
  - STORAGE_TOS_SECRET_KEY
  - STORAGE_TOS_BUCKET
  - STORAGE_TOS_ENDPOINT
  - STORAGE_TOS_REGION
  Optional:
  - STORAGE_TOS_ENABLE_CRC
  Leave STORAGE_TOS_IDC unset unless the deployment explicitly wants PSM mode.
"""

import os
from typing import Optional, TypedDict


class TOSPSMEnv(TypedDict):
    idc: Optional[str]
    service: str
    cluster: str
    remote_psm: str


def tos_psm_env() -> TOSPSMEnv:
    return {
        "idc": os.getenv("STORAGE_TOS_IDC"),
        "service": os.getenv("STORAGE_TOS_SERVICE", "toutiao.tos.tosapi"),
        "cluster": os.getenv("STORAGE_TOS_CLUSTER", "default"),
        "remote_psm": os.getenv("STORAGE_TOS_REMOTE_PSM", "inf.aibrix.metadata"),
    }


def tos_psm_env_vars() -> dict[str, str]:
    values = tos_psm_env()
    env_vars = {
        "STORAGE_TOS_IDC": values["idc"],
        "STORAGE_TOS_SERVICE": values["service"],
        "STORAGE_TOS_CLUSTER": values["cluster"],
        "STORAGE_TOS_REMOTE_PSM": values["remote_psm"],
    }
    return {name: value for name, value in env_vars.items() if value}

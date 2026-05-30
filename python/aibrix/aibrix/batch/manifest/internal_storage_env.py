# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internal BytedRedis/Consul worker-env injection for the batch metastore.

``storage_env._redis_env()`` calls ``redis_env()`` here when this module is
present; in OSS builds it is absent and the public REDIS_HOST path is used.
"""

import os
from typing import Any, Dict, List

from aibrix import envs


def redis_env() -> List[Dict[str, Any]]:
    """Worker env vars for a BytedRedis-backed metastore (PSM-based discovery)."""
    env: List[Dict[str, Any]] = [
        {"name": "BYTEDREDIS_PSM", "value": envs.STORAGE_REDIS_PSM},
        {"name": "REDIS_DB", "value": str(envs.STORAGE_REDIS_DB)},
    ]
    if envs.STORAGE_REDIS_DISABLE_METRICS:
        env.append(
            {
                "name": "BYTEDREDIS_DISABLE_METRICS",
                "value": "1" if envs.STORAGE_REDIS_DISABLE_METRICS else "0",
            }
        )
    if envs.STORAGE_REDIS_DISABLE_AUTH:
        env.append(
            {
                "name": "BYTEDREDIS_DISABLE_AUTH",
                "value": "1" if envs.STORAGE_REDIS_DISABLE_AUTH else "0",
            }
        )
    env.extend(
        [
            {
                "name": "BYTEDREDIS_SOCKET_TIMEOUT",
                "value": str(envs.STORAGE_REDIS_SOCKET_TIMEOUT),
            },
            {
                "name": "BYTEDREDIS_SOCKET_CONNECT_TIMEOUT",
                "value": str(envs.STORAGE_REDIS_SOCKET_CONNECT_TIMEOUT),
            },
            {"name": "CONSUL_HTTP_HOST", "value": os.getenv("CONSUL_HTTP_HOST", "")},
            {"name": "CONSUL_HTTP_PORT", "value": os.getenv("CONSUL_HTTP_PORT", "")},
            {"name": "BYTED_HOST_IPV6", "value": os.getenv("BYTED_HOST_IPV6", "")},
            {"name": "MY_HOST_IPV6", "value": os.getenv("MY_HOST_IPV6", "")},
            {"name": "SEC_KV_AUTH", "value": os.getenv("SEC_KV_AUTH", "")},
        ]
    )
    return env

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

"""Internal-only environment variables (ByteDance infra: PSM / BytedRedis / Consul / Octagram).

Kept out of the upstreamable ``envs.py`` so that file carries no internal concepts.
``envs.py`` re-exports everything here via a single ``from aibrix.internal_envs import *``,
so existing ``envs.STORAGE_REDIS_PSM``-style references keep working unchanged.
"""

import logging
import os
from typing import Callable, List

logger = logging.getLogger(__name__)

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def _is_true(value) -> bool:
    return str(value).upper() in ENV_VARS_TRUE_VALUES


REGION = os.getenv("REGION", "CN")

# BytedRedis (PSM-based) storage / metadata. Overrides the public REDIS_HOST path.
STORAGE_REDIS_DB = int(os.environ.get("BYTEDREDIS_DB", os.environ.get("REDIS_DB", "0")))
STORAGE_REDIS_PSM = os.getenv("BYTEDREDIS_PSM")
STORAGE_REDIS_AVAILABLE = STORAGE_REDIS_PSM is not None
STORAGE_REDIS_DISABLE_AUTH = _is_true(os.getenv("BYTEDREDIS_DISABLE_AUTH", "0"))
STORAGE_REDIS_DISABLE_METRICS = _is_true(os.getenv("BYTEDREDIS_DISABLE_METRICS", "0"))
STORAGE_REDIS_SOCKET_TIMEOUT = float(os.getenv("BYTEDREDIS_SOCKET_TIMEOUT", 0.05))
STORAGE_REDIS_SOCKET_CONNECT_TIMEOUT = float(
    os.getenv("BYTEDREDIS_SOCKET_CONNECT_TIMEOUT", 0.05)
)

# Octagram gateway / Consul service discovery
OCTAGRAM_GATEWAY_DOMAIN = os.getenv("OCTAGRAM_GATEWAY_DOMAIN")
OCTAGRAM_WORKLOAD_NOT_FOUND_GRACE_SECONDS = int(
    os.getenv("AIBRIX_OCTAGRAM_WORKLOAD_NOT_FOUND_GRACE_SECONDS", "600")
)
CONSUL_HTTP_HOST = os.getenv("CONSUL_HTTP_HOST", "127.0.0.1")
CONSUL_HTTP_PORT = os.getenv("CONSUL_HTTP_PORT", "2280")
CONSUL_BATCH_DISCOVERY_TIMEOUT = int(os.getenv("AIBRIX_TASK_EXECUTOR_TIMEOUT", "900"))
CONSUL_BATCH_DISCOVERY_PSM = os.getenv(
    "AIBRIX_TASK_EXECUTOR_PSM", "inf.aibrix.inference_workers"
)

# Metadata service envs
METADATA_MAX_FILE_SIZE = int(
    os.getenv("AIBRIX_METADATA_MAX_FILE_SIZE", str(1024 * 1024 * 1024))
)


class EnvRestorer:
    """A callable that runs an initial function plus any appended ones in sequence."""

    def __init__(self, func: Callable[[], None]):
        self._callables: List[Callable[[], None]] = [func]

    def __call__(self):
        for f in self._callables:
            f()

    def append(self, func: Callable[[], None]) -> "EnvRestorer":
        self._callables.append(func)
        return self


def setdefault(key: str, value: str) -> Callable[[], None]:
    """Set a default for an env key if unset; return a restore function."""
    original_value = os.environ.get(key)
    if original_value is None or original_value == "":
        os.environ[key] = value
        logger.info(f"Set default value {value} for env {key}")

        def restore() -> None:
            os.environ.pop(key, None)

        return restore
    logger.debug(f"Env {key} already set to {original_value}, skip default {value}")
    return lambda: None

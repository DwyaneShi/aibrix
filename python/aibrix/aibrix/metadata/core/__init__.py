# Copyright 2024 The Aibrix Team.
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

from typing import TYPE_CHECKING

from .asyncio_thread import AsyncLoopThread, T
from .httpx_client import HTTPXClientWrapper

__all__ = [
    "AsyncLoopThread",
    "ConsulDiscoveryService",
    "ConsulInferenceEndpoint",
    "ConsulInferenceSnapshot",
    "HTTPXClientWrapper",
    "T",
]

_LAZY_CONSUL = (
    "ConsulDiscoveryService",
    "ConsulInferenceEndpoint",
    "ConsulInferenceSnapshot",
)

if TYPE_CHECKING:
    from .consul_discovery import (  # noqa: F401
        ConsulDiscoveryService,
        ConsulInferenceEndpoint,
        ConsulInferenceSnapshot,
    )


def __getattr__(name: str):
    """Lazy-load optional Consul discovery so OSS imports do not require SDKs."""
    if name in _LAZY_CONSUL:
        from . import consul_discovery

        return getattr(consul_discovery, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

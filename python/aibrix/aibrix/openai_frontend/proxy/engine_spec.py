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

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineSpec:
    name: str
    chat_endpoint: str = "/v1/chat/completions"
    completion_endpoint: str = "/v1/completions"
    embedding_endpoint: str = "/v1/embeddings"
    models_endpoint: str = "/v1/models"
    health_endpoint: str = "/health"
    metrics_endpoint: str = "/metrics"
    supports_chat: bool = True
    supports_completion: bool = True
    supports_embedding: bool = True
    supports_models: bool = True
    supports_metrics: bool = True
    host_arg: str = "--host"
    port_arg: str = "--port"

    def require_chat(self) -> None:
        if not self.supports_chat:
            raise NotImplementedError(
                f"Engine '{self.name}' does not support chat completions"
            )

    def require_completion(self) -> None:
        if not self.supports_completion:
            raise NotImplementedError(
                f"Engine '{self.name}' does not support completions"
            )

    def require_embedding(self) -> None:
        if not self.supports_embedding:
            raise NotImplementedError(
                f"Engine '{self.name}' does not support embeddings"
            )

    def require_models(self) -> None:
        if not self.supports_models:
            raise NotImplementedError(
                f"Engine '{self.name}' does not support listing models"
            )

    def require_metrics(self) -> None:
        if not self.supports_metrics:
            raise NotImplementedError(f"Engine '{self.name}' does not support metrics")


VLLM_SPEC = EngineSpec(name="vllm")
SGLANG_SPEC = EngineSpec(name="sglang")
# Embedding support is only supported if launch with `trtllm-serve embeddings`
TRTLLM_SPEC = EngineSpec(
    name="trtllm",
    supports_embedding=False,
)

ENGINE_SPECS: dict[str, EngineSpec] = {
    "vllm": VLLM_SPEC,
    "sglang": SGLANG_SPEC,
    "trtllm": TRTLLM_SPEC,
}


def get_engine_spec(name: str) -> EngineSpec:
    if name not in ENGINE_SPECS:
        raise ValueError(
            f"Unknown engine spec '{name}'. "
            f"Available: {', '.join(sorted(ENGINE_SPECS))}"
        )
    return ENGINE_SPECS[name]

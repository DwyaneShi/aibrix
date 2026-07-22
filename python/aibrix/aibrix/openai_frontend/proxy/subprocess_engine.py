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

import asyncio
import logging
import os
import signal
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

import httpx

from aibrix.openai_frontend.proxy.engine_spec import EngineSpec
from aibrix.openai_frontend.utils.utils import prefix_line

logger = logging.getLogger(__name__)


class SubprocessEngine:
    def __init__(
        self,
        cmd: List[str],
        port: int = 8001,
        host: str = "127.0.0.1",
        env: Optional[Dict[str, str]] = None,
        spec: Optional[EngineSpec] = None,
    ):
        self._cmd = cmd
        self._port = port
        self._host = host
        self._env = env
        self._spec = spec or EngineSpec(name="default")

        self._proc: Optional[subprocess.Popen] = None
        self._base_url = f"http://{host}:{port}"
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    def spawn(self) -> int:
        env = {
            **os.environ,
            **(self._env or {}),
            "TQDM_MININTERVAL": "5",
            "PYTHONUNBUFFERED": "1",
        }
        full_cmd = [
            *self._cmd,
            self._spec.host_arg,
            self._host,
            self._spec.port_arg,
            str(self._port),
        ]
        self._proc = subprocess.Popen(
            full_cmd,
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stdout_thread = threading.Thread(
            target=self._pipe_with_prefix,
            args=(self._proc.stdout, sys.__stdout__, "[proxy-engine]"),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._pipe_with_prefix,
            args=(self._proc.stderr, sys.__stderr__, "[proxy-engine]"),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        logger.info(
            "Spawned inner engine: cmd=%s, pid=%d, port=%d",
            " ".join(self._cmd),
            self._proc.pid,
            self._port,
        )
        return self._proc.pid

    @staticmethod
    def _pipe_with_prefix(pipe, target, prefix: str) -> None:
        try:
            for line in pipe:
                target.write(prefix_line(line, prefix))
                target.flush()
        except ValueError:
            pass

    async def wait_ready(self) -> None:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as probe:
            while True:
                try:
                    resp = await probe.get(
                        f"{self._base_url}{self._spec.health_endpoint}"
                    )
                    if resp.status_code == 200:
                        logger.info("Inner engine is ready")
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(1.0)

    async def start_client(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=None,
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0,
            ),
        )

    async def forward_chat(
        self,
        payload: Dict[str, Any],
    ) -> httpx.Response:
        self._spec.require_chat()
        assert self._client is not None
        resp = await self._client.post(self._spec.chat_endpoint, json=payload)
        resp.raise_for_status()
        return resp

    async def forward_completion(
        self,
        payload: Dict[str, Any],
    ) -> httpx.Response:
        self._spec.require_completion()
        assert self._client is not None
        resp = await self._client.post(self._spec.completion_endpoint, json=payload)
        resp.raise_for_status()
        return resp

    async def forward_embedding(
        self,
        payload: Dict[str, Any],
    ) -> httpx.Response:
        self._spec.require_embedding()
        assert self._client is not None
        resp = await self._client.post(self._spec.embedding_endpoint, json=payload)
        resp.raise_for_status()
        return resp

    async def forward_health(self) -> bool:
        assert self._client is not None
        try:
            resp = await self._client.get(self._spec.health_endpoint)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def forward_metrics(self) -> str:
        self._spec.require_metrics()
        assert self._client is not None
        try:
            resp = await self._client.get(self._spec.metrics_endpoint)
            resp.raise_for_status()
            return resp.text
        except Exception:
            return ""

    async def forward_models(self) -> httpx.Response:
        self._spec.require_models()
        assert self._client is not None
        resp = await self._client.get(self._spec.models_endpoint)
        resp.raise_for_status()
        return resp

    def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            os.killpg(self._proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except OSError:
                pass
        logger.info("Inner engine process terminated (pid=%d)", self._proc.pid)
        self._proc = None

    async def close_client(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

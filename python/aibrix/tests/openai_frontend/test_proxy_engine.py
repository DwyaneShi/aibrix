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
import json
import logging
import os
import sys
from typing import Optional

import pytest

from aibrix.openai_frontend.common.artifact_fetcher import ArtifactFetcher, FetchResult
from aibrix.openai_frontend.proxy.engine_spec import EngineSpec
from aibrix.openai_frontend.proxy.request_inspector import (
    ChatCompletionInspector,
    CompletionInspector,
    EmbeddingInspector,
)
from aibrix.openai_frontend.proxy.request_rewriter import RequestRewriter
from aibrix.openai_frontend.utils.utils import make_prefix_formatter, prefix_line

# ---------------------------------------------------------------------------
# prefix_line
# ---------------------------------------------------------------------------


class TestPrefixLine:
    def test_plain_line(self):
        assert prefix_line("hello world\n", "[worker-0]") == "[worker-0] hello world\n"

    def test_line_without_newline(self):
        assert prefix_line("hello", "[main]") == "[main] hello"

    def test_empty_string_unchanged(self):
        assert prefix_line("", "[p]") == ""

    def test_whitespace_only_unchanged(self):
        assert prefix_line("   \n", "[p]") == "   \n"

    def test_carriage_return_only(self):
        assert prefix_line("\rprogress", "[p]") == "\r[p] progress"

    def test_trailing_carriage_return(self):
        assert prefix_line("status\r", "[p]") == "\r[p] status"


# ---------------------------------------------------------------------------
# make_prefix_formatter
# ---------------------------------------------------------------------------


class TestMakePrefixFormatter:
    def test_formatter_includes_prefix(self):
        fmt = make_prefix_formatter("[proxy]")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        assert output.startswith("[proxy] ")
        assert "hello" in output

    def test_formatter_preserves_levelname(self):
        fmt = make_prefix_formatter("[proxy]")
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="warn msg",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        assert "WARNING" in output


# ---------------------------------------------------------------------------
# ChatCompletionInspector
# ---------------------------------------------------------------------------


class TestChatCompletionInspector:
    def setup_method(self):
        self.inspector = ChatCompletionInspector()

    def test_single_image_url(self):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/img.png"},
                        },
                    ],
                }
            ]
        }
        locations = self.inspector.extract_urls(request)
        assert len(locations) == 1
        assert locations[0].url == "https://cdn.example.com/img.png"
        assert locations[0].path == ["messages", 0, "content", 1, "image_url", "url"]

    def test_multiple_image_urls_across_messages(self):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/a.jpg"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "And this one"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "s3://bucket/key"},
                        },
                    ],
                },
            ]
        }
        locations = self.inspector.extract_urls(request)
        assert len(locations) == 2
        assert locations[0].url == "https://cdn.example.com/a.jpg"
        assert locations[0].path == ["messages", 0, "content", 0, "image_url", "url"]
        assert locations[1].url == "s3://bucket/key"
        assert locations[1].path == ["messages", 1, "content", 1, "image_url", "url"]

    def test_local_file_url_ignored(self):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "file:///tmp/local.png"},
                        },
                    ],
                }
            ]
        }
        locations = self.inspector.extract_urls(request)
        assert len(locations) == 0

    def test_data_uri_ignored(self):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc123"},
                        },
                    ],
                }
            ]
        }
        locations = self.inspector.extract_urls(request)
        assert len(locations) == 0

    def test_string_content_skipped(self):
        request = {
            "messages": [
                {"role": "user", "content": "Just a text message"},
            ]
        }
        locations = self.inspector.extract_urls(request)
        assert len(locations) == 0

    def test_no_messages(self):
        locations = self.inspector.extract_urls({})
        assert len(locations) == 0

    def test_tos_url_detected(self):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "tos://bucket/path/img.png"},
                        },
                    ],
                }
            ]
        }
        locations = self.inspector.extract_urls(request)
        assert len(locations) == 1
        assert locations[0].url == "tos://bucket/path/img.png"


class TestCompletionInspector:
    def test_returns_empty(self):
        inspector = CompletionInspector()
        locations = inspector.extract_urls({"prompt": "hello"})
        assert locations == []


class TestEmbeddingInspector:
    def test_returns_empty(self):
        inspector = EmbeddingInspector()
        locations = inspector.extract_urls({"input": "hello"})
        assert locations == []


# ---------------------------------------------------------------------------
# RequestRewriter
# ---------------------------------------------------------------------------


class _MockFetcher(ArtifactFetcher):
    """Mock fetcher that maps URLs to local paths without real downloads."""

    def __init__(self, url_to_path: dict[str, str]):
        self._url_to_path = url_to_path
        self.cleaned_up: list[list[FetchResult]] = []

    async def fetch(self, url: str, credentials: dict | None = None) -> FetchResult:
        local = self._url_to_path.get(url, f"file:///tmp/mock_{url.split('/')[-1]}")
        return FetchResult(original_url=url, local_path=local, temp_dir=None)

    async def fetch_many(
        self,
        urls: list[str],
        credentials_map: dict[str, dict] | None = None,
    ) -> list[FetchResult]:
        return [await self.fetch(u) for u in urls]

    async def cleanup(self, results: list[FetchResult]) -> None:
        self.cleaned_up.append(results)


class TestRequestRewriter:
    @pytest.mark.asyncio
    async def test_rewrites_single_image_url(self):
        fetcher = _MockFetcher(
            {"https://cdn.example.com/img.png": "file:///tmp/downloaded_img.png"}
        )
        inspector = ChatCompletionInspector()
        rewriter = RequestRewriter(inspector, fetcher)

        request = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/img.png"},
                        },
                    ],
                }
            ],
        }

        rewritten, results = await rewriter.rewrite(request)
        url_in_rewritten = rewritten["messages"][0]["content"][1]["image_url"]["url"]
        assert url_in_rewritten == "file:///tmp/downloaded_img.png"
        assert len(results) == 1
        assert results[0].original_url == "https://cdn.example.com/img.png"

    @pytest.mark.asyncio
    async def test_rewrites_multiple_urls(self):
        fetcher = _MockFetcher(
            {
                "https://cdn.example.com/a.jpg": "file:///tmp/a.jpg",
                "s3://bucket/key": "file:///tmp/s3_key",
            }
        )
        inspector = ChatCompletionInspector()
        rewriter = RequestRewriter(inspector, fetcher)

        request = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/a.jpg"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "s3://bucket/key"},
                        },
                    ],
                },
            ],
        }

        rewritten, results = await rewriter.rewrite(request)
        assert (
            rewritten["messages"][0]["content"][0]["image_url"]["url"]
            == "file:///tmp/a.jpg"
        )
        assert (
            rewritten["messages"][1]["content"][0]["image_url"]["url"]
            == "file:///tmp/s3_key"
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_no_urls_returns_original(self):
        fetcher = _MockFetcher({})
        inspector = ChatCompletionInspector()
        rewriter = RequestRewriter(inspector, fetcher)

        request = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Just text"},
            ],
        }

        rewritten, results = await rewriter.rewrite(request)
        assert rewritten is request  # same object, not deep-copied
        assert results == []

    @pytest.mark.asyncio
    async def test_does_not_mutate_original(self):
        fetcher = _MockFetcher(
            {"https://cdn.example.com/img.png": "file:///tmp/local.png"}
        )
        inspector = ChatCompletionInspector()
        rewriter = RequestRewriter(inspector, fetcher)

        request = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/img.png"},
                        },
                    ],
                }
            ],
        }
        original_url = request["messages"][0]["content"][0]["image_url"]["url"]

        rewritten, _ = await rewriter.rewrite(request)
        # Original must be unchanged
        assert request["messages"][0]["content"][0]["image_url"]["url"] == original_url
        assert (
            rewritten["messages"][0]["content"][0]["image_url"]["url"]
            == "file:///tmp/local.png"
        )

    @pytest.mark.asyncio
    async def test_duplicate_url_fetched_once(self):
        fetcher = _MockFetcher(
            {"https://cdn.example.com/img.png": "file:///tmp/local.png"}
        )
        inspector = ChatCompletionInspector()
        rewriter = RequestRewriter(inspector, fetcher)

        request = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/img.png"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/img.png"},
                        },
                    ],
                },
            ],
        }

        rewritten, results = await rewriter.rewrite(request)
        assert (
            rewritten["messages"][0]["content"][0]["image_url"]["url"]
            == "file:///tmp/local.png"
        )
        assert (
            rewritten["messages"][1]["content"][0]["image_url"]["url"]
            == "file:///tmp/local.png"
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# EngineSpec
# ---------------------------------------------------------------------------


class TestEngineSpec:
    def test_defaults(self):
        spec = EngineSpec(name="test")
        assert spec.chat_endpoint == "/v1/chat/completions"
        assert spec.health_endpoint == "/health"
        assert spec.host_arg == "--host"
        assert spec.port_arg == "--port"

    def test_custom_endpoints(self):
        spec = EngineSpec(
            name="custom",
            chat_endpoint="/api/chat",
            health_endpoint="/ready",
        )
        assert spec.chat_endpoint == "/api/chat"
        assert spec.health_endpoint == "/ready"

    def test_require_chat_raises(self):
        spec = EngineSpec(name="no-chat", supports_chat=False)
        with pytest.raises(NotImplementedError):
            spec.require_chat()

    def test_require_embedding_raises(self):
        spec = EngineSpec(name="no-emb", supports_embedding=False)
        with pytest.raises(NotImplementedError):
            spec.require_embedding()

    def test_get_engine_spec(self):
        from aibrix.openai_frontend.proxy.engine_spec import get_engine_spec

        vllm = get_engine_spec("vllm")
        assert vllm.name == "vllm"
        with pytest.raises(ValueError):
            get_engine_spec("nonexistent")


class TestBuildProxyEngine:
    def test_builds_from_args(self):
        import argparse

        from aibrix.openai_frontend.main import build_proxy_engine

        args = argparse.Namespace(
            proxy_engine_cmd=["python", "-m", "vllm.entrypoints.openai.api_server"],
            proxy_engine_host="127.0.0.1",
            proxy_engine_port=8001,
            proxy_startup_timeout=60.0,
            proxy_engine_spec="vllm",
            proxy_credentials='{"https://cdn.example.com": {"headers": {"Authorization": "Bearer tok"}}}',
        )
        engine = build_proxy_engine(args)
        from aibrix.openai_frontend.engine.proxy_engine import ProxyEngine

        assert isinstance(engine, ProxyEngine)
        assert engine._subprocess._port == 8001
        assert engine._subprocess._host == "127.0.0.1"
        assert engine._subprocess._cmd == [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
        ]


# ---------------------------------------------------------------------------
# E2E helpers
# ---------------------------------------------------------------------------


class _FileMockFetcher(ArtifactFetcher):
    """Fetcher that maps remote URLs to file:// local paths."""

    def __init__(self, url_to_file: dict[str, str]):
        self._url_to_file = url_to_file

    async def fetch(self, url: str, credentials: Optional[dict] = None) -> FetchResult:
        local = self._url_to_file.get(url, f"file:///tmp/mock_{hash(url)}")
        return FetchResult(original_url=url, local_path=local, temp_dir=None)

    async def fetch_many(
        self,
        urls: list[str],
        credentials_map: dict[str, dict] | None = None,
    ) -> list[FetchResult]:
        return [await self.fetch(u) for u in urls]

    async def cleanup(self, results: list[FetchResult]) -> None:
        pass


def _mock_engine_cmd(port: int) -> list[str]:
    mock_script = os.path.join(os.path.dirname(__file__), "mock_engine.py")
    return [sys.executable, mock_script, "--port", str(port)]


# ---------------------------------------------------------------------------
# E2E tests with real subprocess spawn
# ---------------------------------------------------------------------------


class TestProxyEngineE2E:
    def test_spawn_chat_streaming(self):
        """ProxyEngine spawns inner engine, streaming chat returns StreamingResponse."""

        async def _async():
            inner_port = 18765
            remote_url = "https://cdn.example.com/images/dog.jpg"
            file_url = "file:///data/images/dog.jpg"
            fetcher = _FileMockFetcher({remote_url: file_url})

            from aibrix.openai_frontend.engine.proxy_engine import ProxyEngine

            spec = EngineSpec(name="test")
            engine = ProxyEngine(
                engine_cmd=_mock_engine_cmd(inner_port),
                engine_port=inner_port,
                engine_host="127.0.0.1",
                spec=spec,
                artifact_fetcher=fetcher,
                credentials_map={},
            )

            await engine.start()

            from fastapi.responses import StreamingResponse

            from aibrix.openai_frontend.schemas.openai import (
                CreateChatCompletionRequest,
            )

            request = CreateChatCompletionRequest(
                model="test-model",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": remote_url},
                            },
                        ],
                    }
                ],
                stream=True,
            )

            try:
                response = await engine.chat(request)
                assert isinstance(response, StreamingResponse), (
                    f"Expected StreamingResponse, got {type(response)}"
                )
            finally:
                engine.stop()

        asyncio.run(_async())

    def test_spawn_multiple_image_urls(self):
        """Multiple remote image_urls are all rewritten to file:// in spawned engine."""

        async def _async():
            inner_port = 18766
            fetcher = _FileMockFetcher(
                {
                    "https://cdn.example.com/a.png": "file:///data/a.png",
                    "s3://bucket/key/b.jpg": "file:///data/b.jpg",
                }
            )

            from aibrix.openai_frontend.engine.proxy_engine import ProxyEngine

            spec = EngineSpec(name="test")
            engine = ProxyEngine(
                engine_cmd=_mock_engine_cmd(inner_port),
                engine_port=inner_port,
                engine_host="127.0.0.1",
                spec=spec,
                artifact_fetcher=fetcher,
                credentials_map={},
            )

            await engine.start()

            from aibrix.openai_frontend.schemas.openai import (
                CreateChatCompletionRequest,
            )

            request = CreateChatCompletionRequest(
                model="test-model",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://cdn.example.com/a.png"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "And this one"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "s3://bucket/key/b.jpg"},
                            },
                        ],
                    },
                ],
                stream=False,
            )

            try:
                response = await engine.chat(request)
                assert response.status_code == 200
            finally:
                engine.stop()

        asyncio.run(_async())

    def test_spawn_models_forwarded(self):

        async def _async():
            inner_port = 18768

            from fastapi import Response

            from aibrix.openai_frontend.engine.proxy_engine import ProxyEngine

            spec = EngineSpec(name="test")
            engine = ProxyEngine(
                engine_cmd=_mock_engine_cmd(inner_port),
                engine_port=inner_port,
                engine_host="127.0.0.1",
                spec=spec,
            )

            await engine.start()
            try:
                response = await engine.models()
                assert isinstance(response, Response)
                assert response.status_code == 200
                data = json.loads(response.body)
                assert data["object"] == "list"
                assert len(data["data"]) == 1
                assert data["data"][0]["id"] == "test-model"
            finally:
                engine.stop()

        asyncio.run(_async())

    def test_spawn_metrics_combined(self):
        """ProxyEngine metrics() combines inner engine + proxy metrics."""

        async def _async():
            inner_port = 18769

            from aibrix.openai_frontend.engine.proxy_engine import ProxyEngine

            spec = EngineSpec(name="test")
            engine = ProxyEngine(
                engine_cmd=_mock_engine_cmd(inner_port),
                engine_port=inner_port,
                engine_host="127.0.0.1",
                spec=spec,
            )

            await engine.start()
            try:
                metrics_text = await engine.metrics()
                assert "proxy_fetch_total" in metrics_text
                assert "proxy_fetch_errors" in metrics_text
            finally:
                engine.stop()

        asyncio.run(_async())

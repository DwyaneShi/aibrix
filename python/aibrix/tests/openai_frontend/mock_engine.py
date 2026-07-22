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

"""Standalone mock engine for E2E tests.

Usage:
    python -m aibrix.tests.openai_frontend.mock_engine --port 18761
"""

from __future__ import annotations

import argparse
import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
app.last_chat_request = None  # type: ignore[attr-defined]


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/v1/models")
async def list_models():
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": "test-model",
                    "object": "model",
                    "created": 0,
                    "owned_by": "test",
                }
            ],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    app.last_chat_request = body  # type: ignore[attr-defined]
    is_stream = body.get("stream", False)
    if is_stream:

        async def generate():
            chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "hello"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    resp = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    return JSONResponse(resp)


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    app.last_chat_request = body  # type: ignore[attr-defined]
    resp = {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"text": "hello", "index": 0, "finish_reason": "stop"}],
    }
    return JSONResponse(resp)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    app.last_chat_request = body  # type: ignore[attr-defined]
    resp = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
            }
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
    return JSONResponse(resp)


@app.get("/metrics")
async def metrics():
    return JSONResponse({})


def main():
    parser = argparse.ArgumentParser(description="Mock engine for E2E tests")
    parser.add_argument("--port", type=int, default=18761)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="error")


if __name__ == "__main__":
    main()

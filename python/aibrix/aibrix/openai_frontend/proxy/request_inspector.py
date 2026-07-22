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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass(frozen=True)
class URLLocation:
    url: str
    path: List[Union[str, int]]  # JSON-pointer-like path into the request dict


class RequestInspector(ABC):
    @abstractmethod
    def extract_urls(self, request_dict: Dict[str, Any]) -> List[URLLocation]: ...


class ChatCompletionInspector(RequestInspector):
    def extract_urls(self, request_dict: Dict[str, Any]) -> List[URLLocation]:
        locations: List[URLLocation] = []
        messages = request_dict.get("messages", [])
        for msg_idx, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part_idx, part in enumerate(content):
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "image_url":
                    continue
                image_url_obj = part.get("image_url")
                if not isinstance(image_url_obj, dict):
                    continue
                url = image_url_obj.get("url", "")
                if url and _is_remote_url(str(url)):
                    locations.append(
                        URLLocation(
                            url=str(url),
                            path=[
                                "messages",
                                msg_idx,
                                "content",
                                part_idx,
                                "image_url",
                                "url",
                            ],
                        )
                    )
        return locations


class CompletionInspector(RequestInspector):
    def extract_urls(self, request_dict: Dict[str, Any]) -> List[URLLocation]:
        return []


class EmbeddingInspector(RequestInspector):
    def extract_urls(self, request_dict: Dict[str, Any]) -> List[URLLocation]:
        return []


def _is_remote_url(url: str) -> bool:
    from aibrix.runtime.downloaders import get_downloader

    try:
        get_downloader(url)
        return True
    except ValueError:
        return False

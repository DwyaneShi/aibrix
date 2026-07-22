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
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from aibrix.runtime.downloaders import ArtifactDownloader, get_downloader


@dataclass(frozen=True)
class FetchResult:
    original_url: str
    local_path: str
    temp_dir: Optional[tempfile.TemporaryDirectory]  # None if not a temp download


class ArtifactFetcher(ABC):
    @abstractmethod
    async def fetch(
        self,
        url: str,
        credentials: Optional[Dict] = None,
    ) -> FetchResult: ...

    @abstractmethod
    async def fetch_many(
        self,
        urls: List[str],
        credentials_map: Optional[Dict[str, Dict]] = None,
    ) -> List[FetchResult]: ...

    @abstractmethod
    async def cleanup(self, results: List[FetchResult]) -> None: ...


class DefaultArtifactFetcher(ArtifactFetcher):
    def __init__(
        self,
        base_temp_dir: Optional[str] = None,
        max_concurrent: int = 5,
    ):
        self._base_temp_dir = base_temp_dir
        self._max_concurrent = max_concurrent
        self._active_tmpdirs: Dict[str, tempfile.TemporaryDirectory] = {}

    async def fetch(
        self,
        url: str,
        credentials: Optional[Dict] = None,
    ) -> FetchResult:
        try:
            get_downloader(url)
        except ValueError:
            return FetchResult(original_url=url, local_path=url, temp_dir=None)

        req_id = uuid.uuid4().hex[:8]
        tmpdir = tempfile.TemporaryDirectory(
            prefix=f"aibrix_artifact_{req_id}_",
            dir=self._base_temp_dir,
        )
        local_dir = tmpdir.name

        downloader: ArtifactDownloader = get_downloader(url)
        await downloader.download(url, local_dir, credentials)

        self._active_tmpdirs[req_id] = tmpdir
        return FetchResult(
            original_url=url,
            local_path=local_dir,
            temp_dir=tmpdir,
        )

    async def fetch_many(
        self,
        urls: List[str],
        credentials_map: Optional[Dict[str, Dict]] = None,
    ) -> List[FetchResult]:
        sem = asyncio.Semaphore(self._max_concurrent)

        async def _fetch_one(url: str) -> FetchResult:
            async with sem:
                creds = credentials_map.get(url) if credentials_map else None
                return await self.fetch(url, creds)

        return list(await asyncio.gather(*[_fetch_one(u) for u in urls]))

    async def cleanup(self, results: List[FetchResult]) -> None:
        for result in results:
            if result.temp_dir is not None:
                result.temp_dir.cleanup()
        self._active_tmpdirs.clear()

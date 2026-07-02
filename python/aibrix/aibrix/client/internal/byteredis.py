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

import asyncio
from datetime import datetime, timedelta
from typing import AbstractSet, Any, Callable, Mapping, Optional, cast

import bytedredis

from aibrix import envs
from aibrix.client.redis import RedisPipeline

# The packaged BytedRedis stack in our environment currently exposes the
# redis-py 3.x ZADD surface: nx / xx / ch / incr are supported, while the
# newer gt / lt flags are not present. A dedicated compatibility test probes
# the installed bytedredis/redis-py signatures so this adapter stays aligned
# with the real client surface instead of rejecting supported options or
# forwarding unsupported ones.
_SUPPORTED_ZADD_OPTIONS: frozenset[str] = frozenset({"nx", "xx", "ch", "incr"})


def _require_unsupported_set_options(**kwargs: Any) -> None:
    unsupported = {
        key: value
        for key, value in kwargs.items()
        if value is not None and value is not False
    }
    if unsupported:
        raise NotImplementedError(
            "BytedRedis client does not support these SET options: "
            + ", ".join(sorted(unsupported))
        )


def _require_unsupported_zadd_options(**kwargs: Any) -> None:
    unsupported = {
        key: value
        for key, value in kwargs.items()
        if key not in _SUPPORTED_ZADD_OPTIONS
        and value is not None
        and value is not False
    }
    if unsupported:
        raise NotImplementedError(
            "BytedRedis client does not support these ZADD options: "
            + ", ".join(sorted(unsupported))
        )


class _AsyncBytedRedisPipeline:
    def __init__(self, pipeline: Any):
        self._pipeline = pipeline

    def get(self, name: bytes | str | memoryview) -> Any:
        return self._pipeline.get(name)

    def smembers(self, name: bytes | str | memoryview) -> Any:
        return self._pipeline.smembers(name)

    def set(
        self,
        name: bytes | str | memoryview,
        value: bytes | bytearray | memoryview | str | int | float,
        ex: int | timedelta | None = None,
        px: int | timedelta | None = None,
        nx: bool = False,
        xx: bool = False,
        keepttl: bool = False,
        get: bool = False,
        exat: int | datetime | None = None,
        pxat: int | datetime | None = None,
        ifeq: str | bytes | None = None,
        ifne: str | bytes | None = None,
        ifdeq: Optional[str] = None,
        ifdne: Optional[str] = None,
    ) -> Any:
        _require_unsupported_set_options(
            keepttl=keepttl,
            get=get,
            exat=exat,
            pxat=pxat,
            ifeq=ifeq,
            ifne=ifne,
            ifdeq=ifdeq,
            ifdne=ifdne,
        )
        return self._pipeline.set(name, value, ex=ex, px=px, nx=nx, xx=xx)

    def zadd(
        self,
        name: bytes | str | memoryview,
        mapping: Mapping[Any, bytes | bytearray | memoryview | str | int | float],
        nx: bool = False,
        xx: bool = False,
        ch: bool = False,
        incr: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> Any:
        _require_unsupported_zadd_options(
            nx=nx,
            xx=xx,
            ch=ch,
            incr=incr,
            gt=gt,
            lt=lt,
        )
        return self._pipeline.zadd(name, mapping, nx=nx, xx=xx, ch=ch, incr=incr)

    def sadd(
        self,
        name: bytes | str | memoryview,
        *values: bytes | bytearray | memoryview | str | int | float,
    ) -> Any:
        return self._pipeline.sadd(name, *values)

    def delete(self, *names: bytes | str | memoryview) -> Any:
        return self._pipeline.delete(*names)

    def zrem(
        self,
        name: bytes | str | memoryview,
        *values: bytes | bytearray | memoryview | str | int | float,
    ) -> Any:
        return self._pipeline.zrem(name, *values)

    def srem(
        self,
        name: bytes | str | memoryview,
        *values: bytes | bytearray | memoryview | str | int | float,
    ) -> Any:
        return self._pipeline.srem(name, *values)

    async def execute(self, raise_on_error: bool = True) -> list[Any]:
        del raise_on_error
        return await asyncio.to_thread(self._pipeline.execute)


class _AsyncBytedRedisClient:
    def __init__(
        self,
        redis_psm: str,
        db: int,
        socket_timeout: Optional[float] = None,
        socket_connect_timeout: Optional[float] = None,
    ):
        # Set default values to make bytedredis work.
        envs.setdefault("CONSUL_HTTP_PORT", "2280")
        envs.setdefault("BYTED_HOST_IPV6", "::1")
        envs.setdefault("MY_HOST_IPV6", "::1")
        envs.setdefault("SEC_KV_AUTH", "1")

        self._client = bytedredis.Client(
            db=db,
            redis_psm=redis_psm,
            disable_metrics=envs.STORAGE_REDIS_DISABLE_METRICS,
            disable_auth=envs.STORAGE_REDIS_DISABLE_AUTH,
            socket_timeout=(
                envs.STORAGE_REDIS_SOCKET_TIMEOUT
                if socket_timeout is None
                else socket_timeout
            ),
            socket_connect_timeout=(
                envs.STORAGE_REDIS_SOCKET_CONNECT_TIMEOUT
                if socket_connect_timeout is None
                else socket_connect_timeout
            ),
        )

    async def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(getattr(self._client, name), *args, **kwargs)

    async def get(self, name: bytes | str | memoryview) -> Optional[bytes]:
        value = await self._call("get", name)
        if isinstance(value, str):
            return value.encode("utf-8")
        return cast(Optional[bytes], value)

    async def set(
        self,
        name: bytes | str | memoryview,
        value: bytes | bytearray | memoryview | str | int | float,
        ex: int | timedelta | None = None,
        px: int | timedelta | None = None,
        nx: bool = False,
        xx: bool = False,
        keepttl: bool = False,
        get: bool = False,
        exat: int | datetime | None = None,
        pxat: int | datetime | None = None,
        ifeq: str | bytes | None = None,
        ifne: str | bytes | None = None,
        ifdeq: Optional[str] = None,
        ifdne: Optional[str] = None,
    ) -> Any:
        _require_unsupported_set_options(
            keepttl=keepttl,
            get=get,
            exat=exat,
            pxat=pxat,
            ifeq=ifeq,
            ifne=ifne,
            ifdeq=ifdeq,
            ifdne=ifdne,
        )
        return await self._call("set", name, value, ex=ex, px=px, nx=nx, xx=xx)

    async def exists(self, *names: bytes | str | memoryview) -> Any:
        return await self._call("exists", *names)

    async def delete(self, *names: bytes | str | memoryview) -> Any:
        return await self._call("delete", *names)

    async def ping(self) -> Any:
        return await self._call("ping")

    async def zadd(
        self,
        name: bytes | str | memoryview,
        mapping: Mapping[Any, bytes | bytearray | memoryview | str | int | float],
        nx: bool = False,
        xx: bool = False,
        ch: bool = False,
        incr: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> Any:
        _require_unsupported_zadd_options(
            nx=nx,
            xx=xx,
            ch=ch,
            incr=incr,
            gt=gt,
            lt=lt,
        )
        return await self._call("zadd", name, mapping, nx=nx, xx=xx, ch=ch, incr=incr)

    async def zrange(
        self,
        name: bytes | str | memoryview,
        start: bytes | bytearray | memoryview | str | int | float,
        end: bytes | bytearray | memoryview | str | int | float,
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: type | Any = float,
        byscore: bool = False,
        bylex: bool = False,
        offset: Optional[int] = None,
        num: Optional[int] = None,
    ) -> list[Any]:
        if desc or byscore or bylex or offset is not None or num is not None:
            raise NotImplementedError(
                "BytedRedis client does not support advanced ZRANGE options"
            )
        if score_cast_func is not float:
            raise NotImplementedError(
                "BytedRedis client does not support custom ZRANGE score_cast_func"
            )
        kwargs = {"withscores": withscores} if withscores else {}
        return await self._call("zrange", name, start, end, **kwargs)

    async def zrevrange(
        self,
        name: bytes | str | memoryview,
        start: int,
        end: int,
        withscores: bool = False,
        score_cast_func: type | Any = float,
    ) -> list[Any]:
        if score_cast_func is not float:
            raise NotImplementedError(
                "BytedRedis client does not support custom ZREVRANGE score_cast_func"
            )
        kwargs = {"withscores": withscores} if withscores else {}
        return await self._call("zrevrange", name, start, end, **kwargs)

    async def zrevrank(
        self,
        name: bytes | str | memoryview,
        value: bytes | bytearray | memoryview | str | int | float,
    ) -> Optional[int]:
        return await self._call("zrevrank", name, value)

    async def zrank(
        self,
        name: bytes | str | memoryview,
        value: bytes | bytearray | memoryview | str | int | float,
    ) -> Optional[int]:
        return await self._call("zrank", name, value)

    async def zrem(
        self,
        name: bytes | str | memoryview,
        *values: bytes | bytearray | memoryview | str | int | float,
    ) -> Any:
        return await self._call("zrem", name, *values)

    async def sadd(
        self,
        name: bytes | str | memoryview,
        *values: bytes | bytearray | memoryview | str | int | float,
    ) -> Any:
        return await self._call("sadd", name, *values)

    async def srem(
        self,
        name: bytes | str | memoryview,
        *values: bytes | bytearray | memoryview | str | int | float,
    ) -> Any:
        return await self._call("srem", name, *values)

    async def smembers(self, name: bytes | str | memoryview) -> AbstractSet[Any]:
        return await self._call("smembers", name)

    async def strlen(self, key: bytes | str | memoryview) -> int:
        return await self._call("strlen", key)

    def pipeline(
        self, transaction: bool = True, shard_hint: Optional[str] = None
    ) -> RedisPipeline:
        del transaction, shard_hint
        return cast(RedisPipeline, _AsyncBytedRedisPipeline(self._client.pipeline()))

    def __getattr__(self, name: str):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        async def _async_call(*args, **kwargs):
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _async_call

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await asyncio.to_thread(close)

    async def run_pipeline(
        self, callback: Callable[[RedisPipeline], None]
    ) -> list[Any]:
        pipeline = self.pipeline()
        callback(pipeline)
        return await pipeline.execute()

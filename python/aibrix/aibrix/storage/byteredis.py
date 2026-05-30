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
import time
from typing import (
    AbstractSet,
    Any,
    BinaryIO,
    Callable,
    Optional,
    Protocol,
    TextIO,
    Union,
)

import bytedredis

from aibrix import envs
from aibrix.storage.base import (
    BaseStorage,
    PutObjectOptions,
    StorageConfig,
    StorageType,
)
from aibrix.storage.reader import Reader
from aibrix.storage.utils import ObjectMetadata


class RedisPipeline(Protocol):
    def zadd(self, key: str, mapping: dict[str, float]) -> Any: ...

    def sadd(self, key: str, value: str) -> Any: ...

    def delete(self, key: str) -> Any: ...

    def zrem(self, key: str, value: str) -> Any: ...

    def srem(self, key: str, value: str) -> Any: ...


class AsyncRedis(Protocol):
    async def get(self, key: str) -> Optional[bytes]: ...

    async def set(
        self,
        key: str,
        value: bytes | str,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> Any: ...

    async def exists(self, key: str) -> Any: ...

    async def delete(self, key: str) -> Any: ...

    async def ping(self) -> Any: ...

    async def zadd(self, key: str, mapping: dict[str, float]) -> Any: ...

    async def zrange(
        self, key: str, start: int, end: int, withscores: bool = False
    ) -> list[bytes]: ...

    async def zrevrange(self, key: str, start: int, end: int) -> list[bytes | str]: ...

    async def zrem(self, key: str, value: str) -> Any: ...

    async def smembers(self, key: str) -> AbstractSet[bytes]: ...

    async def strlen(self, key: str) -> int: ...

    async def aclose(self) -> None: ...

    async def run_pipeline(
        self, callback: Callable[[RedisPipeline], None]
    ) -> list[Any]: ...


class _AsyncBytedRedisClient:
    def __init__(
        self,
        redis_psm: str,
        db: int,
    ):
        # Set default values to make bytedredis work
        envs.setdefault("CONSUL_HTTP_PORT", "2280")
        envs.setdefault("BYTED_HOST_IPV6", "::1")
        envs.setdefault("MY_HOST_IPV6", "::1")
        envs.setdefault("SEC_KV_AUTH", "1")

        self._client = bytedredis.Client(
            db=db,
            redis_psm=redis_psm,
            disable_metrics=envs.STORAGE_REDIS_DISABLE_METRICS,
            disable_auth=envs.STORAGE_REDIS_DISABLE_AUTH,
            socket_timeout=envs.STORAGE_REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=envs.STORAGE_REDIS_SOCKET_CONNECT_TIMEOUT,
        )

    async def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(getattr(self._client, name), *args, **kwargs)

    async def get(self, key: str) -> Optional[bytes]:
        return await self._call("get", key)

    async def set(
        self,
        key: str,
        value: bytes | str,
        ex: Optional[int] = None,
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> Any:
        return await self._call("set", key, value, ex=ex, px=px, nx=nx, xx=xx)

    async def exists(self, key: str) -> Any:
        return await self._call("exists", key)

    async def delete(self, key: str) -> Any:
        return await self._call("delete", key)

    async def ping(self) -> Any:
        return await self._call("ping")

    async def zadd(self, key: str, mapping: dict[str, float]) -> Any:
        return await self._call("zadd", key, mapping)

    async def zrange(
        self, key: str, start: int, end: int, withscores: bool = False
    ) -> list[bytes]:
        return await self._call("zrange", key, start, end, withscores=withscores)

    async def zrevrange(self, key: str, start: int, end: int) -> list[bytes | str]:
        return await self._call("zrevrange", key, start, end)

    async def zrem(self, key: str, value: str) -> Any:
        return await self._call("zrem", key, value)

    async def smembers(self, key: str) -> AbstractSet[bytes]:
        return await self._call("smembers", key)

    async def strlen(self, key: str) -> int:
        return await self._call("strlen", key)

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
        def _run():
            pipeline = self._client.pipeline()
            callback(pipeline)
            return pipeline.execute()

        return await asyncio.to_thread(_run)


class RedisStorage(BaseStorage):
    """Redis storage implementation.

    This implementation uses Redis as a key-value store with the following features:
    - No content_type or metadata support for put_object
    - No head_object or object_exists support
    - Hierarchical key support using Redis sets (e.g., "xxx/yyy" creates set "xxx:index")
    - Simple get/put/delete operations
    - List operations that work with Redis structures
    - Timestamp-ordered listing: list_objects returns keys ordered by creation timestamp

    Timestamp Tracking:
    - All keys are tracked in "timestamps:all" sorted set with creation time as score
    - Hierarchical keys are also tracked in "timestamps:{parent}" sorted sets
    - list_objects returns keys in chronological order (oldest first)
    - Timestamps are cleaned up automatically when objects are deleted
    """

    def __init__(
        self,
        redis_psm: str,
        db: int = 0,
        config: Optional[StorageConfig] = None,
    ):
        super().__init__(config)
        self.db = db
        self.redis_psm = redis_psm
        self._redis_clients: dict[int, AsyncRedis] = {}

    def get_type(self) -> StorageType:
        """Get the type of storage.

        Returns:
            Type of storage, set to StorageType.REDIS
        """
        return StorageType.REDIS

    async def _get_redis(self) -> AsyncRedis:
        """Get Redis connection, creating it if necessary."""
        if not self.redis_psm:
            raise ValueError("redis_psm is required for RedisStorage")
        loop_id = id(asyncio.get_running_loop())
        redis_client = self._redis_clients.get(loop_id)
        if redis_client is None:
            redis_client = _AsyncBytedRedisClient(self.redis_psm, self.db)
            self._redis_clients[loop_id] = redis_client
        return redis_client

    async def close(self):
        """Close Redis connection."""
        for redis_client in self._redis_clients.values():
            await redis_client.aclose()
        self._redis_clients.clear()

    def _parse_hierarchical_key(self, key: str) -> tuple[Optional[str], str]:
        """Parse hierarchical key into parent and item.

        Args:
            key: Key like "xxx/yyy" or "simple_key"

        Returns:
            Tuple of (parent_key, item_key). For "xxx/yyy" returns ("xxx", "yyy").
            For "simple_key" returns (None, "simple_key").
        """
        if "/" in key:
            parts = key.split("/", 1)
            return parts[0], parts[1]
        return None, key

    async def put_object(
        self,
        key: str,
        data: Union[bytes, str, BinaryIO, TextIO, Reader],
        content_type: Optional[str] = None,  # Ignored for Redis
        metadata: Optional[dict[str, str]] = None,  # Ignored for Redis
        options: Optional[PutObjectOptions] = None,
    ) -> bool:
        """Put an object to Redis storage with advanced options.

        If key contains "/", creates a Redis list for the parent part.
        For example, "xxx/yyy" will create a list "xxx" and add "yyy" to it,
        then store the actual data under the full key "xxx/yyy".

        Args:
            key: Object key/path
            data: Data to store
            content_type: Ignored for Redis
            metadata: Ignored for Redis
            options: Advanced options for put operation

        Returns:
            True if object was stored, False if conditional operation failed
        """
        # Validate options
        self._validate_put_options(options)

        redis_client = await self._get_redis()

        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            # File-like object or Reader
            reader = self._wrap_data(data)
            data_bytes = reader.read_all()

        # Parse hierarchical key
        parent_key, item_key = self._parse_hierarchical_key(key)

        # Prepare Redis SET options from PutObjectOptions
        redis_ex = None
        redis_px = None
        redis_nx = False
        redis_xx = False

        if options:
            if options.ttl_seconds is not None:
                redis_ex = options.ttl_seconds
            elif options.ttl_milliseconds is not None:
                redis_px = options.ttl_milliseconds

            redis_nx = options.set_if_not_exists
            redis_xx = options.set_if_exists

        # Store the actual data with options
        result = await redis_client.set(
            key, data_bytes, ex=redis_ex, px=redis_px, nx=redis_nx, xx=redis_xx
        )

        # Check if the SET operation succeeded
        if result is None:
            # Conditional SET failed (NX or XX condition not met)
            return False

        # Store creation timestamp for ordering (only if SET succeeded)
        timestamp = time.time()

        def _track_write(pipeline: RedisPipeline) -> None:
            pipeline.zadd("timestamps:all", {key: timestamp})
            # If hierarchical, add to parent list and track timestamp
            if parent_key is not None:
                # Add item to parent list (only if not already present)
                pipeline.sadd(f"{parent_key}:index", item_key)
                # Also track timestamp for hierarchical objects
                pipeline.zadd(f"timestamps:{parent_key}", {item_key: timestamp})

        await redis_client.run_pipeline(_track_write)

        return True

    async def get_object(
        self,
        key: str,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
    ) -> bytes:
        """Get an object from Redis storage.

        Args:
            key: Object key/path
            range_start: Start byte position for range get
            range_end: End byte position for range get (inclusive)

        Returns:
            Object data as bytes

        Raises:
            FileNotFoundError: If object does not exist
        """
        if (range_start is None) != (range_end is None):
            raise ValueError(
                "range_start and range_end must both be provided or both be None"
            )
        redis_client = await self._get_redis()

        data = await redis_client.get(key)
        if data is None:
            raise FileNotFoundError(f"Object not found: {key}")

        # Handle range requests
        if range_start is not None:
            assert range_end is not None
            return data[range_start : range_end + 1]

        return data

    async def delete_object(self, key: str) -> None:
        """Delete an object from Redis storage.

        Also removes the key from parent list if it's hierarchical.

        Args:
            key: Object key/path
        """
        redis_client = await self._get_redis()

        # Parse hierarchical key
        parent_key, item_key = self._parse_hierarchical_key(key)

        def _delete_and_cleanup(pipeline: RedisPipeline) -> None:
            # Delete the actual data
            pipeline.delete(key)
            # Clean up timestamp tracking
            pipeline.zrem("timestamps:all", key)
            # If hierarchical, remove from parent list and timestamps
            if parent_key is not None:
                pipeline.srem(f"{parent_key}:index", item_key)
                pipeline.zrem(f"timestamps:{parent_key}", item_key)

        await redis_client.run_pipeline(_delete_and_cleanup)

    async def list_objects(
        self,
        prefix: str = "",
        delimiter: Optional[str] = None,
        limit: Optional[int] = None,
        continuation_token: Optional[str] = None,
    ) -> tuple[list[str], Optional[str]]:
        """List objects with given prefix ordered by creation timestamp.

        For Redis, this works with both direct keys and hierarchical structures:
        - If prefix corresponds to a Redis list (has :index suffix), returns list members ordered by creation time
        - Otherwise, scans for keys matching the prefix pattern and orders by creation time
        - Supports efficient token-based pagination using Redis ZRANGE

        Args:
            prefix: Key prefix to filter objects
            delimiter: Delimiter for hierarchical listing (typically "/")
            limit: Maximum number of objects to return (None for no limit)
            continuation_token: Offset position as string (e.g., "10" for offset 10)

        Returns:
            Tuple of (object_keys, next_continuation_token)
            - object_keys: List of object keys ordered by creation timestamp (oldest first)
            - next_continuation_token: String offset for next page (None if no more pages)
        """
        redis_client = await self._get_redis()

        # Parse continuation token as offset (default to 0)
        offset = 0
        if continuation_token:
            try:
                offset = int(continuation_token)
            except (ValueError, TypeError):
                offset = 0

        # Check if this prefix corresponds to a list index
        list_key = f"{prefix}:index"
        if await redis_client.exists(list_key):
            # Get members ordered by timestamp from the sorted set
            timestamp_key = f"timestamps:{prefix}"
            if await redis_client.exists(timestamp_key):
                # Calculate pagination bounds
                start = offset
                end = offset + limit - 1 if limit is not None else -1

                # Get members ordered by timestamp (oldest first) with pagination
                members_with_scores = await redis_client.zrange(
                    timestamp_key, start, end, withscores=False
                )
                members = [member.decode("utf-8") for member in members_with_scores]

                # Check if there are more items for next page
                has_more = False
                if limit is not None and len(members) == limit:
                    # Check if there's at least one more item after this page
                    next_item = await redis_client.zrange(
                        timestamp_key, start + limit, start + limit, withscores=False
                    )
                    has_more = len(next_item) > 0
            else:
                # Fallback to unordered members if no timestamps
                members_raw = await redis_client.smembers(list_key)
                all_members = [member.decode("utf-8") for member in members_raw]

                # Apply pagination to unordered list
                if offset > 0:
                    all_members = all_members[offset:]

                members = all_members[:limit] if limit is not None else all_members
                has_more = limit is not None and len(all_members) > limit

            if delimiter:
                # Return hierarchical format with delimiter
                result_keys = [f"{prefix}{delimiter}{member}" for member in members]
            else:
                # Return just the member names
                result_keys = members

            # Generate next token
            next_token = str(offset + len(members)) if has_more else None
            return result_keys, next_token

        # Fallback to key scanning with timestamp ordering
        if prefix:
            # For prefix searches, get all keys from timestamp sorted set and filter
            all_keys_with_timestamps = await redis_client.zrange(
                "timestamps:all", 0, -1, withscores=False
            )

            filtered_keys = []
            for key_bytes in all_keys_with_timestamps:
                key_str = key_bytes.decode("utf-8")
                # Filter by prefix and exclude internal keys
                if (
                    key_str.startswith(prefix)
                    and not key_str.endswith(":index")
                    and not key_str.startswith("timestamps:")
                ):
                    filtered_keys.append(key_str)

            # Apply pagination after filtering
            if offset > 0:
                filtered_keys = filtered_keys[offset:]

            keys = filtered_keys[:limit] if limit is not None else filtered_keys
            has_more = limit is not None and len(filtered_keys) > limit
        else:
            # For no prefix, get all keys first, then filter and paginate
            # This ensures consistent pagination even when internal keys are present
            all_keys_with_timestamps = await redis_client.zrange(
                "timestamps:all", 0, -1, withscores=False
            )
            all_user_keys = []
            for key_bytes in all_keys_with_timestamps:
                key_str = key_bytes.decode("utf-8")
                # Exclude internal keys
                if not key_str.endswith(":index") and not key_str.startswith(
                    "timestamps:"
                ):
                    all_user_keys.append(key_str)

            # Apply pagination to filtered keys
            paginated_keys = all_user_keys[offset:]

            keys = paginated_keys[:limit] if limit is not None else paginated_keys
            has_more = limit is not None and len(paginated_keys) > limit

        # Apply delimiter filtering if specified
        if delimiter and prefix:
            filtered_keys = []
            for key in keys:
                if key.startswith(prefix):
                    remaining = key[len(prefix) :]
                    if delimiter in remaining:
                        # Extract the next level only
                        next_part = remaining.split(delimiter, 1)[0]
                        hierarchical_key = f"{prefix}{next_part}{delimiter}"
                        if hierarchical_key not in filtered_keys:
                            filtered_keys.append(hierarchical_key)
                    else:
                        filtered_keys.append(key)
            keys = filtered_keys
            # Note: has_more might not be accurate after delimiter filtering

        # Generate next token
        next_token = str(offset + len(keys)) if has_more else None
        return keys, next_token

    async def object_exists(self, key: str) -> bool:
        """Check if object exists.

        Note: Not directly supported as per requirements, but implemented
        for compatibility with base class.

        Args:
            key: Object key/path

        Returns:
            True if object exists, False otherwise
        """
        redis_client = await self._get_redis()
        return bool(await redis_client.exists(key))

    async def get_object_size(self, key: str) -> int:
        """Get object size in bytes.

        Args:
            key: Object key/path

        Returns:
            Object size in bytes

        Raises:
            FileNotFoundError: If object does not exist
        """
        redis_client = await self._get_redis()

        size = await redis_client.strlen(key)
        if size == 0:
            # Check if key actually exists
            if not await redis_client.exists(key):
                raise FileNotFoundError(f"Object not found: {key}")

        return size

    async def head_object(self, key: str) -> ObjectMetadata:
        """Get object metadata.

        Note: Not supported for Redis as per requirements.

        Args:
            key: Object key/path

        Raises:
            NotImplementedError: Redis storage doesn't support metadata
        """
        raise NotImplementedError("head_object not supported for Redis storage")

    async def _native_create_multipart_upload(
        self,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """Create a multipart upload session.

        Note: Not needed for Redis as per requirements.

        Raises:
            NotImplementedError: Multipart upload not needed for Redis
        """
        raise NotImplementedError("Multipart upload not needed for Redis storage")

    async def _native_upload_part(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        data: Union[str, bytes, BinaryIO, TextIO, Reader],
    ) -> str:
        """Upload a part in a multipart upload.

        Note: Not needed for Redis as per requirements.

        Raises:
            NotImplementedError: Multipart upload not needed for Redis
        """
        raise NotImplementedError("Multipart upload not needed for Redis storage")

    async def _native_complete_multipart_upload(
        self,
        key: str,
        upload_id: str,
        parts: list[dict[str, Union[str, int]]],
    ) -> None:
        """Complete a multipart upload.

        Note: Not needed for Redis as per requirements.

        Raises:
            NotImplementedError: Multipart upload not needed for Redis
        """
        raise NotImplementedError("Multipart upload not needed for Redis storage")

    async def _native_abort_multipart_upload(
        self,
        key: str,
        upload_id: str,
    ) -> None:
        """Abort a multipart upload.

        Note: Not needed for Redis as per requirements.

        Raises:
            NotImplementedError: Multipart upload not needed for Redis
        """
        raise NotImplementedError("Multipart upload not needed for Redis storage")

    # Feature Support Methods
    def is_ttl_supported(self) -> bool:
        """Check if TTL (Time To Live) is supported.

        Returns:
            True - Redis supports TTL
        """
        return True

    def is_set_if_not_exists_supported(self) -> bool:
        """Check if conditional SET IF NOT EXISTS is supported.

        Returns:
            True - Redis supports NX option
        """
        return True

    def is_set_if_exists_supported(self) -> bool:
        """Check if conditional SET IF EXISTS is supported.

        Returns:
            True - Redis supports XX option
        """
        return True

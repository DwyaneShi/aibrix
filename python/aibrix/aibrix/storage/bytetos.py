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
import json
import mimetypes
import os
from dataclasses import replace
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, BinaryIO, Optional, TextIO, Union
from urllib.parse import unquote, urlparse

import bytedtos
from bytedtos import StaticCredentials
from bytedtos.errors import TosException
from tos.exceptions import TosClientError, TosServerError

from aibrix.storage.base import PutObjectOptions, StorageConfig, StorageType
from aibrix.storage.base2 import BaseStorage2
from aibrix.storage.reader import Reader
from aibrix.storage.utils import ObjectMetadata


class TOSStorage(BaseStorage2):
    """TOS implementation that supports both in-house and Volcano access modes."""

    def __init__(
        self,
        bucket_name: str,
        access_key: str,
        secret_key: str,
        endpoint: Optional[str] = None,
        region: Optional[str] = None,
        force_volcano: bool = False,
        enable_crc: bool = False,
        idc: Optional[str] = None,
        service: str = "toutiao.tos.tosapi",
        cluster: str = "default",
        remote_psm: str = "inf.aibrix.metadata",
        config: Optional[StorageConfig] = None,
    ):
        resolved_config = config or StorageConfig()
        if resolved_config.strict_multipart_min_part_size is None:
            resolved_config = replace(
                resolved_config, strict_multipart_min_part_size=True
            )
        super().__init__(resolved_config)
        self.bucket_name = bucket_name
        self.force_volcano = force_volcano or os.getenv("PAAS_CLOUD_ENV") == "VOLCANO"
        use_psm_client = bool(idc) and not self.force_volcano

        if (
            not use_psm_client
            and not self.force_volcano
            and endpoint
            and self._is_public_tos_endpoint(endpoint)
        ):
            raise ValueError(
                "public TOS endpoints require PAAS_CLOUD_ENV=VOLCANO or force_volcano=True"
            )

        cred = StaticCredentials(access_key, secret_key)

        try:
            kwargs: dict[str, Any] = {}
            if use_psm_client:
                # PSM mode is intentionally mutually exclusive with endpoint
                # mode. Passing endpoint/region here can trigger bytedtos'
                # endpoint validation before internal PSM routing is used.
                kwargs.update(
                    {
                        "service": service,
                        "cluster": cluster,
                        "idc": idc,
                        "remote_psm": remote_psm,
                    }
                )
            else:
                kwargs["enable_crc64"] = enable_crc
                # bytedtos.Client uses requests' HTTPAdapter pool sizing via
                # ``connection_pool_size`` on the non-Volcano path. VeClient
                # currently ignores this kwarg, but passing it through keeps
                # the config surface consistent across TOS backends.
                kwargs["connection_pool_size"] = max(self.config.max_concurrency, 1)
                if self.force_volcano:
                    kwargs["force_volcano"] = True
                    kwargs["ve_cred"] = cred
                    kwargs["ve_region"] = region
                    kwargs["ve_endpoint"] = endpoint
                else:
                    kwargs["region"] = region
                    kwargs["endpoint"] = endpoint

            self.client = bytedtos.ClientV2(
                bucket=bucket_name,
                cred=cred,
                **kwargs,
            )
        except (TosException, TosClientError, TosServerError) as e:
            raise ValueError(f"Failed to create TOS client: {e}")

    def _is_public_tos_endpoint(self, endpoint: str) -> bool:
        parsed = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}")
        host = parsed.hostname or endpoint
        host = host.lower()
        return host.endswith(".volces.com") or host.endswith(".volcengineapi.com")

    def _build_headers(
        self,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        resolved_content_type = content_type or mimetypes.guess_type(key)[0]
        if resolved_content_type:
            headers["content-type"] = resolved_content_type
        else:
            headers["content-type"] = "application/octet-stream"
        if metadata:
            for meta_key, meta_value in metadata.items():
                headers[f"x-tos-meta-{meta_key}"] = meta_value
            headers["x-tos-meta-aibrix-metadata"] = json.dumps(
                metadata, separators=(",", ":")
            )
        return headers

    def _build_upload_payload(
        self,
        key: str,
        file_content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> tuple[bytes, dict[str, str]]:
        headers = self._build_headers(key, content_type, metadata)
        payload = file_content
        if len(payload) == 0:
            payload = b"\0"
            headers["x-tos-meta-aibrix-empty-object"] = "1"
        headers["content-length"] = str(len(payload))
        return payload, headers

    def _is_not_found_error(self, error: Exception) -> bool:
        code = getattr(error, "code", None)
        if str(code) == "404":
            return True
        message = str(error)
        return "404" in message or "NoSuchKey" in message or "Not Found" in message

    def _response_headers(self, response: Any) -> dict[str, str]:
        headers = getattr(response, "headers", None)
        if headers is None:
            return {}
        return dict(headers)

    def _response_data(self, response: Any) -> bytes:
        data = getattr(response, "data", None)
        if data is not None:
            return data
        raw = getattr(response, "raw", None)
        if raw is not None and hasattr(raw, "read"):
            return raw.read()
        reader = getattr(response, "read", None)
        if callable(reader):
            return reader()
        return b""

    def _response_payload(self, response: Any) -> dict[str, Any]:
        try:
            payload = response.json
            if isinstance(payload, dict):
                inner_payload = payload.get("payload")
                if isinstance(inner_payload, dict):
                    return inner_payload
                return payload
        except Exception:
            pass

        data = self._response_data(response)
        if not data:
            return {}

        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception:
            return {}

        if isinstance(payload, dict):
            inner_payload = payload.get("payload")
            if isinstance(inner_payload, dict):
                return inner_payload
            return payload

        return {}

    def _extract_content_length(self, response: Any) -> int:
        size = getattr(response, "size", None)
        if size is not None:
            return int(size)
        headers = self._response_headers(response)
        return int(headers.get("Content-Length") or headers.get("content-length") or 0)

    def _extract_content_type(self, response: Any, key: str) -> str:
        headers = self._response_headers(response)
        return (
            headers.get("Content-Type")
            or headers.get("content-type")
            or mimetypes.guess_type(key)[0]
            or "application/octet-stream"
        )

    def _extract_etag(self, response: Any) -> str:
        headers = self._response_headers(response)
        etag = (
            headers.get("x-tos-etag")
            or headers.get("X-Tos-Etag")
            or headers.get("Etag")
            or headers.get("ETag")
            or headers.get("etag")
            or ""
        )
        return etag.strip('"')

    def _extract_last_modified(self, response: Any) -> Optional[datetime]:
        last_modified = getattr(response, "last_modify_time", None)
        if last_modified is None:
            headers = self._response_headers(response)
            last_modified = self._parse_datetime(
                headers.get("Last-Modified") or headers.get("last-modified")
            )
        if last_modified is not None and hasattr(last_modified, "replace"):
            try:
                return last_modified.replace(tzinfo=None)
            except TypeError:
                return last_modified
        return None

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except Exception:
            return None
        if hasattr(parsed, "replace"):
            try:
                return parsed.replace(tzinfo=None)
            except TypeError:
                return parsed
        return parsed

    def _extract_metadata(self, response: Any) -> dict[str, str]:
        all_metadata = self._extract_all_metadata(response)
        for internal_key in (
            "aibrix-empty-object",
            "aibrix-multipart-manifest",
            "aibrix-upload-id",
            "aibrix-content-length",
            "aibrix-content-type",
        ):
            all_metadata.pop(internal_key, None)
        manifest_metadata = all_metadata.pop("aibrix-metadata", None)
        if manifest_metadata:
            try:
                all_metadata.update(json.loads(manifest_metadata))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return all_metadata

    def _extract_all_metadata(self, response: Any) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for header_key, header_value in self._response_headers(response).items():
            normalized_key = header_key.lower()
            if normalized_key.startswith("x-tos-meta-"):
                metadata_key = normalized_key[len("x-tos-meta-") :]
                while metadata_key.startswith("x-tos-meta-"):
                    metadata_key = metadata_key[len("x-tos-meta-") :]
                if metadata_key in {"content-type", "content-length"}:
                    continue
                metadata[metadata_key] = unquote(str(header_value))
        return metadata

    def _extract_header(
        self,
        response: Any,
        *names: str,
    ) -> Optional[str]:
        headers = self._response_headers(response)
        for name in names:
            value = headers.get(name)
            if value is not None:
                return value
        lower_headers = {key.lower(): value for key, value in headers.items()}
        for name in names:
            value = lower_headers.get(name.lower())
            if value is not None:
                return value
        return None

    def _is_empty_object(self, response: Any) -> bool:
        return self._extract_all_metadata(response).get("aibrix-empty-object") == "1"

    def _is_manifest_object(self, response: Any) -> bool:
        return (
            self._extract_all_metadata(response).get("aibrix-multipart-manifest") == "1"
        )

    async def _read_manifest_object(
        self, response: Any
    ) -> tuple[dict[str, Any], bytes]:
        metadata = self._extract_all_metadata(response)
        upload_id = metadata.get("aibrix-upload-id")
        if not upload_id:
            raise ValueError("Missing multipart manifest upload ID")
        raw_manifest = self._response_data(response)
        manifest = json.loads(raw_manifest.decode("utf-8")) if raw_manifest else {}
        parts = manifest.get("parts", [])
        combined = bytearray()
        for part in sorted(parts, key=lambda item: int(item["part_number"])):
            part_number = int(part["part_number"])
            part_key = self._multipart_upload_key(upload_id, f"part_{part_number:05d}")
            try:
                part_data = await self.get_object(part_key)
            except FileNotFoundError:
                chunk_keys, _ = await self.list_objects(f"{part_key}/")
                part_data = b""
                for chunk_key in sorted(chunk_keys):
                    part_data += await self.get_object(chunk_key)
            combined.extend(part_data)
        return metadata, bytes(combined)

    def _payload_objects_and_prefixes(
        self, payload: dict[str, Any]
    ) -> tuple[list[str], list[str], bool]:
        objects: list[str] = []
        payload_objects = payload.get("objects") or payload.get("contents") or []
        for obj in payload_objects:
            object_key = None
            if isinstance(obj, dict):
                object_key = obj.get("key") or obj.get("Key")
            if object_key:
                objects.append(object_key)

        common_prefixes = [
            prefix
            for prefix in (
                payload.get("commonPrefix") or payload.get("commonPrefixes") or []
            )
            if prefix
        ]
        is_truncated = bool(payload.get("isTruncated"))
        return objects, common_prefixes, is_truncated

    def _collect_all_objects(self, prefix: str) -> list[str]:
        response = self.client.list_prefix(prefix, "", "", 1000)
        payload = self._response_payload(response)
        objects, common_prefixes, _ = self._payload_objects_and_prefixes(payload)
        for common_prefix in common_prefixes:
            objects.extend(self._collect_all_objects(common_prefix))
        return objects

    async def _put_bytes_direct(
        self,
        key: str,
        file_content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        payload, headers = self._build_upload_payload(
            key, file_content, content_type, metadata
        )

        def _put_object() -> None:
            try:
                self.client.put_object(key, payload, headers=headers)
            except (TosException, TosClientError, TosServerError) as e:
                raise ValueError(f"Failed to put object {key}: {e}")

        await asyncio.get_event_loop().run_in_executor(None, _put_object)

    async def _write_multipart_state(
        self, upload_id: str, state: dict[str, Any]
    ) -> None:
        await self.put_object(
            self._multipart_upload_key(upload_id),
            json.dumps(state, separators=(",", ":")),
            "application/json",
        )

    async def _read_multipart_state(self, upload_id: str) -> dict[str, Any]:
        try:
            raw_state = await self.get_object(self._multipart_upload_key(upload_id))
        except FileNotFoundError:
            raise ValueError(f"Multipart upload not found: {upload_id}")
        return json.loads(raw_state.decode("utf-8"))

    async def _delete_multipart_state(self, upload_id: str) -> None:
        await self.delete_object(self._multipart_upload_key(upload_id))

    async def _complete_local_multipart_upload(
        self,
        key: str,
        upload_id: str,
        parts: list[dict[str, Union[str, int]]],
        state: dict[str, Any],
    ) -> None:
        sorted_parts = sorted(parts, key=lambda item: int(item["part_number"]))
        combined_data = bytearray()
        for part in sorted_parts:
            part_number = int(part["part_number"])
            part_key = self._multipart_upload_key(upload_id, f"part_{part_number:05d}")
            try:
                part_data = await self.get_object(part_key)
            except FileNotFoundError:
                chunk_prefix = f"{part_key}/"
                chunk_keys, _ = await self.list_objects(chunk_prefix)
                if not chunk_keys:
                    raise
                part_data = b""
                for chunk_key in sorted(chunk_keys):
                    part_data += await self.get_object(chunk_key)
            combined_data.extend(part_data)

        manifest = {
            "upload_id": upload_id,
            "parts": sorted_parts,
        }
        await self._put_bytes_direct(
            key,
            json.dumps(manifest, separators=(",", ":")).encode("utf-8"),
            "application/json",
            {
                **(state.get("metadata") or {}),
                "aibrix-multipart-manifest": "1",
                "aibrix-upload-id": upload_id,
                "aibrix-content-length": str(len(combined_data)),
                "aibrix-content-type": state.get("content_type")
                or "application/octet-stream",
            },
        )

    def _list_objects_via_volcano_client(
        self,
        prefix: str,
        delimiter: Optional[str],
        limit: Optional[int],
        continuation_token: Optional[str],
        after_key: Optional[str],
    ) -> tuple[list[str], Optional[str]]:
        chosen_client = getattr(self.client, "_choose_client", lambda: None)()
        ve_sdk_client = getattr(chosen_client, "client", None)
        if ve_sdk_client is None or not hasattr(ve_sdk_client, "list_objects"):
            raise TypeError("Volcano list fallback unavailable")

        response = ve_sdk_client.list_objects(
            self.bucket_name,
            prefix=prefix,
            delimiter=delimiter or "",
            marker=continuation_token or after_key or "",
            max_keys=min(limit, 1000) if limit is not None else 1000,
        )

        objects = [obj.key for obj in getattr(response, "contents", [])]
        if delimiter:
            for common_prefix in getattr(response, "common_prefixes", []):
                common_prefix_value = getattr(common_prefix, "prefix", None)
                if common_prefix_value:
                    objects.append(common_prefix_value)

        next_token = (
            objects[-1]
            if getattr(response, "is_truncated", False) and objects
            else None
        )
        return objects, next_token

    def get_type(self) -> StorageType:
        """Get the type of storage.

        Returns:
            Type of storage, set to StorageType.TOS
        """
        return StorageType.TOS

    async def put_object(
        self,
        key: str,
        data: Union[bytes, str, BinaryIO, TextIO, Reader],
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
        options: Optional[PutObjectOptions] = None,
    ) -> bool:
        """Put an object to TOS."""
        # Validate options (TOS doesn't support advanced options)
        self._validate_put_options(options)

        # Unify all data types using Reader wrapper
        reader = self._wrap_data(data)

        try:
            # Check if we should use multipart upload
            try:
                size = reader.get_size()
                if (
                    not key.startswith(".multipart/")
                    and size >= self.config.multipart_threshold
                ):
                    await self.multipart_upload(
                        key,
                        reader,
                        content_type,
                        metadata,
                        bysize=self.config.multipart_threshold,
                    )
                    return True
            except (OSError, IOError, ValueError):
                pass

            # For small files, read all data and upload directly as BytesIO
            # TOS client has issues with custom file-like objects for CRC calculation
            file_content = reader.read_all()
            payload, headers = self._build_upload_payload(
                key, file_content, content_type, metadata
            )

            def _put_object():
                try:
                    self.client.put_object(key, payload, headers=headers)
                except (TosException, TosClientError, TosServerError) as e:
                    raise ValueError(f"Failed to put object {key}: {e}")

            await asyncio.get_event_loop().run_in_executor(None, _put_object)
        finally:
            if not isinstance(data, Reader):
                reader.close()

        return True  # TOS storage always succeeds

    async def get_object(
        self,
        key: str,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
    ) -> bytes:
        """Get an object from TOS with optional range support."""
        if (range_start is None) != (range_end is None):
            raise ValueError(
                "range_start and range_end must both be provided or both be None"
            )

        def _get_object():
            try:
                if range_start is None and range_end is None:
                    return self.client.get_object(key)
                return self.client.get_object_range(key, range_start, range_end)
            except (TosException, TosClientError, TosServerError) as e:
                if self._is_not_found_error(e):
                    raise FileNotFoundError(f"Object not found: {key}")
                raise ValueError(f"Failed to get object {key}: {e}")

        response = await asyncio.get_event_loop().run_in_executor(None, _get_object)
        if self._is_manifest_object(response):
            _, data = await self._read_manifest_object(response)
        else:
            data = self._response_data(response)
            if self._is_empty_object(response):
                data = b""

        return data

    async def delete_object(self, key: str) -> None:
        """Delete an object from TOS."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.client.head_object(key)
            )
        except (TosException, TosClientError, TosServerError):
            response = None

        if response is not None and self._is_manifest_object(response):
            upload_id = self._extract_all_metadata(response).get("aibrix-upload-id")
            if upload_id:
                await self.abort_multipart_upload(key, upload_id)

        def _delete_object():
            try:
                self.client.delete_object(key)
            except (TosException, TosClientError, TosServerError) as e:
                if not self._is_not_found_error(e):
                    raise ValueError(f"Failed to delete object {key}: {e}")

        await asyncio.get_event_loop().run_in_executor(None, _delete_object)

    async def delete_objects(self, keys: list[str]) -> None:
        """Delete multiple objects from TOS with rolling bounded parallelism."""
        if not keys:
            return

        # TODO: If bytedtos adds a native ``delete_multi_objects`` API, replace
        # this rolling per-object scheduler with a backend-native bulk delete.
        # That would reduce request count while preserving the same public API.
        delete_concurrency = max(
            1,
            min(
                self.config.multi_object_delete_limit,
                len(keys),
            ),
        )

        async def _delete_key(key: str) -> None:
            try:
                await self.delete_object(key)
            except Exception as exc:
                raise ValueError(f"Failed to delete object {key}") from exc

        key_iter = iter(keys)
        pending_tasks: set[asyncio.Task[None]] = set()

        def _start_next_task() -> bool:
            try:
                key = next(key_iter)
            except StopIteration:
                return False

            pending_tasks.add(asyncio.create_task(_delete_key(key)))
            return True

        for _ in range(delete_concurrency):
            if not _start_next_task():
                break

        while pending_tasks:
            # Keep the pipeline full in a rolling fashion: as soon as one delete
            # completes, schedule the next key instead of waiting for a full
            # chunk/barrier to finish.
            done_tasks, still_pending = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            pending_tasks = set(still_pending)

            try:
                for task in done_tasks:
                    exc = task.exception()
                    if exc is not None:
                        raise exc
            except Exception:
                # Match BaseStorage2's fail-fast cleanup pattern: once any
                # delete fails, cancel sibling tasks and drain them so no
                # background task keeps running after we re-raise.
                for task in pending_tasks:
                    task.cancel()
                # Drain cancelled sibling tasks so they do not keep running or
                # surface unhandled exceptions after we re-raise the first error.
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                raise

            for _ in range(len(done_tasks)):
                if not _start_next_task():
                    break

    async def list_objects(
        self,
        prefix: str = "",
        delimiter: Optional[str] = None,
        limit: Optional[int] = None,
        continuation_token: Optional[str] = None,
        after_key: Optional[str] = None,
    ) -> tuple[list[str], Optional[str]]:
        """List objects with given prefix."""

        def _list_objects():
            try:
                if delimiter:
                    response = self.client.list_prefix(
                        prefix,
                        delimiter,
                        continuation_token or after_key or "",
                        min(limit, 1000) if limit is not None else 1000,
                    )
                    payload = self._response_payload(response)
                    objects, common_prefixes, is_truncated = (
                        self._payload_objects_and_prefixes(payload)
                    )
                    objects.extend(common_prefixes)
                    next_token = objects[-1] if is_truncated and objects else None
                else:
                    objects = self._collect_all_objects(prefix)
                    if continuation_token:
                        try:
                            offset = int(continuation_token)
                        except (TypeError, ValueError):
                            offset = 0
                    else:
                        offset = 0
                    if continuation_token is None and after_key:
                        try:
                            offset = objects.index(after_key) + 1
                        except ValueError:
                            offset = 0
                    if limit is not None:
                        paginated_objects = objects[offset : offset + limit]
                        next_token = (
                            str(offset + len(paginated_objects))
                            if offset + len(paginated_objects) < len(objects)
                            else None
                        )
                        objects = paginated_objects
                    else:
                        objects = objects[offset:]
                        next_token = None
            except TypeError:
                return self._list_objects_via_volcano_client(
                    prefix, delimiter, limit, continuation_token, after_key
                )
            except (TosException, TosClientError, TosServerError) as e:
                raise ValueError(f"Failed to list objects with prefix {prefix}: {e}")

            return objects, next_token

        return await asyncio.get_event_loop().run_in_executor(None, _list_objects)

    async def object_exists(self, key: str) -> bool:
        """Check if object exists in TOS."""

        def _head_object():
            try:
                self.client.head_object(key)
                return True
            except (TosException, TosClientError, TosServerError) as e:
                if self._is_not_found_error(e):
                    return False
                raise ValueError(f"Failed to check object existence {key}: {e}")

        return await asyncio.get_event_loop().run_in_executor(None, _head_object)

    async def get_object_size(self, key: str) -> int:
        """Get object size in bytes."""

        def _head_object():
            try:
                response = self.client.head_object(key)
                if self._is_manifest_object(response):
                    manifest_metadata = self._extract_all_metadata(response)
                    return int(manifest_metadata.get("aibrix-content-length", "0"))
                size = self._extract_content_length(response)
                if self._is_empty_object(response) and size > 0:
                    return 0
                return size
            except (TosException, TosClientError, TosServerError) as e:
                if self._is_not_found_error(e):
                    raise FileNotFoundError(f"Object not found: {key}")
                raise ValueError(f"Failed to get object size {key}: {e}")

        return await asyncio.get_event_loop().run_in_executor(None, _head_object)

    async def head_object(self, key: str) -> ObjectMetadata:
        """Get object metadata without downloading the object content."""

        def _head_object():
            try:
                response = self.client.head_object(key)
                manifest_metadata = self._extract_all_metadata(response)
                last_modified = self._extract_last_modified(response) or datetime.now()
                user_metadata = self._extract_metadata(response)
                etag = self._extract_etag(response)
                if self._is_manifest_object(response):
                    return ObjectMetadata(
                        content_length=int(
                            manifest_metadata.get("aibrix-content-length", "0")
                        ),
                        content_type=manifest_metadata.get("aibrix-content-type")
                        or self._extract_content_type(response, key),
                        etag=etag,
                        last_modified=last_modified,
                        metadata=user_metadata,
                        storage_class=self._extract_header(
                            response, "x-tos-storage-class", "storage-class"
                        )
                        or "STANDARD",
                        version_id=self._extract_header(
                            response, "x-tos-version-id", "version-id"
                        ),
                        encryption=self._extract_header(
                            response,
                            "x-tos-server-side-encryption",
                            "server-side-encryption",
                        ),
                        checksum=self._extract_header(
                            response, "x-tos-hash-crc64ecma", "X-Tos-Hash-Crc64ecma"
                        ),
                        cache_control=self._extract_header(response, "cache-control"),
                        content_disposition=self._extract_header(
                            response, "content-disposition"
                        ),
                        content_encoding=self._extract_header(
                            response, "content-encoding"
                        ),
                        content_language=self._extract_header(
                            response, "content-language"
                        ),
                        expires=self._parse_datetime(
                            self._extract_header(response, "expires")
                        ),
                    )

                return ObjectMetadata(
                    content_length=0
                    if self._is_empty_object(response)
                    else self._extract_content_length(response),
                    content_type=self._extract_content_type(response, key),
                    etag=etag,
                    last_modified=last_modified,
                    metadata=user_metadata,
                    storage_class=self._extract_header(
                        response, "x-tos-storage-class", "storage-class"
                    )
                    or "STANDARD",
                    version_id=self._extract_header(
                        response, "x-tos-version-id", "version-id"
                    ),
                    encryption=self._extract_header(
                        response,
                        "x-tos-server-side-encryption",
                        "server-side-encryption",
                    ),
                    checksum=self._extract_header(
                        response, "x-tos-hash-crc64ecma", "X-Tos-Hash-Crc64ecma"
                    ),
                    cache_control=self._extract_header(response, "cache-control"),
                    content_disposition=self._extract_header(
                        response, "content-disposition"
                    ),
                    content_encoding=self._extract_header(response, "content-encoding"),
                    content_language=self._extract_header(response, "content-language"),
                    expires=self._parse_datetime(
                        self._extract_header(response, "expires")
                    ),
                )

            except (TosException, TosClientError, TosServerError) as e:
                if self._is_not_found_error(e):
                    raise FileNotFoundError(f"Object not found: {key}")
                raise ValueError(f"Failed to get object metadata {key}: {e}")

        return await asyncio.get_event_loop().run_in_executor(None, _head_object)

    def is_native_multipart_supported(self) -> bool:
        """Check if native multipart upload is supported.

        Returns:
            True for TOS Storage
        """
        return True

    async def _native_create_multipart_upload(
        self,
        key: str,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """Create a multipart upload session."""

        def _create_multipart_upload():
            try:
                response = self.client.init_upload(
                    key, headers=self._build_headers(key, content_type, metadata)
                )
                return response.upload_id
            except (TosException, TosClientError, TosServerError) as e:
                raise ValueError(f"Failed to create multipart upload for {key}: {e}")

        return await asyncio.get_event_loop().run_in_executor(
            None, _create_multipart_upload
        )

    async def _native_upload_part(
        self,
        key: str,
        upload_id: str,
        part_number: int,
        data: Union[str, bytes, BinaryIO, TextIO, Reader],
    ) -> str:
        """Upload a part in a multipart upload."""

        # Unify all data types using Reader wrapper
        reader = self._wrap_data(data)

        def _upload_part():
            try:
                part_response = self.client.upload_part(
                    key,
                    upload_id,
                    part_number,
                    reader.read_all(),
                )
                etag = self._extract_etag(part_response)
                if etag:
                    return etag
                part_token = getattr(part_response, "part_number", "")
                if ":" in part_token:
                    return part_token.split(":", 1)[1]
                return str(part_token or part_number)
            except (TosException, TosClientError, TosServerError) as e:
                raise ValueError(f"Failed to upload part {part_number} for {key}: {e}")

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _upload_part)
        finally:
            # Close the reader if we created it
            if not isinstance(data, Reader):
                reader.close()

    async def _native_complete_multipart_upload(
        self,
        key: str,
        upload_id: str,
        parts: list[dict[str, Union[str, int]]],
    ) -> None:
        """Complete a multipart upload."""

        def _complete_multipart_upload():
            try:
                part_list = []
                for part in sorted(parts, key=lambda item: int(item["part_number"])):
                    part_number = str(part["part_number"])
                    etag = str(part["etag"])
                    if etag.startswith(f"{part_number}:"):
                        part_list.append(etag)
                    elif etag:
                        part_list.append(f"{part_number}:{etag}")
                    else:
                        part_list.append(part_number)
                self.client.complete_upload(
                    key,
                    upload_id,
                    part_list,
                )
            except (TosException, TosClientError, TosServerError) as e:
                raise ValueError(f"Failed to complete multipart upload for {key}: {e}")

        await asyncio.get_event_loop().run_in_executor(None, _complete_multipart_upload)

    async def _native_abort_multipart_upload(
        self,
        key: str,
        upload_id: str,
    ) -> None:
        """Abort a multipart upload."""

        def _abort_multipart_upload():
            try:
                self.client.abort_upload(key, upload_id)
            except (TosException, TosClientError, TosServerError) as e:
                raise ValueError(f"Failed to abort multipart upload for {key}: {e}")

        await asyncio.get_event_loop().run_in_executor(None, _abort_multipart_upload)

    async def copy_object(self, source_key: str, dest_key: str) -> None:
        """Copy an object within TOS."""

        def _copy_object():
            try:
                self.client.copy_object(source_key, dest_key)
            except (TosException, TosClientError, TosServerError) as e:
                if self._is_not_found_error(e):
                    raise FileNotFoundError(f"Source object not found: {source_key}")
                raise ValueError(
                    f"Failed to copy object {source_key} to {dest_key}: {e}"
                )

        await asyncio.get_event_loop().run_in_executor(None, _copy_object)

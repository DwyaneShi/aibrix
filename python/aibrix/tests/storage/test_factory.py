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

"""
Storage factory tests.

Tests the storage factory functionality for creating different storage types.
"""

import asyncio
import sys
import tempfile
import threading
import types
from pathlib import Path

import pytest

from aibrix.storage import (
    LocalStorage,
    StorageConfig,
    StorageListOrdering,
    StorageType,
    create_storage,
)
from aibrix.storage.base2 import BaseStorage2


class TestStorageFactory:
    """Test storage factory functionality."""

    def test_create_local_storage(self):
        """Test creating local storage through factory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = create_storage(StorageType.LOCAL, base_path=tmp_dir)

            assert isinstance(storage, LocalStorage)
            assert storage.base_path == Path(tmp_dir)

    def test_create_local_storage_with_string(self):
        """Test creating local storage with string type."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = create_storage("local", base_path=tmp_dir)

            assert isinstance(storage, LocalStorage)
            assert storage.base_path == Path(tmp_dir)

    def test_create_storage_with_config(self):
        """Test creating storage with custom configuration."""
        config = StorageConfig(
            multipart_threshold=1024 * 1024,  # 1MB
            max_concurrency=7,
            max_session_concurrency=5,
            multi_object_delete_limit=123,
            range_chunksize=512 * 1024,  # 512KB
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = create_storage(
                StorageType.LOCAL, config=config, base_path=tmp_dir
            )

            assert isinstance(storage, LocalStorage)
            assert storage.config.multipart_threshold == 1024 * 1024
            assert storage.config.max_concurrency == 7
            assert storage.config.max_session_concurrency == 5
            assert storage.config.multi_object_delete_limit == 123
            assert storage.config.range_chunksize == 512 * 1024

    def test_create_storage_with_list_ordering_in_config(self):
        """Test selecting the active list ordering through StorageConfig."""
        config = StorageConfig(list_ordering=StorageListOrdering.CREATED_AT_DESC)

        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = create_storage(
                StorageType.LOCAL, config=config, base_path=tmp_dir
            )

            assert isinstance(storage, LocalStorage)
            assert storage.get_list_ordering() == StorageListOrdering.CREATED_AT_DESC

    def test_create_storage_uses_retry_config_from_env(self, monkeypatch):
        monkeypatch.setenv("STORAGE_MAX_RETRIES", "7")

        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = create_storage(StorageType.LOCAL, base_path=tmp_dir)

        assert storage.config.max_retries == 7

    def test_create_storage_rejects_negative_retry_config(self, monkeypatch):
        monkeypatch.setenv("STORAGE_MAX_RETRIES", "-1")

        with pytest.raises(ValueError, match="STORAGE_MAX_RETRIES must be >= 0"):
            create_storage(StorageType.LOCAL)

    def test_create_s3_storage_missing_bucket(self):
        """Test that S3 storage creation fails without bucket name."""
        with pytest.raises(ValueError, match="bucket_name is required"):
            create_storage(StorageType.S3)

    @pytest.mark.skip(reason="S3 accessibility check can fail on local SSL setup")
    def test_create_s3_storage_with_params(self):
        """Test creating S3 storage with parameters."""
        # This will fail due to invalid credentials, but tests parameter passing
        with pytest.raises(ValueError, match="not accessible"):
            create_storage(
                StorageType.S3,
                bucket_name="test-bucket",
                region_name="us-east-1",
                aws_access_key_id="fake-key",
                aws_secret_access_key="fake-secret",
            )

    def test_create_tos_storage_missing_params(self):
        """Test that TOS storage creation fails without required parameters."""
        with pytest.raises(ValueError, match="bucket_name is required"):
            create_storage(StorageType.TOS)

        with pytest.raises(ValueError, match="access_key is required"):
            create_storage(StorageType.TOS, bucket_name="test-bucket")

        with pytest.raises(ValueError, match="secret_key is required"):
            create_storage(StorageType.TOS, bucket_name="test-bucket", access_key="key")

        with pytest.raises(ValueError, match="endpoint is required"):
            create_storage(
                StorageType.TOS,
                bucket_name="test-bucket",
                access_key="key",
                secret_key="secret",
                region="cn-beijing",
            )

        with pytest.raises(ValueError, match="region is required"):
            create_storage(
                StorageType.TOS,
                bucket_name="test-bucket",
                access_key="key",
                secret_key="secret",
                endpoint="http://example.com",
            )

    def _install_fake_bytedtos(self, monkeypatch):
        calls = {}

        class StaticCredentials:
            def __init__(self, access_key: str, secret_key: str):
                self.access_key = access_key
                self.secret_key = secret_key

        class ClientV2:
            def __init__(self, bucket, cred=None, **kwargs):
                calls["bucket"] = bucket
                calls["cred"] = cred
                calls["kwargs"] = kwargs

        fake_bytedtos = types.ModuleType("bytedtos")
        fake_bytedtos.ClientV2 = ClientV2
        fake_bytedtos.StaticCredentials = StaticCredentials
        fake_bytedtos_errors = types.ModuleType("bytedtos.errors")
        fake_bytedtos_errors.TosException = Exception
        fake_tos = types.ModuleType("tos")
        fake_tos_exceptions = types.ModuleType("tos.exceptions")
        fake_tos_exceptions.TosClientError = Exception
        fake_tos_exceptions.TosServerError = Exception

        fake_tos.exceptions = fake_tos_exceptions
        monkeypatch.setitem(sys.modules, "tos", fake_tos)
        monkeypatch.setitem(sys.modules, "bytedtos", fake_bytedtos)
        monkeypatch.setitem(sys.modules, "bytedtos.errors", fake_bytedtos_errors)
        monkeypatch.setitem(sys.modules, "tos.exceptions", fake_tos_exceptions)
        monkeypatch.delitem(sys.modules, "aibrix.storage.bytetos", raising=False)
        return calls

    @pytest.mark.parametrize("idc", ["mya", "maliva", "cn-beijing", "us-east-1"])
    def test_create_tos_storage_with_psm_client(self, monkeypatch, idc):
        """TOS PSM storage passes PSM params to bytedtos.ClientV2."""
        calls = self._install_fake_bytedtos(monkeypatch)

        storage = create_storage(
            StorageType.TOS,
            bucket_name="test-bucket",
            access_key="ak",
            secret_key="sk",
            endpoint="https://tos-cn-beijing.volces.com",
            region="cn-beijing",
            idc=idc,
            service="toutiao.tos.tosapi",
            cluster="default",
            remote_psm="inf.aibrix.metadata",
        )

        assert storage.bucket_name == "test-bucket"
        assert calls["bucket"] == "test-bucket"
        assert calls["cred"].access_key == "ak"
        assert calls["cred"].secret_key == "sk"
        assert calls["kwargs"] == {
            "service": "toutiao.tos.tosapi",
            "cluster": "default",
            "idc": idc,
            "remote_psm": "inf.aibrix.metadata",
        }

    def test_create_tos_storage_with_force_volcano_preserves_ve_kwargs(
        self, monkeypatch
    ):
        """force_volcano keeps the existing Volcano kwargs path."""
        calls = self._install_fake_bytedtos(monkeypatch)

        storage = create_storage(
            StorageType.TOS,
            bucket_name="test-bucket",
            access_key="ak",
            secret_key="sk",
            endpoint="https://tos-cn-beijing.volces.com",
            region="cn-beijing",
            force_volcano=True,
            enable_crc=True,
        )

        assert storage.bucket_name == "test-bucket"
        assert calls["bucket"] == "test-bucket"
        assert calls["cred"].access_key == "ak"
        assert calls["cred"].secret_key == "sk"
        assert calls["kwargs"]["enable_crc64"] is True
        assert calls["kwargs"]["connection_pool_size"] >= 1
        assert calls["kwargs"]["force_volcano"] is True
        assert calls["kwargs"]["ve_cred"] is calls["cred"]
        assert calls["kwargs"]["ve_region"] == "cn-beijing"
        assert calls["kwargs"]["ve_endpoint"] == "https://tos-cn-beijing.volces.com"

    def test_bytetos_uses_shared_strict_multipart_aggregation(self, monkeypatch):
        self._install_fake_bytedtos(monkeypatch)
        storage = create_storage(
            StorageType.TOS,
            config=StorageConfig(multipart_threshold=8),
            bucket_name="test-bucket",
            access_key="ak",
            secret_key="sk",
            idc="mya",
            service="toutiao.tos.tosapi",
            cluster="default",
            remote_psm="inf.aibrix.metadata",
        )
        multipart_calls = []

        async def multipart_upload(
            key,
            data,
            content_type=None,
            metadata=None,
            byline=0,
            bysize=0,
            parts=1,
        ):
            multipart_calls.append(
                {
                    "key": key,
                    "data": data,
                    "content_type": content_type,
                    "metadata": metadata,
                    "byline": byline,
                    "bysize": bysize,
                    "parts": parts,
                }
            )

        storage.multipart_upload = multipart_upload

        assert asyncio.run(storage.put_object("key", b"12345678")) is True
        assert isinstance(storage, BaseStorage2)
        assert storage.config.strict_multipart_min_part_size is True
        assert len(multipart_calls) == 1
        assert multipart_calls[0]["bysize"] == 7

    def test_tos_object_exists_retries_transient_head_error(self, monkeypatch):
        self._install_fake_bytedtos(monkeypatch)
        storage = create_storage(
            StorageType.TOS,
            config=StorageConfig(max_retries=2),
            bucket_name="test-bucket",
            access_key="ak",
            secret_key="sk",
            idc="mya",
            service="toutiao.tos.tosapi",
            cluster="default",
            remote_psm="inf.aibrix.metadata",
        )

        import aibrix.storage.bytetos as bytetos_mod

        monkeypatch.setattr(bytetos_mod.time, "sleep", lambda _: None)
        calls = []

        class HeadObjectError(Exception):
            status_code = "408"

        def head_object(key):
            calls.append(key)
            if len(calls) == 1:
                raise HeadObjectError("408::request timeout")
            return object()

        storage.client.head_object = head_object

        assert asyncio.run(storage.object_exists(".multipart/upload/metadata")) is True
        assert calls == [".multipart/upload/metadata", ".multipart/upload/metadata"]

    @pytest.mark.parametrize(
        ("attributes", "message", "expected"),
        [
            ({"status_code": "408"}, "", True),
            ({"code": "4038"}, "", True),
            ({"code": "408"}, "", False),
            ({"status_code": "4038"}, "", False),
            ({}, "request timed out", False),
        ],
    )
    def test_tos_retry_codes_are_checked_separately(
        self, monkeypatch, attributes, message, expected
    ):
        self._install_fake_bytedtos(monkeypatch)
        storage = create_storage(
            StorageType.TOS,
            bucket_name="test-bucket",
            access_key="ak",
            secret_key="sk",
            idc="mya",
            service="toutiao.tos.tosapi",
            cluster="default",
            remote_psm="inf.aibrix.metadata",
        )
        error = RuntimeError(message)
        for name, value in attributes.items():
            setattr(error, name, value)

        assert storage._is_retryable_tos_error(error) is expected

    def test_tos_object_exists_does_not_retry_status_in_object_name(self, monkeypatch):
        self._install_fake_bytedtos(monkeypatch)
        storage = create_storage(
            StorageType.TOS,
            config=StorageConfig(max_retries=2),
            bucket_name="test-bucket",
            access_key="ak",
            secret_key="sk",
            idc="mya",
            service="toutiao.tos.tosapi",
            cluster="default",
            remote_psm="inf.aibrix.metadata",
        )
        calls = []

        class HeadObjectError(Exception):
            pass

        def head_object(key):
            calls.append(key)
            raise HeadObjectError(f"permission denied for {key}")

        storage.client.head_object = head_object
        object_key = "image_500_tos.jsonl"

        with pytest.raises(ValueError, match="permission denied"):
            asyncio.run(storage.object_exists(object_key))
        assert calls == [object_key]

    def test_tos_object_exists_preserves_not_found_semantics(self, monkeypatch):
        self._install_fake_bytedtos(monkeypatch)
        storage = create_storage(
            StorageType.TOS,
            config=StorageConfig(max_retries=2),
            bucket_name="test-bucket",
            access_key="ak",
            secret_key="sk",
            idc="mya",
            service="toutiao.tos.tosapi",
            cluster="default",
            remote_psm="inf.aibrix.metadata",
        )

        class HeadObjectError(Exception):
            code = "404"

        storage.client.head_object = lambda _: (_ for _ in ()).throw(
            HeadObjectError("404::not found")
        )

        assert asyncio.run(storage.object_exists(".multipart/upload/metadata")) is False

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("text", b"text"),
            (bytearray(b"bytearray"), b"bytearray"),
            (memoryview(b"memoryview"), b"memoryview"),
        ],
    )
    def test_tos_response_data_accepts_bytes_like_values(
        self, monkeypatch, payload, expected
    ):
        self._install_fake_bytedtos(monkeypatch)
        storage = create_storage(
            StorageType.TOS,
            bucket_name="test-bucket",
            access_key="ak",
            secret_key="sk",
            idc="mya",
            service="toutiao.tos.tosapi",
            cluster="default",
            remote_psm="inf.aibrix.metadata",
        )

        response = types.SimpleNamespace(data=payload)

        assert storage._response_data(response) == expected

    def test_tos_response_data_encodes_text_from_reader(self, monkeypatch):
        self._install_fake_bytedtos(monkeypatch)
        storage = create_storage(
            StorageType.TOS,
            bucket_name="test-bucket",
            access_key="ak",
            secret_key="sk",
            idc="mya",
            service="toutiao.tos.tosapi",
            cluster="default",
            remote_psm="inf.aibrix.metadata",
        )

        response = types.SimpleNamespace(
            data=types.SimpleNamespace(read=lambda: "text")
        )

        assert storage._response_data(response) == b"text"

    def test_tos_upload_part_reads_once_in_executor_across_retries(self, monkeypatch):
        self._install_fake_bytedtos(monkeypatch)
        storage = create_storage(
            StorageType.TOS,
            config=StorageConfig(max_retries=1),
            bucket_name="test-bucket",
            access_key="ak",
            secret_key="sk",
            idc="mya",
            service="toutiao.tos.tosapi",
            cluster="default",
            remote_psm="inf.aibrix.metadata",
        )

        import aibrix.storage.bytetos as bytetos_mod

        monkeypatch.setattr(bytetos_mod.time, "sleep", lambda _: None)
        caller_thread = threading.get_ident()
        read_threads = []
        upload_calls = []

        class RetryableUploadError(Exception):
            status_code = "408"

        class Stream:
            def read(self, size=-1):
                del size
                read_threads.append(threading.get_ident())
                return b"payload"

        def upload_part(key, upload_id, part_number, payload):
            upload_calls.append(
                (key, upload_id, part_number, payload, threading.get_ident())
            )
            if len(upload_calls) == 1:
                raise RetryableUploadError("request timed out")
            return types.SimpleNamespace(headers={"ETag": "part-etag"})

        storage.client.upload_part = upload_part

        etag = asyncio.run(storage._native_upload_part("key", "upload-id", 1, Stream()))

        assert etag == "part-etag"
        assert len(read_threads) == 1
        assert read_threads[0] != caller_thread
        assert [call[:4] for call in upload_calls] == [
            ("key", "upload-id", 1, b"payload"),
            ("key", "upload-id", 1, b"payload"),
        ]
        assert all(call[4] == read_threads[0] for call in upload_calls)

    def test_unsupported_storage_type(self):
        """Test error handling for unsupported storage types."""
        with pytest.raises(ValueError, match="Unsupported storage type"):
            create_storage("unsupported")

        with pytest.raises(ValueError, match="Unsupported storage type"):
            create_storage("invalid_type")

    def test_case_insensitive_storage_type(self):
        """Test that storage type strings are case insensitive."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test various case combinations
            for type_str in ["LOCAL", "local", "Local", "LOCAL"]:
                storage = create_storage(type_str, base_path=tmp_dir)
                assert isinstance(storage, LocalStorage)

    def test_storage_type_enum_vs_string(self):
        """Test that enum and string types produce equivalent results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_enum = create_storage(StorageType.LOCAL, base_path=tmp_dir)
            storage_string = create_storage("local", base_path=tmp_dir)

            assert type(storage_enum) is type(storage_string)
            assert storage_enum.base_path == storage_string.base_path

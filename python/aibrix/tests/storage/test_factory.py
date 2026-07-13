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

import sys
import tempfile
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

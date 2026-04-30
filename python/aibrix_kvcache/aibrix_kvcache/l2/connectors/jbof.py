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

import hashlib
import time
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import List

import kv_client
import torch

from ... import envs
from ...common import AsyncBase
from ...common.absl_logging import getLogger
from ...memory import MemoryRegion
from ...status import Status, StatusCodes
from ...utils import round_up
from . import Connector, ConnectorFeature

logger = getLogger(__name__)


JBOF_VALUE_SIZE_ALIGNMENT = 4096
JBOF_IOV_CHUNK_SIZE = 1048576
JBOF_PUT_NUM_RETRIES = 6


@dataclass
class JBOFConfig:
    """JBOF config"""

    kv_addr: str
    kv_nqn: int
    kv_cores: int
    block_nbytes: int
    use_iov_api: bool


@AsyncBase.async_wrap(
    exists="_exists", get="_get", put="_put", delete="_delete"
)
class JBOFConnector(Connector[bytes, torch.Tensor], AsyncBase):
    """JBOF (JBOF) connector"""

    def __init__(
        self,
        config: JBOFConfig,
        key_suffix: str,
        executor: Executor,
    ):
        super().__init__(executor)
        self.config = config
        self.key_suffix = key_suffix
        self.value_size = round_up(config.block_nbytes, JBOF_VALUE_SIZE_ALIGNMENT)
        self.rt = None

    @classmethod
    def from_envs(
        cls, conn_id: str, executor: Executor, **kwargs
    ) -> "JBOFConnector":
        """Create a connector from environment variables."""

        assert "block_nbytes" in kwargs
        config = JBOFConfig(
            kv_addr=envs.AIBRIX_KV_CACHE_OL_JBOF_KV_ADDR,
            kv_nqn=envs.AIBRIX_KV_CACHE_OL_JBOF_KV_NQN,
            kv_cores=envs.AIBRIX_KV_CACHE_OL_JBOF_KV_CORES,
            block_nbytes=kwargs["block_nbytes"],
            use_iov_api=envs.AIBRIX_KV_CACHE_OL_JBOF_USE_IOV_API,
        )
        return cls(config, conn_id, executor)

    @property
    def name(self) -> str:
        return "JBOF"

    @property
    def feature(self) -> ConnectorFeature:
        feature = ConnectorFeature(
            rdma=True,  # trigger mr registration
        )
        return feature

    def __del__(self) -> None:
        self.close()

    def _key(self, key: bytes) -> bytes:
        # JBOF supports max 16-byte keys.
        # MD5 produces exactly 16 bytes (128 bits).
        jbof_key = key.hex() + self.key_suffix
        return hashlib.md5(jbof_key.encode('utf-8')).digest()

    @Status.capture_exception
    def open(self) -> Status:
        self.rt = kv_client.spdk_initiator_start(
            self.config.kv_addr,
            self.config.kv_nqn,
            hex(self.config.kv_cores), # Pass core mask as hex string, e.g. "0xff"
            self.value_size,
        )
        assert self.rt is not None
        return Status.ok()

    @Status.capture_exception
    def close(self) -> Status:
        if self.rt is not None and hasattr(kv_client, "spdk_initiator_stop"):
            kv_client.spdk_initiator_stop(self.rt)
        self.rt = None
        return Status.ok()

    @Status.capture_exception
    def register_slabs(self, slabs: List[torch.Tensor]) -> Status:
        for slab in slabs:
            addr = slab.data_ptr()
            length = slab.numel() * slab.itemsize
            kv_client.mem_register(addr, length)
            logger.info(f"Registered SPDK DMA memory for slab: ptr={hex(addr)}, size={length}")
        return Status.ok()

    @Status.capture_exception
    def _exists(self, key: bytes) -> Status:
        """Check if key is in the store."""
        assert self.rt is not None
        if kv_client.exist(self.rt, self._key(key)):
            return Status.ok()
        return Status(StatusCodes.NOT_FOUND)

    @Status.capture_exception
    def _get(self, key: bytes, mr: MemoryRegion) -> Status:
        """Get a value."""
        assert self.rt is not None
        if self.config.use_iov_api:
            tensor = mr.to_tensor(torch.uint8)
            arr = tensor.numpy()
            mv = memoryview(arr)
            total = self.value_size
            if len(mv) < total:
                val = kv_client.get(self.rt, self._key(key), self.value_size)
                if val is None:
                    return Status(StatusCodes.NOT_FOUND)
                mr.fill(val[: self.config.block_nbytes])
                return Status.ok()

            bufs = [mv[i : i + JBOF_IOV_CHUNK_SIZE] for i in range(0, total, JBOF_IOV_CHUNK_SIZE)]
            ok = kv_client.getv_into(self.rt, self._key(key), bufs)
            if ok:
                return Status.ok()

            return Status(StatusCodes.NOT_FOUND)
        else:
            val = kv_client.get(self.rt, self._key(key), self.value_size)
            if val is None:
                return Status(StatusCodes.NOT_FOUND)
            # ignore padding bytes
            mr.fill(val[:self.config.block_nbytes])
            return Status.ok()

    @Status.capture_exception
    def _put(self, key: bytes, mr: MemoryRegion) -> Status:
        """Put a key value pair"""
        assert self.rt is not None
        if self.config.use_iov_api:
            tensor = mr.to_tensor(torch.uint8)
            arr = tensor.numpy()
            mv = memoryview(arr)
            total = self.value_size
            if len(mv) < total:
                value = mr.tobytes()
                kv_client.put(self.rt, self._key(key), value)
                return Status.ok()

            bufs = [mv[i : i + JBOF_IOV_CHUNK_SIZE] for i in range(0, total, JBOF_IOV_CHUNK_SIZE)]
            last_exc: Exception | None = None
            for attempt in range(JBOF_PUT_NUM_RETRIES):
                try:
                    kv_client.putv(self.rt, self._key(key), bufs)
                    return Status.ok()
                except Exception as e:
                    last_exc = e
                    logger.error(f"putv exception: {e}")
                    time.sleep(0.01 * (attempt + 1))

            assert last_exc is not None
            raise last_exc
        else:
            value = mr.tobytes()
            kv_client.put(self.rt, self._key(key), value)
            return Status.ok()

    @Status.capture_exception
    def _delete(self, key: bytes) -> Status:
        """Delete a key."""
        assert self.rt is not None
        kv_client.delete(self.rt, self._key(key))
        return Status.ok()

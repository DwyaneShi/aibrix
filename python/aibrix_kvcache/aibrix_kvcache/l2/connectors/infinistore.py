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

import functools
from concurrent.futures import Executor
from typing import List, Sequence, Tuple

import infinistore
import torch

from ... import envs
from ...common import AsyncBase
from ...memory import MemoryRegion
from ...status import Status, StatusCodes
from . import Connector, ConnectorFeature, ConnectorRegisterDescriptor


@AsyncBase.async_wrap(delete="_delete")
class InfiniStoreConnector(Connector[bytes, torch.Tensor], AsyncBase):
    """InfiniStore connector."""

    def __init__(
        self,
        config: infinistore.ClientConfig,
        key_suffix: str,
        executor: Executor,
    ):
        super().__init__(executor)
        self.config = config
        self.key_suffix = key_suffix
        self.conn: infinistore.InfinityConnection | None = None

    @classmethod
    def from_envs(
        cls, conn_id: str, executor: Executor
    ) -> "InfiniStoreConnector":
        """Create a connector from environment variables."""
        config = infinistore.ClientConfig(
            host_addr=envs.AIBRIX_KV_CACHE_OL_INFINISTORE_HOST_ADDR,
            service_port=envs.AIBRIX_KV_CACHE_OL_INFINISTORE_SERVICE_PORT,
            connection_type=envs.AIBRIX_KV_CACHE_OL_INFINISTORE_CONNECTION_TYPE,
            ib_port=envs.AIBRIX_KV_CACHE_OL_INFINISTORE_IB_PORT,
            link_type=envs.AIBRIX_KV_CACHE_OL_INFINISTORE_LINK_TYPE,
            dev_name=envs.AIBRIX_KV_CACHE_OL_INFINISTORE_DEV_NAME,
        )
        return cls(config, conn_id, executor)

    @property
    def name(self) -> str:
        return "InfiniStore"

    @property
    def feature(self) -> ConnectorFeature:
        feature = ConnectorFeature()
        if (
            self.config is not None
            and self.config.connection_type == infinistore.TYPE_RDMA
        ):
            # InfiniStore has a 4MB size limit
            # feature.mput_mget = True
            feature.rdma = True
        return feature

    def _key(self, key: bytes) -> str:
        return key.hex() + self.key_suffix

    @Status.capture_exception
    def open(self) -> Status:
        """Open a connection."""
        if self.conn is None:
            self.conn = infinistore.InfinityConnection(self.config)
            self.conn.connect()
        return Status.ok()

    @Status.capture_exception
    def close(self) -> Status:
        """Close a connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        return Status.ok()

    @Status.capture_exception
    def register_mr(
        self, addr: int, length: int
    ) -> Status[ConnectorRegisterDescriptor]:
        assert self.conn is not None
        ret = self.conn.register_mr(addr, length)
        if ret != 0:
            return Status(StatusCodes.INVALID)
        return Status.ok(ConnectorRegisterDescriptor())

    @Status.capture_exception
    def deregister_mr(self, desc: ConnectorRegisterDescriptor) -> Status:
        # InfiniStore does not expose deregister function
        return Status.ok()

    @Status.capture_exception
    async def exists(self, key: bytes) -> Status:
        """Check if key is in the store."""
        assert self.conn is not None
        if self.conn.check_exist(self._key(key)):
            return Status.ok()
        return Status(StatusCodes.NOT_FOUND)

    def get_batches(
        self,
        keys: Sequence[bytes],
        mrs: Sequence[MemoryRegion],
        batch_size: int,
    ) -> Sequence[Sequence[Tuple[bytes, MemoryRegion]]]:
        lists: List[List[Tuple[bytes, MemoryRegion]]] = []
        for key, mr in zip(keys, mrs):
            if (
                len(lists) == 0
                or lists[-1][0][1].data_ptr() != mr.slab.data_ptr()
                or len(lists[-1]) >= batch_size
            ):
                lists.append([(key, mr)])
            else:
                lists[-1].append((key, mr))
        return lists

    @Status.capture_exception
    async def mget(
        self, keys: Sequence[bytes], mrs: Sequence[MemoryRegion]
    ) -> Sequence[Status]:
        assert self.conn is not None
        base_addr = mrs[0].slab.data_ptr()
        block_size = mrs[0].length
        blocks = [None] * len(mrs)
        for i, mr in enumerate(mrs):
            blocks[i] = (self._key(keys[i]), mr.addr)  # type: ignore

        try:
            await self.conn.rdma_read_cache_async(blocks, block_size, base_addr)
        except infinistore.InfiniStoreKeyNotFound:
            return [Status(StatusCodes.NOT_FOUND)] * len(mrs)
        return [Status.ok()] * len(mrs)

    @Status.capture_exception
    async def mput(
        self, keys: Sequence[bytes], mrs: Sequence[MemoryRegion]
    ) -> Sequence[Status]:
        assert self.conn is not None
        base_addr = mrs[0].slab.data_ptr()
        block_size = mrs[0].length
        blocks = [None] * len(mrs)
        for i, mr in enumerate(mrs):
            blocks[i] = (self._key(keys[i]), mr.addr)  # type: ignore

        await self.conn.rdma_write_cache_async(blocks, block_size, base_addr)
        return [Status.ok()] * len(mrs)

    @Status.capture_exception
    async def get(self, key: bytes, mr: MemoryRegion) -> Status[torch.Tensor]:
        """Get a value."""
        if self.config.connection_type == infinistore.TYPE_RDMA:
            return await self._rdma_get(key, mr)
        else:
            tcp_get = functools.partial(self._tcp_get, key, mr)
            return await self.event_loop.run_in_executor(
                self._executor, tcp_get
            )

    def _tcp_get(self, key: bytes, mr: MemoryRegion) -> Status:
        """Get a value via TCP."""
        assert self.conn is not None
        val = self.conn.tcp_read_cache(self._key(key))
        if val is None or len(val) == 0:
            return Status(StatusCodes.NOT_FOUND)
        mr.fill(val)
        return Status.ok()

    async def _rdma_get(self, key: bytes, mr: MemoryRegion) -> Status:
        """Get a value via RDMA."""
        assert self.conn is not None
        try:
            await self.conn.rdma_read_cache_async(
                [(self._key(key), mr.addr)], mr.length, mr.slab.data_ptr()
            )
        except infinistore.InfiniStoreKeyNotFound:
            return Status(StatusCodes.NOT_FOUND)
        return Status.ok()

    @Status.capture_exception
    async def put(self, key: bytes, mr: MemoryRegion) -> Status:
        """Put a key value pair"""
        if self.config.connection_type == infinistore.TYPE_RDMA:
            return await self._rdma_put(key, mr)
        else:
            tcp_put = functools.partial(self._tcp_put, key, mr)
            return await self.event_loop.run_in_executor(
                self._executor, tcp_put
            )

    async def _rdma_put(self, key: bytes, mr: MemoryRegion) -> Status:
        """Put a value via RDMA."""
        assert self.conn is not None
        await self.conn.rdma_write_cache_async(
            [(self._key(key), mr.addr)], mr.length, mr.slab.data_ptr()
        )
        return Status.ok()

    def _tcp_put(self, key: bytes, mr: MemoryRegion) -> Status:
        """Put a value via TCP."""
        assert self.conn is not None
        self.conn.tcp_write_cache(self._key(key), mr.data_ptr(), mr.length)
        return Status.ok()

    @Status.capture_exception
    def _delete(self, key: bytes) -> Status:
        """Delete a key."""
        assert self.conn is not None
        self.conn.delete_keys(self._key(key))
        return Status.ok()

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

from dataclasses import dataclass
from threading import Lock
from typing import List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from .. import envs
from ..common.absl_logging import getLogger
from ..status import Status, StatusCodes
from ..utils import round_up
from .ref_counted_obj import RefCountedObj

logger = getLogger(__name__)

try:
    import aibrix_kvcache._cpu_C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from aibrix_kvcache._cpu_C with %r", e)

MR_USE_COMPACT_LAYOUT = not envs.AIBRIX_KV_CACHE_OL_TOKEN_VALIDATION_ENABLED


@dataclass
class MemoryRegionFooter:
    prefix_length: int
    tokens_length: int

    def __init__(self, prefix_length: int, tokens_length: int):
        self.prefix_length = prefix_length
        self.tokens_length = tokens_length
        self._storage = np.array(
            [self.prefix_length, self.tokens_length], dtype=np.int32
        )

    def __post_init__(self):
        if self.prefix_length < 0:
            raise ValueError("prefix_length must be non-negative")
        if self.tokens_length < 0:
            raise ValueError("tokens_length must be non-negative")

    def to_numpy(self) -> np.ndarray:
        return self._storage

    @staticmethod
    def from_numpy(storage: np.ndarray) -> "MemoryRegionFooter":
        return MemoryRegionFooter(
            prefix_length=int(storage[0]), tokens_length=int(storage[1])
        )

    @staticmethod
    def nbytes() -> int:
        return np.dtype(np.int32).itemsize * 2


class MemoryRegion(RefCountedObj):
    """A memory region representation used by Allocator.
    Layout: [cache block, magic, footer, tokens]
    """

    MAGIC: int = 0x3A7F1C42

    def __init__(
        self,
        allocator: "TensorPoolAllocator",
        slab: torch.Tensor,
        addr: int,
        len: int,
    ) -> None:
        super().__init__()
        assert allocator is not None
        self.allocator = allocator
        self.slab = slab
        self.addr = addr
        self.capacity = len
        self.length = self.capacity
        self._init_meta()

    def _init_meta(self) -> None:
        self._block_nbytes = -1
        self._is_sealed = False
        self._prefix: Tuple[int, ...] | None = None
        self._tokens: Tuple[int, ...] = tuple()

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return (
            f"MemoryRegion(addr={self.slab.data_ptr() + self.addr},"
            f" length={self.length}, capacity={self.capacity},"
            f" ref={self.ref_count}, sealed={self._is_sealed})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def block_nbytes(self) -> int:
        """Get the size of the cache block in bytes."""
        assert self._block_nbytes > 0, "block_nbytes must be set"
        return self._block_nbytes

    @block_nbytes.setter
    def block_nbytes(self, block_nbytes: int) -> None:
        """Set the size of the cache block in bytes."""
        assert block_nbytes > 0, "block_nbytes must be positive"
        self._block_nbytes = block_nbytes

    @property
    def is_sealed(self) -> bool:
        """Check if the MR is sealed."""
        return self._is_sealed

    def seal(self) -> None:
        if self._is_sealed:
            return

        if not MR_USE_COMPACT_LAYOUT:
            bytes_per_int = np.dtype(np.int32).itemsize
            start = self.addr + self.block_nbytes
            stop = start + bytes_per_int
            magic = self.slab[start:stop].view(torch.int32).numpy()[0]

            assert magic == MemoryRegion.MAGIC, (
                "Magic mismatch, MUST pack tokens before sealing."
            )

            start = stop
            stop = start + MemoryRegionFooter.nbytes()
            footer = MemoryRegionFooter.from_numpy(
                self.slab[start:stop].view(torch.int32).numpy()
            )
            ntokens = footer.prefix_length + footer.tokens_length
            actual_length = self.calculate_size(self.block_nbytes, ntokens)
            assert actual_length <= self.length, (
                f"{actual_length} > {self.length}"
            )

            self.length = actual_length

        self._is_sealed = True

    def fill(self, data: bytes) -> None:
        assert len(data) == self.length
        self.slab[self.addr : self.addr + len(data)].copy_(
            torch.frombuffer(data, dtype=torch.uint8)
        )

    def tobytes(self) -> bytes:
        tensor = self.slab[self.addr : self.addr + self.length]
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy().tobytes()

    def data_ptr(self) -> int:
        return self.slab.data_ptr() + self.addr

    def destroy_unsafe(self):
        self._init_meta()
        self.allocator._finalize_mr(self.slab, self.addr, self.capacity)

    def to_tensor(
        self,
        mr_dtype: torch.dtype | None = None,
        mr_shape: Tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Convert MR to tensor"""
        ret = self.slab[self.addr : self.addr + self.block_nbytes]
        if mr_dtype is not None:
            ret = ret.view(mr_dtype)
        if mr_shape is not None:
            ret = ret.view(*mr_shape)
        return ret

    @staticmethod
    def to_tensors(
        mrs: Sequence["MemoryRegion"],
        mr_dtype: torch.dtype | None = None,
        mr_shape: Tuple[int, ...] | None = None,
    ) -> Sequence[torch.Tensor]:
        """Convert MRs to tensors. Contiguous MRs are supposed to form
        a single tensor.
        """
        if mrs is None or len(mrs) == 0:
            return []

        return [mr.to_tensor(mr_dtype, mr_shape) for mr in mrs]

    def pack_tokens(
        self,
        *,
        tokens: Tuple[int, ...],
        prefix: Tuple[int, ...] | None = None,
    ) -> None:
        """Pack tokens into the MR.
        Args:
            prefix: The prefix tokens.
            tokens: The tokens to be set.
        """
        ntokens = len(tokens)
        assert ntokens > 0, "tokens must not be empty"

        if MR_USE_COMPACT_LAYOUT:
            self._prefix = prefix
            self._tokens = tokens
            return

        bytes_per_token = np.dtype(np.int32).itemsize

        ntokens_limit = (
            self.length - self.block_nbytes - MemoryRegionFooter.nbytes()
        ) // bytes_per_token - 1
        assert ntokens <= ntokens_limit, (
            f"tokens ({ntokens}) must not exceed the limit ({ntokens_limit})"
        )

        self._prefix = prefix
        self._tokens = tokens

        # Write magic
        start = self.addr + self.block_nbytes
        stop = start + bytes_per_token
        self.slab[start:stop].copy_(
            torch.from_numpy(
                np.array([MemoryRegion.MAGIC], dtype=np.int32)
            ).view(torch.uint8)
        )
        # Write footer
        prefix_length = len(prefix) if prefix is not None else 0
        footer = MemoryRegionFooter(prefix_length, len(tokens))
        start = stop
        stop = start + MemoryRegionFooter.nbytes()
        self.slab[start:stop].copy_(
            torch.from_numpy(footer.to_numpy()).view(torch.uint8)
        )
        # Pack tokens
        all = np.array((prefix or tuple()) + tokens, dtype=np.int32)
        start = stop
        stop = start + bytes_per_token * len(all)
        self.slab[start:stop].copy_(torch.from_numpy(all).view(torch.uint8))

    def unpack_tokens(self) -> Tuple[Tuple[int, ...] | None, Tuple[int, ...]]:
        """Unpack tokens from the MR.
        Returns:
            The prefix and tokens.
        """
        if len(self._tokens) > 0 or MR_USE_COMPACT_LAYOUT:
            return self._prefix, self._tokens

        bytes_per_token = np.dtype(np.int32).itemsize
        start = self.addr + self.block_nbytes
        stop = start + bytes_per_token
        magic = self.slab[start:stop].view(torch.int32).numpy()[0]

        if magic != MemoryRegion.MAGIC:
            # corrupted mr or current mr is not packed with tokens
            return None, tuple()

        start = stop
        stop = start + MemoryRegionFooter.nbytes()
        footer = MemoryRegionFooter.from_numpy(
            self.slab[start:stop].view(torch.int32).numpy()
        )

        if footer.prefix_length <= 0:
            prefix = None
        else:
            start = stop
            stop = start + bytes_per_token * footer.prefix_length
            prefix = tuple(
                self.slab[start:stop].view(torch.int32).numpy().tolist()
            )

        if footer.tokens_length <= 0:
            return None, tuple()

        start = stop
        stop = start + bytes_per_token * footer.tokens_length
        tokens = tuple(self.slab[start:stop].view(torch.int32).numpy().tolist())

        self._prefix = prefix
        self._tokens = tokens
        return self._prefix, self._tokens

    @staticmethod
    def use_compact_layout() -> bool:
        return MR_USE_COMPACT_LAYOUT

    @staticmethod
    def calculate_size(block_nbytes: int, ntokens: int) -> int:
        """Calculate the size of the MR.
        Args:
            block_nbytes: The size of the cache block in bytes.
            ntokens: The number of tokens.
        Returns:
            The size of the MR in bytes.
        """
        if MR_USE_COMPACT_LAYOUT:
            return block_nbytes
        else:
            # Layout: [cache block, magic, footer, tokens]
            magic_nbytes = np.dtype(np.int32).itemsize
            footer_nbytes = MemoryRegionFooter.nbytes()
            tokens_nbytes = np.dtype(np.int32).itemsize * ntokens
            size = int(
                block_nbytes + magic_nbytes + footer_nbytes + tokens_nbytes
            )
            return round_up(size, TensorPoolAllocator.ALLOC_SIZE_ALIGNMENT)


class TensorPoolAllocator:
    SLAB_MAX_NBYTES = 1 * 1024**3  # 1GB in bytes
    ALLOC_SIZE_ALIGNMENT = 16

    def __init__(
        self,
        *,
        capacity_nbytes: int,
        device: str = "cpu",
        pin_memory: bool = False,
    ) -> None:
        """Initialize the tensor pool allocator.
        Args:
            capacity_nbytes: The capacity of the allocator in bytes.
            device: The device to allocate the memory on.
            pin_memory: Whether to pin the memory.
        """
        self.capacity_nbytes: int = 0
        self._used_nbytes: int = 0
        self.device: str = "cpu" if device is None else device
        self.pin_memory: bool = pin_memory

        self._lock: Lock = Lock()
        self._original_slabs: List[torch.Tensor] = []
        self._merged_slabs: List[torch.Tensor] = []

        self._init(capacity_nbytes)

    def __len__(self) -> int:
        """Return nbytes allocated by the allocator."""
        with self._lock:
            return self._used_nbytes

    def __repr__(self) -> str:
        return (
            f"TensorPoolAllocator(capacity_nbytes={self.capacity_nbytes}, "
            f"used={self._used_nbytes}, device={self.device}, "
            f"pin_memory={self.pin_memory})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def slabs(self) -> List[torch.Tensor]:
        return self._merged_slabs

    def _init(self, size_nbytes: int) -> None:
        assert size_nbytes > 0, "size_nbytes must be greater than 0"
        slab_nbytes = self.SLAB_MAX_NBYTES

        size_nbytes = round_up(size_nbytes, slab_nbytes)
        nslabs = size_nbytes // slab_nbytes
        self.capacity_nbytes += nslabs * slab_nbytes

        for _ in tqdm(range(nslabs), desc="Allocate slabs"):
            slab = torch.empty(
                slab_nbytes,
                dtype=torch.uint8,
                device=self.device,
                pin_memory=self.pin_memory,
            )
            self._original_slabs.append(slab)

        # sort by memory address
        self._original_slabs = sorted(
            self._original_slabs, key=lambda x: x.data_ptr()
        )

        # merge adjacent slabs
        self._merged_slabs.append(self._original_slabs[0])
        for i in range(1, len(self._original_slabs)):
            slab1 = self._merged_slabs[-1]
            slab2 = self._original_slabs[i]
            if slab1.data_ptr() + slab1.numel() == slab2.data_ptr():
                merged_tensor = torch.as_strided(
                    slab1,
                    size=(slab1.numel() + slab2.numel(),),
                    stride=(1,),
                    storage_offset=slab1.storage_offset(),
                )
                self._merged_slabs[-1] = merged_tensor
            else:
                self._merged_slabs.append(slab2)

        slab_ptr_sizes = [
            (slab.data_ptr(), slab.numel()) for slab in self._merged_slabs
        ]
        self._pool = torch.classes._cpu_C_mem_ops.TensorPool(
            slab_ptr_sizes, TensorPoolAllocator.SLAB_MAX_NBYTES
        )

    def alloc(
        self, sizes: int | Sequence[int]
    ) -> Status[Sequence[MemoryRegion]]:
        if isinstance(sizes, int):
            sizes = (sizes,)

        if len(sizes) == 0:
            return Status(StatusCodes.INVALID)

        slab_idx_offsets = self._pool.allocate(sizes)
        if len(slab_idx_offsets) == 0:
            return Status(StatusCodes.OUT_OF_MEMORY)

        mrs: List[MemoryRegion] = []
        used_nbytes = 0
        for i, (slab_idx, offset) in enumerate(slab_idx_offsets):
            mr = MemoryRegion(
                self, self._merged_slabs[slab_idx], offset, sizes[i]
            )
            mrs.append(mr)
            used_nbytes += mr.length
            assert mr.length == sizes[i]

        with self._lock:
            self._used_nbytes += used_nbytes

        return Status.ok(mrs)

    def _finalize_mr(self, slab: torch.Tensor, addr: int, length: int) -> None:
        with self._lock:
            self._used_nbytes -= length
            assert self._used_nbytes >= 0
            self._pool.deallocate(slab.data_ptr() + addr)

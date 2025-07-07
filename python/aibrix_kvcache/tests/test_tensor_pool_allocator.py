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

import random

import pytest

from aibrix_kvcache.memory import TensorPoolAllocator


@pytest.fixture
def allocator(compact_layout_enabled):
    # use a small slab size for testing
    TensorPoolAllocator.SLAB_MAX_NBYTES = 2 ** 21
    TensorPoolAllocator.ALLOC_SIZE_ALIGNMENT = 8
    return TensorPoolAllocator(capacity_nbytes=2 ** 31)


def test_basic_allocation(allocator):
    """Test basic allocation and deallocation."""
    assert allocator.capacity_nbytes == 2 ** 31
    sizes = [2 ** 20] * 64  # 64 MB
    status = allocator.alloc(sizes)
    assert status.is_ok()
    mrs = status.value
    assert len(mrs) == len(sizes)
    assert sum([mr.length for mr in mrs]) == sum(sizes)
    [mr.ref_down() for mr in mrs]  # Trigger garbage collection


def test_allocating_large(allocator):
    """Test allocating with a set of sizes whose sum is larger than
    the slab size.
    """
    sizes = [2 ** 20 + 512] * 144
    status = allocator.alloc(sizes)
    assert status.is_ok()
    mrs = status.value
    assert len(mrs) == len(sizes)
    assert sum([mr.length for mr in mrs]) == sum(sizes)
    [mr.ref_down() for mr in mrs]  # Trigger garbage collection


def test_allocating_heterogeneous(allocator):
    """Test allocating with heterogeneous sizes."""
    sizes = [random.randint(2 ** 13, 2 ** 20) for _ in range(31)]
    status = allocator.alloc(sizes)
    assert status.is_ok()
    mrs = status.value
    assert len(mrs) == len(sizes)
    assert sum([mr.length for mr in mrs]) == sum(sizes)
    [mr.ref_down() for mr in mrs]  # Trigger garbage collection

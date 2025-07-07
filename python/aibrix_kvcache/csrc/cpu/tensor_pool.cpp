#include "tensor_pool.h"

#include <algorithm>

namespace aibrix {
namespace internal {

struct SlabRangeComparator {
  bool operator()(void *ptr, const std::pair<void *, slab_info> &entry) const {
    // Compare ptr with the end of the slab's memory range
    void *slab_end = static_cast<uint8_t *>(entry.first) + entry.second.size;
    return ptr < slab_end;
  }
};
} // namespace internal

tensor_pool::tensor_pool(const std::vector<std::vector<int64_t>> &slab_sizes,
                         int64_t slab_nbytes) {
  TORCH_CHECK(!slab_sizes.empty());
  TORCH_CHECK(slab_nbytes > 0);

  for (int i = 0; i < slab_sizes.size(); ++i) {
    const auto &slab_size = slab_sizes[i];
    TORCH_CHECK(slab_size.size() == 2);

    auto ptr = reinterpret_cast<void *>(slab_size[0]);
    auto size = slab_size[1];
    TORCH_CHECK(size % slab_nbytes == 0);

    slabs_.push(ptr);
    size = size - size / slab_nbytes * kSlabReservedSize;
    slab_map_[ptr] = internal::slab_info{i, size};
  }

  slab_nbytes_ = slab_nbytes - kSlabReservedSize;

  rml::MemPoolPolicy args(allocate_request, deallocate_request, slab_nbytes_);
  auto res = rml::pool_create_v1(intptr_t(this), &args, &my_pool);

  TORCH_CHECK(res == rml::POOL_OK);
}

tensor_pool::~tensor_pool() { destroy(); }

void *tensor_pool::allocate_request(intptr_t pool_id, size_t &bytes) {
  auto &self = *reinterpret_cast<tensor_pool *>(pool_id);
  TORCH_CHECK(0 == bytes % self.slab_nbytes_);

  void *slab_data_ptr;
  if (!self.slabs_.try_pop(slab_data_ptr)) {
    return nullptr;
  }

  // We only support allocate one slab at once
  bytes = self.slab_map_[slab_data_ptr].size;
  return slab_data_ptr;
}

int tensor_pool::deallocate_request(intptr_t pool_id, void *raw_ptr,
                                    size_t raw_bytes) {
  auto &self = *reinterpret_cast<tensor_pool *>(pool_id);
  TORCH_CHECK(0 == raw_bytes % self.slab_nbytes_);

  self.slabs_.push(raw_ptr);

  return 0;
}

void *tensor_pool::find_slab(void *ptr) const {
  auto it = std::upper_bound(slab_map_.begin(), slab_map_.end(), ptr,
                             internal::SlabRangeComparator{});
  TORCH_CHECK(it != slab_map_.end());
  return it->first;
}

std::vector<std::vector<int64_t>>
tensor_pool::allocate(const std::vector<int64_t> &sizes) {
  if (sizes.empty()) {
    return {};
  }

  std::vector<std::vector<int64_t>> buffers;
  buffers.reserve(sizes.size());

  for (auto size : sizes) {
    auto ptr = this->malloc(size);
    if (ptr == nullptr) {
      return buffers;
    }

    auto slab_ptr = find_slab(ptr);
    const auto &slab_info = slab_map_[slab_ptr];
    auto offset =
        static_cast<uint8_t *>(ptr) - static_cast<uint8_t *>(slab_ptr);
    buffers.push_back({slab_info.idx, offset});
  }
  return buffers;
}

void tensor_pool::deallocate(int64_t ptr) {
  this->free(reinterpret_cast<void *>(ptr));
}
} // namespace aibrix

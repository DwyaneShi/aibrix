#pragma once

#include "../define.h"

#include <map>
#include <vector>

#include <tbb/concurrent_queue.h>
#include <tbb/scalable_allocator.h>
#include <torch/all.h>

namespace aibrix {
namespace internal {
class pool_base : public no_copy {
public:
  void *malloc(size_t size) { return rml::pool_malloc(my_pool, size); }

  void free(void *ptr) { rml::pool_free(my_pool, ptr); }

protected:
  // destroy pool, must be called in a child class
  void destroy() { rml::pool_destroy(my_pool); }

  rml::MemoryPool *my_pool;
};

struct slab_info {
  int64_t idx;
  int64_t size;
};

} // namespace internal

class tensor_pool : public internal::pool_base,
                    public torch::CustomClassHolder {
public:
  // std::pair isn't directly supported for conversion between C++ and Python
  // so we use std::vector instead.
  explicit tensor_pool(const std::vector<std::vector<int64_t>> &slab_sizes,
                       int64_t slab_nbytes);
  ~tensor_pool();

  std::vector<std::vector<int64_t>> allocate(const std::vector<int64_t> &sizes);

  void deallocate(int64_t ptr);

private:
  // We use 32 bytes as a reserve for the slab to avoid tbb mergeing
  // adjacent slabs.
  static constexpr size_t kSlabReservedSize = 32;

  static void *allocate_request(intptr_t pool_id, size_t &bytes);
  static int deallocate_request(intptr_t pool_id, void *, size_t raw_bytes);

  void *find_slab(void *ptr) const;

  int64_t slab_nbytes_;
  tbb::concurrent_queue<void *> slabs_;
  std::map<void *, internal::slab_info> slab_map_;
};

} // namespace aibrix

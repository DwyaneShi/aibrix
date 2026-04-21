# CRITICAL BUG: FP8 Handling Broken in New Version

## User Context
在 vllm e2e 测试中 model 的 dytpe 是FP8 W8A8

## Analysis

### Old Version (torch-based)

**File:** `cache_kernels.cu:113, 162-167`
```cpp
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt, ...>
__global__ void reshape_and_cache_multi_layer_kernel(...) {
  ...
  if (TOnload) {
    kv_cache_layer[kv_cache_offset] =
        vllm::fp8::scaled_convert<cache_t, scalar_t, kv_dt>(
            offload_kv_cache_block[offload_kv_cache_offset],
            (kv_type == 0) ? *k_scale : *v_scale);
  }
}
```

**Dispatch for FP8 (via DISPATCH_BY_KV_CACHE_DTYPE):**
```cpp
if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {
  FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);      // Explicit E4M3
  FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
  FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
}
```

**Result:** Uses explicit `kFp8E4M3` for FP8 mode.

### New Version (ctypes-based)

**File:** `cache_kernels.cu:152-156`
```cpp
if constexpr (kIsOnload) {
  if constexpr (IsFP8<CacheT>::value) {
    cache_layer[cache_offset] =
        vllm::fp8::scaled_convert<CacheT, T,
                                  vllm::Fp8KVCacheDataType::kAuto>(  // HARDCODED kAuto!
            offload_block[offload_offset], *scale);
  }
}
```

### scaled_convert Function

**File:** `quant_utils.cuh:516-525`
```cpp
template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin &x, const float scale) {
#ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E5M2);
  }
#endif
  assert(false);  // <-- HITS THIS FOR kAuto!
  __builtin_unreachable();
}
```

## THE BUG

The `scaled_convert` function ONLY handles `kFp8E4M3` and `kFp8E5M2`. There is NO case for `kAuto`!

**Old version:** Uses explicit `kFp8E4M3` → Works correctly  
**New version:** Uses hardcoded `kAuto` → Hits `assert(false)` → **CRASH or UNDEFINED BEHAVIOR**

## Impact

- **E2E tests with FP8 W8A8 will FAIL**
- The kernel will either crash or produce garbage results
- This explains the different results in E2E testing!

## Fix Required

The new kernel must use explicit FP8 format like the old version:

```cpp
// Option 1: Template parameter for FP8 type
template <typename T, typename CacheT, bool kIsOnload, Layout kLayout, 
          vllm::Fp8KVCacheDataType kFp8Type = vllm::Fp8KVCacheDataType::kFp8E4M3>
__global__ void cache_kernel(...) {
  ...
  if constexpr (IsFP8<CacheT>::value) {
    cache_layer[cache_offset] =
        vllm::fp8::scaled_convert<CacheT, T, kFp8Type>(  // Use template param
            offload_block[offload_offset], *scale);
  }
}
```

Or pass the FP8 type through the dispatch logic to use the correct enum value.

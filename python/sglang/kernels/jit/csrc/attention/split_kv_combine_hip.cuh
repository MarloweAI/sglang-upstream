/// Split-KV combine for eight-head TileLang partials on gfx950.
/// Each block folds four heads over a 128-element value slice. The grid has
/// eight blocks per query token, with 128 threads per block.
#pragma once

#ifndef USE_ROCM
#error "The split-KV HIP combine requires ROCm"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx950__)
#error "The split-KV HIP combine is specialized for gfx950"
#endif

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_bfloat16.h>

#include <array>
#include <bit>
#include <cstdint>

namespace sglang {
namespace split_kv_combine {

// Keep hip_bfloat16 conversions and scalar math: changing them changes rounding.
using bfloat16_t = hip_bfloat16;
inline constexpr int kWidthFactor = 4;
inline constexpr int kSplits = 32;
inline constexpr int kThreads = 128;
inline constexpr int kGroups = 1;
inline constexpr int kHeadsPerBlock = 4;
inline constexpr int kValueWidth = 512;
inline constexpr int kSlice = kValueWidth / kWidthFactor;
inline constexpr int kLseTile = kSplits * kHeadsPerBlock;
static_assert(kThreads * kGroups * 4 == kHeadsPerBlock * kSlice);

// Preserve ascending split accumulation and the TileLang empty-split sentinel.

__global__ void __launch_bounds__(kThreads) split_kv_combine_hip_kernel(
    bfloat16_t* __restrict__ Output,
    const float* __restrict__ Partial_Lse,
    const bfloat16_t* __restrict__ Partial_O,
    int seq_len) {
  __shared__ float shared_lse[kLseTile];

  const int tid = (int)threadIdx.x;
  const int bx = (int)blockIdx.x;
  const int slice = bx % kWidthFactor;
  const int hb = (bx / kWidthFactor) & 1;
  const int s_i = bx / (2 * kWidthFactor);

  for (int j = tid; j < kLseTile; j += kThreads) {
    const int k = j >> 2;
    const int h = j & 3;
    shared_lse[k * kHeadsPerBlock + h] =
        Partial_Lse[(int64_t)s_i * (kSplits * 8) + (int64_t)k * 8 + (int64_t)hb * 4 + h];
  }
  __syncthreads();

#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
    const int e = (g * kThreads + tid) * 4;
    const int h_local = e / kSlice;
    const int d_local = e - h_local * kSlice;

    float lse[kSplits];
#pragma unroll
    for (int k = 0; k < kSplits; ++k)
      lse[k] = shared_lse[k * kHeadsPerBlock + h_local];

    float m = -1.073742e+09f;
#pragma unroll
    for (int k = 0; k < kSplits; ++k)
      m = max(m, lse[k]);

    float ssum = 0.000000e+00f;
#pragma unroll
    for (int k = 0; k < kSplits; ++k)
      ssum = (ssum + exp2f((lse[k] - m)));

    const float lg = __log2f(ssum);
    float sc[kSplits];
#pragma unroll
    for (int k = 0; k < kSplits; ++k)
      sc[k] = exp2f(((lse[k] - m) - lg));

    const int64_t o_off = (int64_t)s_i * 4096 + (int64_t)(hb * kHeadsPerBlock + h_local) * kValueWidth +
                          (int64_t)(slice * kSlice + d_local);
    const int64_t p_off = (int64_t)s_i * (kSplits * 4096) + (int64_t)(hb * kHeadsPerBlock + h_local) * kValueWidth +
                          (int64_t)(slice * kSlice + d_local);

    float4 acc = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
#pragma unroll
    for (int k = 0; k < kSplits; ++k) {
      uint2 packed;
      __builtin_memcpy(&packed, Partial_O + p_off + (int64_t)k * 4096, sizeof(packed));
      const auto values = std::bit_cast<std::array<bfloat16_t, 4>>(packed);
      const float f0 = (float)values[0];
      const float f1 = (float)values[1];
      const float f2 = (float)values[2];
      const float f3 = (float)values[3];
      acc.x = (acc.x + (sc[k] * f0));
      acc.y = (acc.y + (sc[k] * f1));
      acc.z = (acc.z + (sc[k] * f2));
      acc.w = (acc.w + (sc[k] * f3));
    }

    const std::array<bfloat16_t, 4> values{(bfloat16_t)acc.x, (bfloat16_t)acc.y, (bfloat16_t)acc.z, (bfloat16_t)acc.w};
    const auto packed = std::bit_cast<uint2>(values);
    __builtin_memcpy(Output + o_off, &packed, sizeof(packed));
  }
}

}  // namespace split_kv_combine

/// Validate contiguous BF16 partials, FP32 base-2 LSE and BF16 output, then
/// launch on the checked input device's current stream. The Python entry point
/// establishes the device context; direct FFI calls must do the same.
inline void
SplitKVCombineHIP(tvm::ffi::TensorView partial_o, tvm::ffi::TensorView partial_lse, tvm::ffi::TensorView out) {
  using namespace host;
  using namespace split_kv_combine;
  auto tokens = SymbolicSize{"tokens"};
  auto dev = SymbolicDevice{};
  dev.set_options<kDLGPU>();
  TensorMatcher({1, tokens, kSplits, 8, kValueWidth}).with_dtype<bf16_t>().with_device(dev).verify(partial_o);
  TensorMatcher({1, tokens, kSplits, 8}).with_dtype<float>().with_device(dev).verify(partial_lse);
  TensorMatcher({1, tokens, 8, kValueWidth}).with_dtype<bf16_t>().with_device(dev).verify(out);
  CHECK_HOST(tokens.unwrap() == 1 || tokens.unwrap() == 4) << "split-KV HIP combine supports one or four tokens";
  CHECK_HOST(reinterpret_cast<uintptr_t>(partial_o.data_ptr()) % alignof(uint2) == 0)
      << "partial_o must be aligned for four BF16 values";
  CHECK_HOST(reinterpret_cast<uintptr_t>(out.data_ptr()) % alignof(uint2) == 0)
      << "out must be aligned for four BF16 values";
  int current_device = -1;
  CHECK_CUDA(hipGetDevice(&current_device));
  CHECK_HOST(current_device == dev.unwrap().device_id)
      << "split-KV HIP combine requires the input device to be current";
  LaunchKernel(static_cast<uint32_t>(tokens.unwrap() * 2 * kWidthFactor), kThreads, dev.unwrap())(
      split_kv_combine_hip_kernel,
      static_cast<bfloat16_t*>(out.data_ptr()),
      static_cast<const float*>(partial_lse.data_ptr()),
      static_cast<const bfloat16_t*>(partial_o.data_ptr()),
      static_cast<int>(tokens.unwrap()));
}
}  // namespace sglang

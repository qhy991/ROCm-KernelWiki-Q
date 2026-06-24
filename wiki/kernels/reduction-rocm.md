---
id: kernel-reduction-rocm
title: "Reduction Kernels on ROCm"
type: wiki-kernel
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [reduction, wave-reduction, lds]
confidence: verified
sources: [pr-triton-457]
kernel_types: [reduction]
languages: [hip-cpp]
reproducibility: concept
---

# Reduction Kernels on ROCm

Reduction is a foundational primitive used heavily in operations like Softmax, LayerNorm, and RMSNorm. In ROCm/HIP, reductions are typically performed in two stages: Wavefront reduction and Thread Block reduction.

## 1. Wavefront-Level Reduction

Because AMD architectures utilize a 64-thread Wavefront, a single wave can reduce 64 elements very efficiently without touching shared memory (LDS).

This is done using Data Parallel Primitives (DPP) or standard warp shuffle operations (`__shfl_down`).

```cpp
template<typename T>
__device__ T wave_reduce_sum(T val) {
    // 64-thread wavefront reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, warpSize);
    }
    return val;
}
```

## 2. Block-Level Reduction

To reduce values across multiple wavefronts within a Thread Block, the LDS (Local Data Share) is required.

**Steps:**
1. Each Wavefront performs a `wave_reduce_sum`.
2. The first thread of each wavefront (`lane_id == 0`) writes its result to LDS.
3. A synchronization barrier (`__syncthreads()`) ensures all waves have written.
4. The first wavefront reads the values from LDS and performs a final `wave_reduce_sum`.

### Performance Considerations
- **Vectorized Loads**: When a single thread is reducing multiple elements from Global Memory, always use vectorized loads (e.g., `float4`) to maximize memory bandwidth before starting the cross-thread reduction.
- **Avoid Atomics**: Unless absolutely necessary, avoid global memory atomics (`atomicAdd`) for the final grid reduction. Instead, write partial results to a temporary buffer and launch a secondary kernel, or use a single-pass atomic strategy with a global counter.

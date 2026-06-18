---
id: technique-atomic-operations-hip
title: HIP Atomic Operations and Contention Reduction
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [memory, optimization, hip, synchronization]
confidence: source-reported
techniques: [wave-reduction]
hardware_features: [lds]
kernel_types: [reduction, histogram]
related: []
sources: []
reproducibility: snippet
---

# HIP Atomic Operations and Contention Reduction

## Introduction
HIP provides a rich set of atomic operations for both global memory and Local Data Share (LDS). On AMD architectures like CDNA2 (MI250X) and CDNA3 (MI300X), atomics are executed via dedicated hardware instructions. However, atomic contention can quickly become a performance bottleneck if many threads attempt to update the same memory location simultaneously. This page covers HIP atomic operations, underlying hardware instructions, performance characteristics, and strategies to minimize contention.

## Hardware Instructions
AMD GPUs accelerate atomics using specific instructions depending on the memory space and operation.

### Global Memory Atomics
For global memory, the GPU relies on the memory controllers and L2 cache. Instructions like `global_atomic_add` or `buffer_atomic_add` are used. 
On modern architectures like CDNA3, certain operations like `global_atomic_add_f32` (FP32 atomic add) are natively supported in hardware. In cases where native hardware support is lacking for a specific data type, the compiler typically generates a compare-and-swap (CAS) loop using `global_atomic_cmpswap`, which drastically reduces performance.

### LDS Atomics
LDS (Local Data Share) atomics are handled directly by the LDS hardware within the Compute Unit (CU). Instructions such as `ds_add_rtn_u32` or `ds_add_f32` are employed. LDS atomics are significantly faster than global atomics due to their low latency and proximity to the ALUs, but severe bank conflicts or bank serialization can occur under heavy contention.

## Performance Implications
- **Throughput:** Global atomics have high latency and limited throughput, bounded by the L2 cache bandwidth and atomic functional units. LDS atomics have much higher throughput but can still serialize wave execution if all threads target the same bank.
- **Return Values:** Instructions that return the old value (e.g., `atomicAdd` vs `atomicAdd_block`) typically have higher latency (e.g., `ds_add_rtn_f32` vs `ds_add_f32`). If the return value is not needed, avoid using it by casting the result to `void`.
- **CAS Loops:** Unoptimized atomics falling back to CAS loops (`atomicCAS`) will serialize threads and degrade performance exponentially with contention.

## Strategies to Reduce Atomic Contention

### 1. Wave-Level Reduction Before Atomics
Instead of having all 64 threads in a wavefront perform an atomic operation to the same address, perform a wave-level reduction first, and have only the first active thread issue a single atomic. This reduces atomic pressure by a factor of 64.

```cpp
#include <hip/hip_runtime.h>

__device__ void optimized_atomic_add(float* global_addr, float val) {
    // Perform cross-lane reduction using standard wave sync
    // Assuming 64 threads per wavefront on AMD
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    
    // Only lane 0 performs the atomic add
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(global_addr, val);
    }
}
```

### 2. LDS Privatization (Shared Memory Atomics)
If the kernel logic requires heavy atomics (e.g., histograms or prefix sums), it is better to accumulate intermediate results in LDS (which is fast) and then perform a final atomic update to global memory.

```cpp
__global__ void histogram_kernel(int* data, int* hist, int num_elements) {
    __shared__ int lds_hist[256];
    
    int tid = threadIdx.x;
    if (tid < 256) lds_hist[tid] = 0;
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int val = data[idx];
        // Fast atomic in LDS
        atomicAdd(&lds_hist[val], 1);
    }
    
    __syncthreads();
    
    // Final update to global memory using only a few atomics
    if (tid < 256) {
        atomicAdd(&hist[tid], lds_hist[tid]);
    }
}
```

### 3. Avoiding Atomics via Output Tiling
Whenever possible, restructure the algorithm so that each thread block writes to a unique memory location. For example, in a reduction kernel, each block can write its partial sum to a separate index in a global array. A second pass kernel can then read these partial sums and perform the final reduction. This completely eliminates global atomics in the first pass.

### 4. Floating Point Determinism
Atomic operations on floating-point numbers (`float` or `double`) are non-deterministic due to the non-associativity of floating-point addition. Execution order affects the final value. If determinism is strictly required, atomics must be avoided, and a deterministic reduction tree must be used instead.

## Architecture Specifics
- **CDNA2 (MI250X):** Features native support for 32-bit and 64-bit integer atomics, as well as FP32 atomics. Certain FP64 atomics might use CAS loops depending on the exact instruction or memory target. High contention on LDS atomics can serialize LDS access.
- **CDNA3 (MI300X):** Extended hardware support for atomics, including improved FP32 and FP64 atomic add throughput in the L2 cache. `buffer_atomic_add_f32` provides robust scaling for global reductions.

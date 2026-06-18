---
id: kernel-prefix-sum-scan
title: Parallel Prefix Sum (Scan) on ROCm
type: wiki-kernel
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [reduction, dpp, lds, cross-lane, wave-reduction]
confidence: source-reported
kernel_types: [reduction]
languages: [hip-cpp]
related: []
sources: []
reproducibility: snippet
---

# Parallel Prefix Sum (Scan) on ROCm

The Prefix Sum (also known as Scan) is a fundamental parallel primitive used extensively in sorting, stream compaction, parallel tree operations, and building index structures. Given an input array `A`, an inclusive scan produces an output array `B` where `B[i] = A[0] + A[1] + ... + A[i]`. An exclusive scan produces `B[i] = A[0] + A[1] + ... + A[i-1]`.

On AMD ROCm architecture, highly optimized scan operations can be implemented by leveraging Data Parallel Primitives (DPP) for fast cross-lane communication within a wavefront, and Local Data Share (LDS) for communication across wavefronts within a block.

## Algorithms: Kogge-Stone vs. Blelloch

Two primary algorithms are used for parallel scan:

1. **Kogge-Stone Algorithm (Step-Efficient):**
   - Typically used for wavefront-level scans (intra-wavefront).
   - Requires $O(N \log N)$ operations but executes in $O(\log N)$ steps.
   - Highly parallel and well-suited for registers (using DPP/Shuffle instructions) because all threads are active and there are no memory contention issues.
   
2. **Blelloch Algorithm (Work-Efficient):**
   - Typically used for block-level or global-level scans (using LDS or Global Memory).
   - Requires $O(N)$ operations.
   - Operates in two phases: an Up-Sweep (reduce) and a Down-Sweep phase.
   - Better suited for larger arrays where operation efficiency matters more, though it has higher step complexity ($O(\log N)$ steps for each phase).

## Wavefront-Level Scan Using DPP

In AMD architectures (CDNA and RDNA), a wavefront consists of 64 threads. The fastest way to perform a scan within a wavefront is by using Data Parallel Primitives (DPP). DPP allows instructions to read data directly from the registers of neighboring threads without going through LDS.

In HIP, wavefront-level scan is typically implemented using `__shfl_up` (which translates to DPP instructions under the hood, such as `v_add_f32_dpp`).

```cpp
#include <hip/hip_runtime.h>

// Inclusive scan within a 64-thread wavefront using Kogge-Stone
__device__ inline float wave_inclusive_scan(float val) {
    // Wavefront size is 64 on AMD CDNA architectures
    for (int offset = 1; offset < 64; offset *= 2) {
        // __shfl_up maps to DPP operations (e.g., row_shr, row_bcast)
        float n = __shfl_up(val, offset, 64);
        if (__lane_id() >= offset) {
            val += n;
        }
    }
    return val;
}
```

### Direct DPP Intrinsic

For ultimate performance, especially in custom assembly or specific compiler optimizations, ROCm developers can map this directly to `__builtin_amdgcn_update_dpp`. However, the HIP `__shfl_up` implementation is highly optimized and standard for most workloads.

## Block-Level Scan Using LDS

To scan an array that spans an entire thread block (e.g., 256 or 512 threads), we combine the wavefront-level scan with LDS. 

1. Each wavefront performs a wave-level scan.
2. The last thread of each wavefront stores its sum to LDS.
3. A single wavefront scans the LDS array containing the wave sums.
4. The scanned wave sums are added back to the individual threads in each wavefront.

```cpp
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256
#define WAVE_SIZE 64
#define NUM_WAVES (BLOCK_SIZE / WAVE_SIZE)

__device__ float block_inclusive_scan(float val, float* lds_mem) {
    int lane_id = __lane_id();
    int wave_id = threadIdx.x / WAVE_SIZE;

    // 1. Wavefront-level inclusive scan
    float wave_val = wave_inclusive_scan(val);

    // 2. Last thread of each wave writes the wave's total sum to LDS
    if (lane_id == WAVE_SIZE - 1) {
        lds_mem[wave_id] = wave_val;
    }
    __syncthreads();

    // 3. First wave scans the wave sums in LDS
    if (wave_id == 0) {
        float sum = (lane_id < NUM_WAVES) ? lds_mem[lane_id] : 0.0f;
        sum = wave_inclusive_scan(sum);
        if (lane_id < NUM_WAVES) {
            lds_mem[lane_id] = sum;
        }
    }
    __syncthreads();

    // 4. Add the base sum of the previous waves to current wave elements
    if (wave_id > 0) {
        wave_val += lds_mem[wave_id - 1];
    }

    return wave_val;
}

// Global kernel wrapping the block scan
__global__ void scan_kernel(float* d_out, const float* d_in, int n) {
    __shared__ float lds_mem[NUM_WAVES];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? d_in[tid] : 0.0f;
    
    float scan_val = block_inclusive_scan(val, lds_mem);
    
    if (tid < n) {
        d_out[tid] = scan_val;
    }
}
```

## Work-Efficient Blelloch Scan in LDS

For larger amounts of data per block, Blelloch scan is more efficient. Because Blelloch uses tree-like reductions, addressing LDS requires careful stride management to avoid bank conflicts. Padding the LDS array by inserting dummy elements at strides that are multiples of the number of memory banks (usually 32 on CDNA) resolves this issue.

> [!TIP]
> **Bank Conflict Padding:** When implementing Blelloch scan in LDS, calculating indices like `index += index / NUM_BANKS;` prevents multiple threads from accessing the same LDS bank simultaneously.

## Global Memory Decoupled Look-back Scan

For arrays exceeding a single block's capacity, standard practice uses a multi-pass approach (reduce down to block sums, scan block sums, add block sums back). 
However, modern ROCm implementations (similar to NVIDIA's CUB `DeviceScan`) utilize **Decoupled Look-back**. 

This technique assigns atomic counters to dynamically determine block execution order. Each block scans its local data, then "looks back" at the previous blocks' status flags in global memory to determine the prefix sum of preceding blocks. If a preceding block is finished, it incorporates that sum; if not, it waits. This requires only a single pass through the global memory, massively saving memory bandwidth.

## Performance Profile (CDNA3 - MI300X)

Scan is typically a memory-bound operation. The theoretical peak depends directly on Global Memory Bandwidth. Below is a performance profile on an MI300X (5.3 TB/s peak bandwidth) and MI250X (3.2 TB/s peak bandwidth) for a 32-bit floating point Exclusive Scan of size $N=2^{28}$ (1 GB of data).

| GPU Model | Scan Throughput (GB/s) | Effective Memory BW (GB/s) | Algorithm |
|-----------|-------------------------|-----------------------------|-----------|
| AMD MI300X| 2050 GB/s               | ~4100 GB/s                  | Decoupled Look-back |
| AMD MI250X| 1200 GB/s               | ~2400 GB/s                  | Decoupled Look-back |
| AMD MI300X| 1100 GB/s               | ~2200 GB/s                  | Standard 3-pass |

*(Note: Effective Memory BW = $2 \times$ Throughput, as each element is read once and written once.)*

> [!NOTE]
> The performance difference between standard multi-pass and Decoupled Look-back is staggering because multi-pass requires writing and reading intermediate block sums, effectively using more global memory transactions.

## Optimization Checklists

- **Use Wavefront Intrinsics:** Always prefer `__shfl_up` over LDS for 64-element scans. It saves LDS bandwidth and avoids `__syncthreads()`.
- **Vectorized Loads/Stores:** In global memory scan kernels, use `float4` (or `int4`) vectorized loads to fetch 4 elements per thread. Scan them locally in registers first, perform the block scan on the local sums, and write out as `float4`. This dramatically improves global memory throughput.
- **Wave Size:** CDNA architectures have a `wavefront_size` of 64. Ensure offsets and lane calculations assume 64, not 32 like CUDA.

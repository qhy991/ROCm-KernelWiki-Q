---
id: hw-hbm-bandwidth
title: HBM3 Bandwidth Characteristics on CDNA3
type: wiki-hardware
architectures: [cdna3]
tags: [memory, bandwidth, hbm, mi300x]
confidence: source-reported
hardware_features: []
related: []
sources: []
cuda_equivalent: hbm
---

## Overview

The AMD Instinct MI300X (CDNA3) is equipped with 8 stacks of High Bandwidth Memory 3 (HBM3), providing up to 192 GB of capacity and a theoretical peak bandwidth of 5.3 TB/s. Optimizing memory-bound kernels for the MI300X requires a deep understanding of the HBM bank structure, vectorizing loads and stores, and mitigating memory stalls effectively.

## HBM3 Architecture on CDNA3

### Memory Bank and Channel Structure
- **Stacks**: The MI300X package includes 8 HBM3 stacks surrounding the XCD (compute) dies.
- **Channels**: Each HBM3 stack typically features 16 independent memory channels, which are further divided into 2 pseudo-channels each, yielding 32 pseudo-channels per stack.
- **Total Interfaces**: Across all 8 stacks, the MI300X exposes 256 independent pseudo-channels, allowing massive spatial parallelism for memory requests.
- **Bus Width**: 1024-bit interface per stack (8192-bit aggregate bus width across the GPU).

Each memory request originating from a wavefront is processed through the L1 cache (TCP), L2 cache (TCC), and finally routed to the HBM memory controllers (UMC). To saturate the 5.3 TB/s bandwidth, a kernel must maintain enough in-flight memory requests across these channels to hide the HBM latency.

## Peak vs. Achievable Bandwidth

While the theoretical peak bandwidth of MI300X is 5.3 TB/s, achieving this in practice depends heavily on access patterns, vectorization, and thread occupancy. Due to protocol overhead, refresh cycles, and controller efficiency, the maximum achievable sustained bandwidth on large sequential reads (like a BabelStream triad) is typically lower than the theoretical peak.

### Performance Data

| GPU | Architecture | Memory Type | Peak Bandwidth | Max Achievable (Stream Triad) | Efficiency |
|---|---|---|---|---|---|
| MI250X | CDNA2 | HBM2e (128GB) | 3.2 TB/s | ~2.7 TB/s | ~84% |
| MI300X | CDNA3 | HBM3 (192GB) | 5.3 TB/s | ~4.6 TB/s | ~87% |

To reach achievable bandwidth (e.g., >4.5 TB/s on MI300X), kernels must adhere to the following best practices:
1. **Use Vectorized Memory Instructions**: Utilize `buffer_load_dwordx4` or `global_load_dwordx4` (128-bit loads per thread) to maximize payload per memory request.
2. **Coalesce Memory Accesses**: Ensure threads within a wavefront access contiguous memory addresses so the memory controller can combine requests into single 64-byte or 128-byte transactions.
3. **Maintain High Occupancy**: Keep enough wavefronts active on each Compute Unit (CU) to issue new memory requests while other wavefronts are stalled waiting for memory returns.

### Code Example: Vectorized Copy for High Bandwidth

Using `uint4` (128-bit) or `float4` types in HIP C++ ensures the compiler emits optimal `global_load_dwordx4` instructions, which is crucial for maximizing memory throughput on CDNA.

```cpp
#include <hip/hip_runtime.h>

// Vectorized copy kernel using 128-bit loads/stores
__global__ void copy_kernel_vectorized(const uint4* __restrict__ src, 
                                       uint4* __restrict__ dst, 
                                       size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads and stores 16 bytes (128 bits) per iteration
    if (tid < n) {
        // Generates global_load_dwordx4 and global_store_dwordx4 in ISA
        dst[tid] = src[tid]; 
    }
}

// Host launch configuration example
void launch_copy(const uint4* src, uint4* dst, size_t elements, hipStream_t stream) {
    int threads = 256; // 4 wavefronts per block, good for occupancy
    int blocks = (elements + threads - 1) / threads;
    
    hipLaunchKernelGGL(copy_kernel_vectorized, dim3(blocks), dim3(threads), 0, stream, src, dst, elements);
}
```

## Profiling Memory Stalls

When a kernel's bandwidth falls short of expectations, profiling memory stalls is essential to determine if the issue is latency exposure, memory controller saturation, or uncoalesced accesses.

### rocprofiler Metrics

Use `rocprof` (or the newer `rocprofv2` in ROCm 6.x) to capture key memory stall metrics on CDNA architectures:

- `MemUnitBusy`: The percentage of time the memory unit is active processing requests. If this is high (>80%), the kernel is genuinely memory bandwidth bound.
- `MemUnitStalled`: The percentage of time the memory unit is stalled. High stall times can indicate excessive uncoalesced accesses, bank conflicts in the memory subsystem, or L2 cache thrashing.
- `SQ_WAVES_WAITING_FOR_VMEM`: Shows the average number of wavefronts waiting on vector memory (VMEM) instructions. High values suggest that memory latency is not being fully hidden by compute or other active wavefronts.
- `TCC_HIT_RATE`: L2 cache hit rate. A low hit rate coupled with high `MemUnitBusy` implies high HBM traffic, which is expected for purely memory-bound streaming kernels.

### Example rocprof Command

```bash
# Profile key memory metrics for a HIP application
rocprof --stats -m MemUnitBusy,MemUnitStalled,TCC_HIT_RATE,SQ_WAVES_WAITING_FOR_VMEM ./my_hip_app
```

### Strategies for Addressing Memory Stalls

If profiling reveals that memory stalls are the primary bottleneck, consider the following optimization strategies:
1. **Unroll Loops**: Loop unrolling allows the compiler to issue multiple independent load instructions before executing a `s_waitcnt` (wait count) instruction, thereby increasing memory-level parallelism (MLP).
2. **Software Pipelining / Pre-fetching**: Fetch data into Vector General Purpose Registers (VGPRs) or Local Data Share (LDS) earlier in the pipeline to overlap memory latency with compute (e.g., using double buffering patterns).
3. **Data Alignment**: Ensure data structures are aligned to 16-byte boundaries (`__align__(16)`) so they can be loaded cleanly using vectorized types without compiler fallback to scalar or smaller vector loads.

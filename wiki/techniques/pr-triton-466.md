---
id: technique-pr-triton-466
title: "Fix vecSize for fp8 and int8 on MI300"
type: wiki-technique
architectures: [cdna3]
tags: [vectorized-load, triton-rocm, fp8, int8, bandwidth, memory-bound, optimization]
confidence: inferred
sources:
  - pr-triton-466
---

# PR Insight: Fix vecSize for fp8 and int8 on MI300

## Summary
This PR addresses a critical optimization bottleneck in the ROCm Triton backend where the vector size (`vecSize`) for 8-bit data types (`fp8` and `int8`) was not optimally configured for MI300 (CDNA3) hardware. By correcting the maximum allowable `vecSize` to 16 for 8-bit types, the compiler is now able to emit optimal 128-bit (16-byte) vectorized load and store instructions. This dramatically improves memory bandwidth utilization and reduces instruction issue overhead for low-precision kernels.

## Problem Statement

In the AMDGPU instruction set, memory operations achieve peak efficiency when using the widest possible memory instructions. A single thread can fetch or store up to 128 bits (16 bytes) at a time using instructions like `global_load_dwordx4` for HBM and `ds_read_b128` for Local Data Share (LDS). 

To maximize the utilization of the memory subsystem, compilers group adjacent elements into vector types:

| Data Type | Element Size (Bytes) | Max Elements for 128-bit Load (`vecSize`) | Under-the-hood AMDGPU Instruction |
|-----------|----------------------|-------------------------------------------|-----------------------------------|
| `fp32` / `int32` | 4 | 4 | `global_load_dwordx4` |
| `fp16` / `bf16` | 2 | 8 | `global_load_dwordx4` (packed) |
| **`fp8` / `int8`** | **1** | **16** | **`global_load_dwordx4` (packed)** |

Prior to this fix, the Triton backend had restrictive logic for the maximum `vecSize` of 8-bit types, likely limiting them to 4 or 8 elements due to unoptimized type inference constraints during the introduction of `fp8` support. This resulted in suboptimal memory access patterns (e.g., using `vecSize=8` or `vecSize=4`), which effectively translates to `global_load_dwordx2` (8 bytes) or smaller. This artificial limit cuts the peak memory throughput in half, leaving significant performance on the table for memory-bound operators.

> [!IMPORTANT]
> AMD CDNA architectures impose a maximum memory operation width of 128 bits (16 bytes) per thread for standard global and LDS vectorized loads. To saturate this link width using 1-byte data types, a vector size of exactly 16 must be achieved.

## Architectural Context: CDNA3 (MI300)

MI300 (CDNA3 architecture) introduces native hardware acceleration for `fp8` and `bf8` matrix math operations via specialized MFMA (Matrix Fused Multiply-Add) instructions. However, to keep these high-throughput compute units fed, the memory pipeline (moving data from Global Memory to LDS, and from LDS to VGPRs) must be perfectly optimized.

By exploiting a 16-element vector size, each wavefront minimizes the number of memory instructions it issues to accomplish the same amount of data movement:
1. **Reduced Instruction Count:** Halving the number of memory instructions reduces instruction cache footprint and relieves congestion in the instruction scheduler.
2. **VGPR Address Conservation:** Minimizing separate memory loads reduces the amount of vector general-purpose registers (VGPRs) consumed purely for address generation/offsets, which can directly improve occupancy.
3. **Full Link Bandwidth:** Fully utilizes the 16-byte memory channels available per thread to the LDS or global memory interface.

## Implementation Insights & Impact

- **Vectorized Loads/Stores**: The compiler pass responsible for computing memory coalescing and vectorization now properly identifies contiguous blocks of 8-bit elements and coalesces them up to 16 elements. 
- **Register Packing**: At the LLVM IR and ISA levels, the 16 bytes of `fp8`/`int8` data are seamlessly mapped onto 4 32-bit VGPRs.
- **Performance Impact**: Workloads heavily utilizing `fp8` or `int8`—such as quantized GEMM variants, Flash Attention, and MOE routing—experience a noticeable uplift in raw memory bandwidth. This directly reduces latency in the memory-bound phases of low-precision compute kernels.

## Conclusion
Correcting the vector size limits for 8-bit types allows the Triton compiler to transparently emit the widest possible memory operations (`global_load_dwordx4`, `global_store_dwordx4`, `ds_read_b128`) for `fp8` and `int8` arrays. This is an essential compiler-level optimization for achieving theoretical roofline performance on the AMD MI300 architecture.

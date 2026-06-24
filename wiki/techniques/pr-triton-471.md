---
id: pr-triton-471
type: source-pr
repo: ROCm/triton
pr: 471
title: "fix warp size in lowering reduce op"
author: scxiao
date: '2024-01-18'
url: https://github.com/ROCm/triton/pull/471
source_category: upstream-code
architectures: 
  - cdna2
  - cdna3
  - cdna4
tags: 
  - rocm
  - rocm-kernel
  - optimization
kernel_types: 
  - reduction
languages: 
  - triton-rocm
captured_at: '2026-06-18'
status: merged
inclusion_reason: "may contain relevant infrastructure changes"
confidence: source-reported
---

# Fix Warp Size in Lowering Reduce Op

## 1. Background and Motivation

In the Triton programming model, high-level tensor operations such as `tl.reduce` (which encapsulates sum, max, min, and other associative reduction operations) are lowered through a series of MLIR dialects before becoming target-specific LLVM IR. For GPU targets, reduction operations are optimized using efficient intra-warp shuffles (often called butterfly reductions) followed by inter-warp reductions via shared memory if the reduction axis spans multiple warps.

A critical architectural distinction between NVIDIA GPUs and AMD GPUs is their execution wavefront width. NVIDIA's warp size is uniformly 32 threads. However, AMD's wavefront size (often configured as `wave64` on CDNA architectures such as CDNA2, CDNA3, and CDNA4) operates with 64 threads.

When Triton was initially developed, certain reduction lowerings implicitly assumed a warp size of 32 or relied on hardcoded fallbacks instead of correctly querying the target-specific `threadsPerWarp` parameter from the `triton_gpu` dialect. When executed on ROCm with `wave64`, an incorrect warp size configuration during reduction lowering results in partial intra-warp reductions (e.g., only 32 threads communicating via shuffle instead of 64), leading to incorrect computation results.

## 2. Technical Analysis and Code Intent

The pull request **"fix warp size in lowering reduce op"** directly addresses this hardware-specific lowering mismatch by ensuring that the actual wave size is used during the `TritonGPUToLLVM` reduction lowering phase.

### 2.1 The Lowering Pipeline for `tl.reduce`
1. **Triton Dialect (`triton.reduce`)**: Represents the reduction operation mathematically along a specified axis.
2. **TritonGPU Dialect (`triton_gpu.reduce`)**: Annotates the reduction with distributed layout information, detailing how elements are sharded across threads, warps, and thread blocks.
3. **LLVM Dialect Conversion (`TritonGPUToLLVM`)**: The compiler lowers the reduction loop into specific shuffle primitives (like `ds_bpermute` or `v_add_f32` on AMD).

### 2.2 Root Cause of the Bug
In a butterfly reduction, threads exchange data iteratively at varying stride offsets. The total number of shuffle steps is given by $\log_2(\text{warp\_size})$.
- For $W = 32$, this requires **5 steps** (strides of 1, 2, 4, 8, 16).
- For $W = 64$, this requires **6 steps** (strides of 1, 2, 4, 8, 16, 32).

If the compiler hardcodes or miscalculates the warp size as 32 for an AMD wave64 target, the 6th shuffle step (offset 32) is entirely omitted. As a result, threads 0-31 will not exchange data with threads 32-63, yielding two independent reduction results per wavefront rather than one unified scalar result.

### 2.3 The Architectural Fix
The fix ensures that the reduction lowering dynamically computes the number of iterations based on the target execution characteristics:

- **Dynamic Query**: The code is adjusted to dynamically evaluate the target's configured `threadsPerWarp` (querying the module or function attributes for wave64 vs wave32 specifications) instead of defaulting to 32.
- **Loop Bounds Calculation**: The butterfly reduction loop bounds are correctly updated to evaluate $\log_2(\text{threadsPerWarp})$, allowing the LLVM lowering phase to emit all 6 iteration steps on AMD CDNA architectures.

## 3. Impact and Optimizations

- **Mathematical Correctness**: Resolves data corruption and incomplete reduction results on AMD CDNA GPUs, restoring accuracy for any operator relying on `tl.reduce` (such as `softmax`, `sum`, `max`, `argmin`).
- **Performance Fidelity**: Correctly leveraging wave64 for reductions reduces the overhead of falling back to Shared Memory (LDS) for inter-warp synchronization, maximizing hardware throughput.
- **Backend Abstraction**: Reinforces the backend-agnostic design of Triton by formalizing `threadsPerWarp` as a dynamically queried compiler parameter, ensuring forward compatibility with potential future architectures of varying execution widths.

## 4. Key Takeaways for ROCm Kernel Optimization

1. **Warp vs. Wavefront**: When porting or maintaining compiler passes from CUDA to ROCm, the assumption that `warp_size == 32` must be meticulously audited. AMD architectures operate fundamentally on `wave64` for maximum tensor core efficiency on CDNA.
2. **Intra-Wave Shuffles**: Operations that rely on $O(\log_2 W)$ lane communications must adapt gracefully. Failing to account for all 64 threads partitions the wavefront into disconnected halves.
3. **Compiler Attributes**: Always extract architectural limits via target attributes (e.g., `triton_gpu.num-warps`, `threadsPerWarp`) rather than relying on global constants in MLIR transformations.

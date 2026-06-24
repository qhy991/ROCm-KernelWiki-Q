---
id: technique-pr-triton-513
title: "Bfloat16 Support for Fused Attention (Transposed V) in Triton"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - optimization
  - bf16
  - triton-rocm
  - mfma
  - memory-bound
  - flash-attention
confidence: inferred
sources:
  - pr-triton-513
---

# Bfloat16 Support for Fused Attention (Transposed V) in Triton

## Overview

PR [#513](https://github.com/ROCm/triton/pull/513) in the `ROCm/triton` repository introduces native `bf16` (bfloat16) support to the performance-critical Triton kernel `06-fused-attention-fwd-transV.py`. This kernel implements a fused attention forward pass (a core building block of FlashAttention architectures) where the Value (`V`) matrix is stored in a transposed layout. Implementing `bf16` support enables this kernel to directly utilize hardware Matrix Core acceleration on AMD CDNA architectures while drastically reducing memory bandwidth pressure.

## Architectural Context

Fused attention kernels are classically constrained by HBM memory bandwidth and Local Data Share (LDS) throughput rather than raw compute. Using the `bf16` numerical format is a vital optimization: it maintains the dynamic range of `fp32` (using the same 8-bit exponent) while halving the memory footprint (16 bits per element). 

On AMD CDNA2, CDNA3, and CDNA4 hardware, `bf16` is a native operand type for Matrix Fused Multiply-Add (MFMA) instructions (e.g., `v_mfma_f32_16x16x16bf16`). Supporting `bf16` in Triton's fused attention means lowering the logical tensor operations efficiently into these high-throughput hardware instructions without incurring runtime cast overheads or register bloat.

## Key Optimizations & Techniques

### 1. Vectorized Memory Accesses
Halving the element size from `fp32` to `bf16` enables doubling the elements loaded per vector instruction. Since the `V` matrix is transposed (`transV`), thread-to-memory mapping must be carefully coordinated to ensure that vector loads along the contiguous dimension happen effectively. This vectorization (using `global_load_dwordx4` for 128-bit loads fetching 8 `bf16` elements) is strictly necessary to saturate HBM bandwidth.

### 2. MFMA Layout Alignment
Triton's AMD backend converts warp-level dot products into AMD's MFMA intrinsics. For `bf16`, the compiler and kernel layout must arrange data to match the precise register fragment layout expected by `bf16` MFMA instructions. 
- **First GEMM (`Q * K^T`)**: Queries and Keys are multiplied to produce `fp32` accumulators.
- **Second GEMM (`P * V`)**: The Transposed `V` layout must be fed into the MFMA units properly. The transposed memory format implies that the `V` fragment layouts within VGPRs must be correctly ordered to match the expected non-transposed standard iteration expected by the underlying dot operation, avoiding costly runtime permutation operations in the registers.

### 3. LDS Bank Conflict Mitigation
Holding intermediate tiles of `K` and `V` in LDS (shared memory) requires mitigating bank conflicts, which are particularly sensitive when data types change. AMD's LDS features 32 banks, each 4 bytes wide. Because `bf16` elements are 2 bytes, a consecutive read of 16 elements exactly fills a 32-byte cache line without spanning full bank width if poorly aligned. For a transposed `V` matrix, reading columns could trigger severe multi-way bank conflicts. Triton relies on XOR-based swizzling to permute addresses dynamically, ensuring that threads within a wavefront load `bf16` fragments collision-free.

## Memory Bounds Analysis

- **Memory/Bandwidth Bound Phase**: Moving to `bf16` strictly reduces the byte-to-FLOP ratio by 2x. This pushes the memory-bound limits higher, enabling larger sequence lengths and batch sizes to fit within the same HBM bandwidth profile.
- **Compute Bound Phase**: Leveraging `v_mfma_f32_16x16x16bf16` unlocks the peak computational throughput of the matrix cores (e.g., over 1.3 PFLOPS on MI300X). The inner loops doing the `Q * K^T` and `P * V` reductions are highly compute-bound, meaning full occupancy and properly pipelined MFMAs are essential to hide latency.

> [!TIP]
> **Implementation Note:** When porting Triton kernels to support `bf16` on AMD hardware, pay special attention to tensor block layouts. While `bf16` theoretically speeds up computation, incorrect layouts can cause the Triton compiler to emit extensive `ds_bpermute` or LDS traffic to realign fragments for the MFMA instructions, wiping out performance gains.

---
id: technique-pr-triton-606
title: "Triton GEMM Tuning Script v3.3"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - optimization
  - rocm
  - gemm
  - triton-rocm
  - occupancy-tuning
  - double-buffering
confidence: inferred
sources:
  - pr-triton-606
---

# Triton GEMM Tuning Script v3.3 (PR #606)

## Context
This page details the architectural concepts behind [ROCm/triton PR 606](https://github.com/ROCm/triton/pull/606), which introduces version 3.3 of the GEMM tuning script for ROCm. In Triton, high performance on AMD CDNA architectures relies heavily on auto-tuning matrix block sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`), wavefront dimensions (`num_warps`), and software pipelining depths (`num_stages`). 

## Intent
The primary intent of the updated tuning script is to systematically explore the kernel configuration search space to find optimal settings for any given matrix shape (M, N, K) and data type. By providing an automated and robust tuning suite, developers can ensure that compiled Triton kernels achieve peak hardware utilization and throughput on ROCm devices without labor-intensive manual tuning.

## Optimization Technique
The tuning script generates, compiles, and benchmarks multiple configurations to map the problem effectively onto the hardware:
- **Block Tiling Strategy**: Iterating through different tile dimensions (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) to optimize L2 cache hit rates and Local Data Share (LDS) usage.
- **Wavefront Occupancy Scaling**: Adjusting `num_warps` to balance Vector General Purpose Register (VGPR) consumption and Thread Block (Workgroup) occupancy. On CDNA hardware, maintaining sufficient active wavefronts is critical for hiding memory fetch latency.
- **Software Pipelining & Double Buffering**: Varying `num_stages` to find the optimal buffer depth for multi-buffering. This facilitates maximum overlap of asynchronous global-to-LDS memory copies with Matrix Fused Multiply-Add (MFMA) execution.
- **MFMA Alignment**: The script inherently discovers tile configurations that best map to the hardware-native MFMA instruction shapes available on CDNA architectures (e.g., 32x32x8, 16x16x16).

## Memory Bounds and Performance Trade-offs
- **Compute-Bound Scenarios**: For large M, N, K matrices with high arithmetic intensity, the tuning script favors configurations that maximize raw MFMA throughput. This typically involves selecting larger block dimensions to reuse data effectively in LDS and VGPRs.
- **Memory-Bound Scenarios**: For matrix dimensions where K is small or shapes represent "skinny" matrices (such as in certain batched or grouped GEMMs), the operation becomes memory-bound. Here, the script pivots toward configurations that maximize global memory read bandwidth, occasionally selecting wider vectorized loads and fewer pipeline stages to lower register pressure.
- **LDS and VGPR Pressure Limits**: A fundamental limitation the script maps out is the trade-off between deeper pipelines (`num_stages`) and tile sizes versus available LDS and VGPRs per Compute Unit (CU). Exceeding hardware capacities triggers register spilling or lowers occupancy, degrading performance. The tuning sweep systematically detects these "cliffs" to select the highest-performing valid configuration.

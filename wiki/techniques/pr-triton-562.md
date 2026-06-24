---
id: technique-pr-triton-562
title: "PR Insight: triton #562 - Add kpack and matrix_instr_nonkdim for Stream-K Implementation"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
hardware_features:
  - mfma
  - lds
  - compute-unit
tags:
  - rocm-kernel
  - optimization
  - scheduling
  - compute
techniques:
  - mfma-scheduling
  - occupancy-tuning
  - vectorized-load
kernel_types:
  - gemm
languages:
  - triton-rocm
confidence: inferred
sources:
  - pr-triton-562
---

# Analysis of PR #562 in ROCm/triton

## Summary
PR #562 enables two critical tuning parameters—`kpack` and `matrix_instr_nonkdim` (also referred to as `mfma_instr_nokdim`)—in the auto-tuning space for **Stream-K** GEMM implementations within the AMD ROCm backend for Triton. By exposing these options, the PR empowers the Triton autotuner to systematically optimize vectorization of Local Data Share (LDS) reads and fine-tune Matrix Fused Multiply-Add (MFMA) instruction shapes specifically for the Stream-K load-balancing scheduling strategy. 

## Architectural and Code Analysis

### Background: Stream-K Load Balancing
Standard tiled GEMM execution splits the output matrix into tiles across the M and N dimensions and assigns each tile to a Compute Unit (CU). This strategy can lead to wave quantization issues (the "tail effect") where some CUs finish early while others are still computing, leading to underutilization of the GPU. 
**Stream-K** mitigates this by evenly distributing the total iterations (across the M, N, *and* K dimensions) among all available CUs. While this provides near-perfect load balancing, it complicates the kernel due to the need for partial tile reductions and cross-CU synchronization, which alters the kernel's register pressure and shared memory (LDS) usage profiles.

### The Role of `kpack` and `matrix_instr_nonkdim`
Before this PR, the `stream-k` implementation might have used suboptimal, hardcoded heuristics for LDS reads and MFMA shapes.

1. **`kpack` (LDS Vectorization):**
   - The `kpack` parameter explicitly controls the ratio between `kWidth` (the block width in the K dimension loaded per iteration) and `k_base` (the base K dimension required by the instruction layout).
   - Tuning `kpack` allows the compiler to generate wider, vectorized LDS read instructions (e.g., `ds_read_b128`). For lower precision data types (FP16/FP8), packing multiple elements into a single 128-bit load significantly improves LDS bandwidth utilization and reduces the instruction count in inner loops.

2. **`matrix_instr_nonkdim` (MFMA Instruction Shape Tuning):**
   - AMD CDNA hardware provides multiple MFMA instruction variants for a single data type (e.g., `16x16` vs `32x32` output blocks).
   - This parameter controls the non-K dimensions of the chosen MFMA instruction. A larger shape (`32x32`) maximizes compute density but dramatically increases the Vector General Purpose Register (VGPR) footprint. A smaller shape (`16x16`) reduces VGPR pressure, potentially allowing more concurrent wavefronts (higher occupancy) to better hide memory latencies.

### The Optimization
By introducing both `kpack` and `matrix_instr_nonkdim` into the Stream-K tuning space, the PR allows the Triton autotuner to evaluate the multi-dimensional trade-off:
- **Stream-K + High Register Pressure:** Stream-K's fixup and reduction phases add register overhead. The autotuner can now offset this by choosing `matrix_instr_nonkdim = 16`, reclaiming VGPRs to maintain healthy occupancy.
- **Memory vs. Compute Balance:** Stream-K introduces different memory access patterns compared to standard tiling. By tuning `kpack`, the compiler can guarantee optimal LDS-to-VGPR bandwidth, ensuring that the MFMA instructions are not starved for data, which is critical for CDNA architectures where compute throughput far outstrips memory bandwidth.

## Optimization Techniques Applied

- **Vectorized Loads:** Utilizing `kpack` to force wider memory reads from LDS to VGPRs, minimizing the total number of load instructions and maximizing bandwidth.
- **Occupancy Tuning:** Tuning `matrix_instr_nonkdim` to control VGPR allocation per thread, balancing compute intensity against the ability to hide latency via active wavefronts.
- **MFMA Scheduling:** Providing the compiler with the ideal instruction shapes and data chunks to efficiently interleave compute and memory operations.

> [!TIP]
> **Tuning Stream-K on CDNA Architectures**
> When dealing with performance regressions or low occupancy in Stream-K kernels on CDNA2/CDNA3/CDNA4 hardware, ensure that `kpack` is sufficiently large (typically leading to 128-bit vectorized loads) and experiment with smaller `matrix_instr_nonkdim` values to alleviate register pressure.

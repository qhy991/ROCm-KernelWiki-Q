---
id: technique-pr-triton-470
title: "PR Insight: triton #470 - Add matrix_instr_nonkdim to Tuning Space"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
hardware_features:
  - mfma
  - wavefront
tags:
  - optimization
  - rocm-kernel
  - compute
  - vgpr
  - occupancy
  - tiling
techniques:
  - occupancy-tuning
  - mfma-scheduling
kernel_types:
  - gemm
languages:
  - triton-rocm
confidence: inferred
sources:
  - pr-triton-470
---

# Analysis of PR #470 in ROCm/triton

## Summary
PR #470 introduces the `matrix_instr_nonkdim` parameter to the Triton GEMM tuning space (specifically targeting ROCm backend auto-tuning scripts, e.g., GEMM Tuning Script v3.1). By exposing the non-K dimension of the Matrix Fused Multiply-Add (MFMA) instructions to the autotuner, the PR enables systematic exploration of different MFMA shapes for a given set of tile dimensions. This enables better balancing of VGPR utilization, wavefront occupancy, and instruction-level concurrency, ultimately improving performance for GEMM kernels on AMD CDNA architectures.

## Architectural and Code Analysis

### Background: MFMA Instructions in AMD CDNA
Unlike basic vector ALUs, AMD's CDNA architectures accelerate matrix multiplication through specialized MFMA (Matrix Fused Multiply-Add) instructions. For a given data type combination (e.g., FP16 inputs, FP32 accumulation), the CDNA ISA provides multiple MFMA variations that compute different block sizes.
For instance:
- `v_mfma_f32_32x32x8f16`: Computes a 32x32 output block, 8 dot products at a time.
- `v_mfma_f32_16x16x16f16`: Computes a 16x16 output block, 16 dot products at a time.

Here, the "non-K dimension" refers to the M and N spatial dimensions of the instruction (e.g., 32 or 16). 

### The Problem
Previously, Triton's ROCm backend heuristics might rigidly select the largest applicable MFMA instruction shape that divides the `BLOCK_SIZE_M` and `BLOCK_SIZE_N` dimensions. While larger instructions (like 32x32) issue more work per instruction, they also consume significantly more Vector General Purpose Registers (VGPRs) per thread to hold the large accumulator fragments. 
High VGPR consumption can restrict the number of concurrent wavefronts that can reside on a Compute Unit (CU), thereby lowering occupancy. If occupancy is too low, the scheduler cannot effectively hide memory latencies (global memory loads and LDS synchronization).

### The Optimization: Adding `matrix_instr_nonkdim` to Tuning Space
By explicitly including `matrix_instr_nonkdim` in the search grid alongside standard tuning parameters (like `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `waves_per_eu`, `num_warps`, and `num_stages`), the autotuner can evaluate trade-offs directly:

1. **VGPR vs. Occupancy Trade-off**:
   - Selecting `matrix_instr_nonkdim = 16` might reduce the register footprint per thread, allowing `waves_per_eu` to be configured higher. Higher occupancy can lead to better performance for memory-bound shapes by allowing more overlapping of compute and memory fetch operations.
   - Selecting `matrix_instr_nonkdim = 32` maximizes compute per instruction issue, potentially performing better for compute-bound scenarios where memory latency hiding is less of a bottleneck.

2. **Finer-Grained Instruction Scheduling**:
   - Smaller MFMA instructions execute with lower latencies per instruction. This allows the compiler to interleave global memory loads, LDS reads, and MFMA instructions more finely, which can be critical when tuning pipeline stages (`num_stages`).

3. **LDS Bank Conflicts and Layouts**:
   - The non-K dimension also dictates the data layout requirements from the Local Data Share (LDS). Changing the instruction shape might change how operands are fetched from LDS, indirectly influencing bank conflict patterns. The autotuner can discover the combination of block sizes and MFMA instructions that minimizes LDS access bottlenecks.

## Optimization Techniques Applied

- **Occupancy Tuning**: By varying register pressure via different MFMA instruction sizes, the autotuner can optimize the ratio of active wavefronts, directly affecting latency hiding capability.
- **MFMA Scheduling Strategy**: Altering the basic compute quantum changes the dynamic behavior of instruction interleaving (compute vs. asynchronous memory copy).
- **Tiling**: Integrates deeply with standard block-level tiling (`BLOCK_SIZE_M/N/K`) to select the most efficient hardware primitive for processing each tile.

## Usage in Triton
When using the updated GEMM tuning scripts, users should expect the tuning grid to include combinations such as:
```python
# Example of tuning space expansion
triton.Config({
    'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
    'matrix_instr_nonkdim': 16, # Added parameter
    'waves_per_eu': 2, 
    ...
})
```
The autotuner will automatically benchmark multiple variations and compile the kernel using the non-K dimension that yields the highest TFLOPS.

> [!TIP]
> **Performance Tuning Advice**
> If a GEMM configuration is hitting register spilling limits or achieving poor occupancy on CDNA2/CDNA3, consider restricting the tuning search space to smaller `matrix_instr_nonkdim` values (e.g., 16 instead of 32) to free up VGPRs and potentially improve memory latency hiding.

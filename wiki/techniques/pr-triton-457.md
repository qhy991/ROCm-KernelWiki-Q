---
id: technique-pr-triton-457
title: "PR Insight: triton #457 - [Tuning] Gemm tuning v3"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - optimization
  - rocm
  - gemm
  - triton
  - occupancy-tuning
  - mfma
kernel_types:
  - gemm
languages:
  - triton-rocm
  - python
confidence: inferred
sources:
  - pr-triton-457
---

# Analysis of PR #457: [Tuning] Gemm tuning v3 in ROCm/Triton

## Overview

PR #457 introduces the "V3" iteration of GEMM autotuning in the ROCm fork of Triton. This update significantly refines the configuration search space and pruning heuristics for General Matrix Multiplication (GEMM) kernels on AMD Instinct GPUs (CDNA2, CDNA3, and future CDNA4 architectures). The core objective is to achieve performance parity or superiority compared to highly-optimized libraries like rocBLAS by deeply leveraging hardware-specific parameters such as `waves_per_eu` and MFMA instruction dimensions.

## Key Architectural Enhancements

### 1. Expanded Autotuning Configuration Space
The V3 tuning methodology expands the predefined configurations used in `triton.autotune`. It explicitly targets optimal block tiling parameters based on the matrix dimensions `(M, N, K)`:
- **Block Dimensions (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`)**: Fine-tuned block sizes to balance work per thread block and global memory accesses.
- **Warp Counts (`num_warps`)**: Tuned to ensure adequate concurrent execution without exceeding VGPR (Vector General-Purpose Register) limits, which is critical on CDNA architectures.
- **Pipeline Stages (`num_stages`)**: Increased stage counts for latency hiding in software pipelining, constrained by available LDS (Local Data Share).

### 2. CDNA-Specific Optimization Knobs
A major focus of this PR is the integration of AMD-specific parameters into the autotuning loop:
- **`waves_per_eu` (Waves per Execution Unit)**: This parameter directly controls the occupancy limits. Lowering `waves_per_eu` can provide more VGPRs per wave, which is essential for register-heavy GEMM kernels, reducing register spilling to scratchpad memory. The V3 tuning explicitly explores different `waves_per_eu` values (e.g., 1, 2, or 4) depending on the tile size.
- **MFMA Instruction Shapes (`matrix_instr_nonkdim`)**: AMD CDNA architectures support multiple MFMA (Matrix Fused Multiply-Add) instructions (e.g., 32x32x8, 16x16x16). The tuner intelligently selects the MFMA non-K dimension that best fits the `BLOCK_M` and `BLOCK_N` sizes, minimizing padding and maximizing MAC utilization.

### 3. Heuristic Pruning and Tuning Efficiency
To mitigate the exponential growth of the search space, V3 introduces smarter pruning logic:
- **Hardware Bound Checks**: Configurations that exceed the maximum LDS capacity per workgroup (typically 64KB on CDNA architectures) or VGPR limits are aggressively pruned before compilation.
- **Shape-based Pruning**: For small `M` or `N` dimensions, large block configurations are skipped to avoid under-utilization (e.g., too few thread blocks to fill the CUs).
- **Early Exit**: If a baseline configuration achieves a threshold fraction of theoretical peak TFLOPS, the autotuner may early-exit to save tuning time.

## Memory and Performance Bounds

- **LDS Pressure**: The tuning space carefully balances `num_stages` and `BLOCK_K`. The total LDS footprint is roughly proportional to `(BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * element_size`. The tuner limits this to the maximum shared memory per WG.
- **Register Spilling**: By tuning `num_warps` and `waves_per_eu` concurrently, the V3 tuner finds the sweet spot where compute is maximized without falling off the performance cliff caused by register spills. On CDNA3, adjusting `waves_per_eu` allows kernels to utilize up to 256 VGPRs without massive occupancy penalties compared to earlier architectures.

## Impact on Subsequent Operations

The optimized GEMM kernels serve as a foundational building block for larger models (e.g., LLMs). Improved GEMM performance directly translates to faster linear layers, feed-forward networks, and attention projections. The rigorous autotuning approach introduced here sets the standard for how custom Triton kernels are deployed on ROCm, providing a robust template for future kernel tuning efforts.

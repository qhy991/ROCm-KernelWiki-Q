---
id: technique-triton-594-gemm-tuning
title: Persistent Autotuning Results in Triton GEMM
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, optimization]
confidence: inferred
---

# Persistent Autotuning Results in Triton GEMM

## Context
Triton provides a powerful autotuning framework to search for optimal kernel configurations (e.g., `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `num_stages`, `num_warps`) for different input shapes. For GEMM operations on AMD ROCm architectures (CDNA2, CDNA3), the design space of these configurations is large, and optimal settings are highly dependent on the matrix dimensions and hardware specifics. The `tune_gemm.py` script is commonly used to sweep this configuration space and benchmark the performance of the generated Triton kernels.

## Intent and Architecture
PR #594 updates `tune_gemm.py` to persist benchmarking results to a file. While seemingly a simple infrastructure change, this has profound architectural implications for kernel optimization workflows:
1. **Offline Analysis**: Writing results to a file allows engineers to plot and analyze the performance landscape offline.
2. **Regression Tracking**: As the ROCm compiler backend or Triton's AMD backend evolves, saved results provide a baseline to identify performance regressions for specific tile sizes.
3. **Continuous Integration (CI)**: CI pipelines can ingest the output file to automatically monitor TFLOPS for key matrix shapes, ensuring that compiler updates do not degrade kernel performance.
4. **Heuristics Generation**: The persistent data can be used to train heuristics or generate static configuration tables, replacing the need for dynamic autotuning in production deployments.

## GEMM Tuning Configuration Space
For ROCm architectures, the GEMM autotuning process typically explores:
- **Matrix Block Sizes**: Determining the optimal `BLOCK_M`, `BLOCK_N`, `BLOCK_K`. Larger blocks increase data reuse and compute-to-memory-access ratios but increase VGPR (Vector General Purpose Register) pressure.
- **Wavefront Distribution (`num_warps`)**: Controls how many wavefronts operate concurrently on the tile. Higher `num_warps` can hide memory latency but may decrease occupancy if register or LDS (Local Data Share) usage is too high.
- **Pipeline Stages (`num_stages`)**: Software pipelining depth. More stages can better hide global memory latency but require more LDS to buffer the tiles. On CDNA3 (MI300X), balancing `num_stages` and LDS usage is critical due to the dual Compute Matrix Accelerator (CMA) design and high-bandwidth memory.

## Performance and Memory Bounds
The benchmarking results saved by `tune_gemm.py` generally reveal the roofline bounds of the GEMM kernels:
- **Compute-Bound Configurations**: For large matrix sizes, the performance is limited by the MFMA (Matrix Fused Multiply-Add) throughput. Autotuning seeks configurations that maximize MFMA utilization.
- **Memory-Bound Configurations**: For smaller batch sizes or extreme aspect ratios (e.g., tall-and-skinny matrices), the kernel becomes memory bandwidth-bound. The tuner will favor configurations that optimize memory coalescing, minimize LDS bank conflicts, and fully utilize the available memory bandwidth.

By analyzing the saved benchmarking files, developers can identify which region of the roofline model a specific configuration falls into and focus their optimization efforts accordingly.

---
id: technique-pr-triton-614
title: "Triton GEMM Autotuning Script (tune_gemm)"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - triton-rocm
  - gemm
  - optimization
  - occupancy
  - rocm
confidence: inferred
sources:
  - pr-triton-614
---

# Triton GEMM Autotuning Script (`tune_gemm`)

## Overview

PR [#614](https://github.com/ROCm/triton/pull/614) migrates the `tune_gemm` (v3.3) script from the `triton-mlir` branch to the `main_perf` branch within the ROCm Triton repository. This script is a critical infrastructure component for performance optimization on AMD Instinct GPUs, enabling automated exploration of Triton kernel configurations for General Matrix Multiply (GEMM) operations.

## Intent and Context

Triton empowers developers to write high-level matrix operations while relying on the compiler and autotuning to find the optimal hardware mapping. However, the performance of a GEMM kernel is heavily dependent on specific launch configurations that dictate how the workload is partitioned across the compute units (CUs). 

The inclusion of the `tune_gemm` script into the main performance branch indicates a shift towards systematic, script-driven autotuning for GEMM kernels to achieve peak performance across different matrix dimensions and CDNA architectural generations (CDNA2, CDNA3, CDNA4).

## Optimization Technique: Autotuning the Search Space

The `tune_gemm` script operates by automatically generating, compiling, and profiling a variety of kernel configurations. The primary parameters explored in this tuning space include:

1. **Tile Sizes (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`):** Determines the dimensions of the sub-matrices loaded into the Local Data Share (LDS) for each thread block.
2. **Number of Warps (`num_warps`):** Determines the size of the thread block. For AMD GPUs, warps (wavefronts) are typically 64 threads.
3. **Number of Pipeline Stages (`num_stages`):** Controls the depth of the software pipeline for asynchronous memory copies from global memory to LDS. 

### Why Tuning is Necessary

Instead of relying on a single heuristic, `tune_gemm` benchmarks multiple valid combinations. Different matrix dimensions have different bottlenecks:
- **Compute-Bound (Large GEMMs):** Performance is limited by the MFMA (Matrix Fused Multiply-Add) units. Larger tiles and more warps increase data reuse and keep the compute units fed.
- **Memory-Bound (Tall-and-Skinny or Short-and-Wide GEMMs):** Performance is limited by Global Memory Bandwidth. Tuning finds the right block size to maximize memory coalescing and pipeline stages to hide latency.

## Memory Bounds and Hardware Constraints

The tuning process inherently balances several competing hardware constraints on CDNA architectures:

- **LDS Usage:** Larger `BLOCK_SIZE_K` and `num_stages` require more LDS to double-buffer (or multi-buffer) the tiles. Exceeding the LDS capacity per CU will reduce occupancy (the number of active thread blocks per CU) or cause the kernel launch to fail.
- **VGPR (Vector General-Purpose Register) Allocation:** Thread-level data, including the accumulators for the MFMA instructions, reside in VGPRs. Excessive VGPR usage per wavefront limits the number of wavefronts that can reside on the CU concurrently. 
- **Occupancy vs. ILP (Instruction-Level Parallelism):** The `tune_gemm` script discovers the optimal tradeoff. Sometimes, a configuration with lower occupancy (fewer active waves) but higher ILP (more work per wave via larger tiles) yields better overall throughput because it maximizes the utilization of the MFMA engines.

## Key Files in the PR

The PR introduces several Python scripts and utilities to manage the tuning lifecycle:
- `tune_gemm.py` / `tune_gemm.sh`: The main entry points for defining the configuration space and executing the search.
- `matmul.py` / `matmul_kernel.py`: The Triton kernel templates for GEMM that are instantiated with the various configurations.
- `rocprof_gemm.py`: Integrates with AMD's `rocprof` tool to accurately measure kernel execution time and extract hardware counters.
- `icache_flush.py`: Ensures consistent benchmarking results by flushing the instruction cache between kernel runs.

## Conclusion

The `tune_gemm` script represents a standard, empirical approach to performance portability in ROCm Triton. By sweeping through various tiling strategies, wavefront counts, and pipeline stages, it systematically discovers the configuration that best balances LDS limits, VGPR pressure, and MFMA utilization for any given GEMM shape on AMD Instinct accelerators.

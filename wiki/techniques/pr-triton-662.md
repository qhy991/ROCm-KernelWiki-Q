---
id: technique-pr-triton-662
title: "Triton GEMM Tuning Configurations for CI Pipeline"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - optimization
  - rocm
  - pipeline
  - occupancy-tuning
confidence: inferred
sources:
  - pr-triton-662
---

# Triton GEMM Tuning Configurations for CI Pipeline

## Architectural Context
The ROCm backend for OpenAI's Triton compiler translates Python-based kernel definitions into optimized AMD GPU ISA. For matrix multiplications (GEMM), achieving peak performance on CDNA hardware (MI200, MI300, and MI350 series) requires carefully navigating a complex search space of tile sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`), wavefront distribution (`num_warps`), and pipeline depths (`num_stages`). 

This page analyzes the architectural implications of introducing weekly tuning configurations into the ROCm Triton CI pipeline, based on PR #662.

## Intent and Auto-Tuning Strategy
The core intent of PR #662 is to integrate structured Auto-Tuning into standard Continuous Integration. The Triton compiler relies on autotuning to dynamically select the most performant kernel configuration for a given input shape and hardware target. 

Without exhaustive tuning configs, the compiler may fall back to suboptimal heuristics, leading to underutilization of the underlying `mfma` (Matrix Fused Multiply-Add) instructions and degraded data locality in the Local Data Share (LDS). By maintaining these tuning configs in the CI:
1. **Performance Regression Tracking:** AMD engineers can systematically track standard GEMM dimensions week-over-week against established baselines on `cdna2`, `cdna3`, and `cdna4`.
2. **Search Space Pruning:** The configs define realistic bounds for `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`, along with reasonable limits on `num_warps` and `num_stages` to ensure high occupancy without VGPR spilling.

## Memory Bounds and Occupancy Tuning
GEMM operations generally transition between memory-bound and compute-bound regimes depending on the matrix dimensions and hardware characteristics.

### Compute-Bound Considerations
For large square matrices, the execution is purely compute-bound. The tuning configurations prioritize configurations that maximize `mfma` instruction throughput:
- **Large Block Sizes:** High `BLOCK_M` and `BLOCK_N` (e.g., 128x128 or 256x128) maximize data reuse from LDS to VGPRs.
- **Deep Pipelining:** Higher `num_stages` (e.g., 3-5) are used to completely overlap asynchronous global memory loads to LDS with MFMA computations. On CDNA architectures, this effectively utilizes async copy instructions and hides HBM latency.

### Memory-Bound Considerations
For "skinny" matrices (where one dimension is extremely small, often seen in decoding phases of LLMs), the kernels become memory bandwidth-bound. The configurations adjust accordingly:
- **Smaller Tile Sizes:** Utilizing smaller blocks reduces LDS footprint, which allows more workgroups to run concurrently on a Compute Unit (CU) (improving theoretical occupancy).
- **Fewer Stages:** With less compute to hide the memory latency, having deep pipelines wastes LDS memory. Lowering `num_stages` prevents exceeding the LDS capacity per CU (which limits occupancy).

## Hardware Synergy
The ROCm tuning configurations implicitly map high-level Triton parameters to AMD-specific hardware features:
- **Wavefront Distribution:** The `num_warps` parameter defines how the thread block is partitioned. In CDNA, a wavefront is 64 threads (`wavefront`). The tuning configs ensure optimal distribution to balance VGPR allocation.
- **LDS Bank Conflicts:** Although handled by Triton's compiler backend, tuning block dimensions indirectly impacts the memory layout. Configs are tuned to dimensions that minimize bank conflicts during LDS reads.
- **VGPR Usage:** The configurations effectively act as an **occupancy tuning** mechanism. They balance the trade-off between keeping data in registers (which requires many VGPRs and reduces the number of concurrent waves) versus spilling or redundantly loading data.

## Continuous Integration Pipeline
The weekly tuning CI operates against these reference configs to run nightly or weekly sweeps. This guarantees that any changes to the LLVM IR generation or MLIR lowering passes in Triton do not inadvertently degrade standard workload throughput. The resulting tuned binaries ensure that users pulling the latest `triton-rocm` packages have out-of-the-box optimized performance for foundational AI shapes.

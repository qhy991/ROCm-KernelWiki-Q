---
id: technique-pr-triton-658
title: "Triton GEMM Tuning: Software Pipelining Minimum Stages"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - triton-rocm
  - gemm
  - double-buffering
  - optimization
  - memory-bound
  - pipeline
  - scheduling
  - rocm-kernel
confidence: inferred
sources:
  - pr-triton-658
---

# Triton GEMM Tuning: Software Pipelining Minimum Stages

## Context

In Triton, high-performance general matrix multiplication (GEMM) kernels rely heavily on **software pipelining** to overlap global memory fetches (from HBM to LDS) with matrix core arithmetic computations (MFMA). This pipelining is controlled by the `num_stages` parameter, which determines the number of pipeline buffers allocated in shared memory.

Prior to [triton PR #658](https://github.com/ROCm/triton/pull/658), the automated tuning search space for GEMM kernels (`tune_gemm`) incorrectly allowed `num_stages=0` to be evaluated. This configuration disables the asynchronous memory pipeline, leading to severe performance regressions because data movement and computation become strictly serialized.

## Technique: Enforcing Minimum `num_stages` for Pipelining

The PR corrects the autotuner search space to enforce a minimum of `num_stages=2` (which corresponds to **double-buffering**) rather than `num_stages=0`. This aligns the GEMM tuner with fixes previously applied to other performance kernels in the Triton ROCm backend.

### Why `num_stages=0` is Problematic
1. **Serialization**: Without multiple stages, the kernel issues a global load for a tile, waits for the data to arrive in LDS, executes the MFMA instructions, and only then issues the next load. This exposes the full latency of HBM.
2. **Resource Underutilization**: Matrix cores (MFMA units) stall waiting for data, while memory buses remain idle during the compute phase. 
3. **Autotuner Waste**: Including `num_stages=0` in the autotuner search space forces the compiler to waste time evaluating and benchmarking a fundamentally suboptimal configuration.

### Double Buffering (`num_stages=2`)
Setting `num_stages=2` allocates two buffers in LDS, establishing a basic pipeline:
- **Stage 0**: Loads tile $i$ from HBM to LDS buffer A.
- **Stage 1 (Steady State)**: Computes on tile $i$ from buffer A, while simultaneously issuing asynchronous loads for tile $i+1$ from HBM into LDS buffer B.

By evaluating tuning configurations starting at `num_stages=2` and extending to higher stages (e.g., 3, 4, or 5 depending on available LDS and VGPRs), the autotuner focuses exclusively on valid, latency-hiding pipeline configurations.

## Performance and Memory Bounds

- **Compute vs. Memory Bound**: Without pipelining (`num_stages < 2`), a kernel that should be compute-bound becomes artificially latency-bound. The HBM latency directly limits the throughput. By using multi-stage pipelining (`num_stages >= 2`), the kernel overlaps memory operations with math and pushes toward the theoretical compute bounds of the MFMA units.
- **LDS (Shared Memory) Capacity**: Increasing `num_stages` consumes more LDS capacity. `num_stages=2` is often the "sweet spot" for very large tile sizes where higher stages would cause LDS allocation to exceed the hardware limits per Workgroup, forcing occupancy to drop. By correcting the tuner space, Triton accurately explores the tradeoffs between `num_stages`, tile sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`), and wave occupancy.
- **Register Pressure**: While more stages hide memory latency, they also consume more VGPRs for tracking pointers, loop state, and asynchronous load tokens. The autotuner must balance this. Removing `num_stages=0` avoids a spurious minimum-VGPR edge case that never achieves high overall throughput.

## Architectural Relevance (CDNA)

On AMD CDNA architectures (CDNA2, CDNA3, CDNA4), the massive gap between HBM bandwidth and matrix core (MFMA) throughput requires deep software pipelining. CDNA utilizes dual-issue execution (VALU and scalar/memory ops) and relies on asynchronous `global_load` instructions (`global_load_dwordx4`, etc.) into LDS to keep the compute units saturated. Enforcing correct `num_stages` bounds ensures that Triton-generated GEMMs efficiently map to the latency-hiding mechanisms of MI200, MI300, and subsequent accelerators.

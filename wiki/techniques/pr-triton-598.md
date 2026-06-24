---
id: wiki-technique-pr-triton-598
title: "Stream-K Auto-Tuning and Correctness Testing in Triton"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, optimization, scheduling, occupancy]
confidence: inferred
---

# Stream-K Auto-Tuning and Correctness Testing in Triton

This page provides a deep architectural analysis of the optimizations and testing infrastructure related to Triton's Stream-K implementation for AMD ROCm GPUs, inferred from [PR #598 in ROCm/triton](../../sources/prs/triton/PR-598.md).

## Executive Summary

The PR addresses an issue within the `tune_streamk` mechanism, specifically fixing the usage of `test_correctness`. Ensuring robust validation is essential for Stream-K execution models, as these models deliberately bypass standard tile-to-Compute Unit (CU) mappings to distribute perfectly balanced workloads. These fractional workloads require global accumulations that are inherently sensitive to testing methodology.

## Architectural Context: The Need for Stream-K

Typical tile-based GEMM (General Matrix Multiply) kernels statically assign fixed output tiles to specific CUs. For matrix dimensions that do not perfectly divide into the total number of available CUs (or the maximum number of waves the GPU can hold), this results in "tail waves"—some CUs complete their work earlier than others, leaving the GPU underutilized at the end of the compute phase.

The **Stream-K execution model** resolves this structural inefficiency:
1. **Fractional Workloads**: Instead of computing a discrete $M \times N$ output tile, each CU processes a contiguous linear chunk of the total compute space (the $K$ dimension multiplied by the output matrix dimensions). 
2. **Global Reductions**: Since a single output tile's calculations might be split across multiple CUs, Stream-K requires a final reduction step. This is often handled through global memory atomic operations or a separate reduction kernel pass.

### Performance and Occupancy Tuning

Stream-K aims to maximize **occupancy** by ensuring every CU is active until the entire phase completes. However, to realize these gains, the kernel must be tuned:
- `BLOCK_M`, `BLOCK_N`, and `BLOCK_K` sizes (the sub-tiles processed).
- The tradeoff between standard tile-based execution (minimizing global synchronization) and Stream-K (maximizing CU utilization but paying a penalty for cross-CU reductions).

The `tune_streamk` module automatically explores this parameter space to find the optimal configuration for a given matrix shape and target architecture.

## Deep Dive: The Correctness Testing Dilemma

Correctly validating a Stream-K kernel poses unique challenges that this PR seeks to resolve. 

### 1. Floating-Point Non-Associativity
When work is partitioned linearly using Stream-K, the order in which partial products are accumulated is non-deterministic or varies drastically based on the CU breakdown. In floating-point arithmetic (especially lower-precision formats like `fp16` or `bf16`), $(A + B) + C \neq A + (B + C)$. The `test_correctness` logic must employ appropriate error tolerances (`atol` and `rtol`) that account for these accumulation differences compared to a standard reference implementation (e.g., PyTorch's native `matmul`).

### 2. Tuning Loop Overheads
The auto-tuner generates multiple variations of the kernel. If `test_correctness` is improperly integrated, it might erroneously fail valid configurations due to uninitialized global reduction buffers, or it could crash the test harness. Fixing `test_correctness` ensures the auto-tuner seamlessly skips invalid configurations and accurately benchmarks valid ones.

## Optimization Technique: Robust Auto-Tuning

Robust auto-tuning in Triton requires two coupled phases:
1. **Validation**: The compiled kernel must produce mathematically equivalent results to a known good baseline.
2. **Benchmarking**: Only strictly validated kernels are profiled to measure performance.

By rectifying the correctness check, `tune_streamk` guarantees that the final selected configuration is both high-performance and functionally robust.

> [!TIP]
> **Stream-K Heuristics**
> Stream-K is most beneficial for medium-sized matrices where the number of output tiles is just slightly larger than a multiple of the total available CUs on the GPU (e.g., on an MI300X with 304 CUs). For massive matrices, traditional tile-based scheduling is often sufficient and avoids the latency overheads associated with global reductions.

## Memory Bounds and Hardware Implications

In Stream-K kernels:
- **Compute-Bound Phase**: The initial phase where CUs process their assigned chunks is heavily compute-bound, relying on MFMA matrix core instructions.
- **Memory-Bound Reduction**: The final combination of partial results requires spilling intermediate data to global High Bandwidth Memory (HBM) and potentially using cross-CU atomic adds. 
- **LDS Usage**: Stream-K relies heavily on Local Data Share (LDS) to buffer chunks before they are globally reduced. Efficient use of LDS is critical to ensure high occupancy without spilling to scratch memory.

By ensuring that `tune_streamk` correctly evaluates kernels, the Triton compiler can automatically balance these compute and memory bounds based on the specific architectural traits of CDNA generations.

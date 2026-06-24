---
id: technique-pr-triton-630
title: "XCD-Aware PID Remapping and Profiler Tooling Reversion"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - optimization
  - scheduling
  - rocm
  - cross-cu
confidence: inferred
sources:
  - pr-triton-630
---

# XCD-Aware PID Remapping for Triton GEMM Tuning

## Context and Intent
In Triton PR #630 (`[tune gemm v3.4] Add xcd-based pid remapping and change back to rocprofv1`), the primary goal is to introduce **XCD-based PID (Program ID) remapping** to optimize GEMM tuning on AMD architectures. The PR also addresses tooling instability by reverting to `rocprofv1` due to severe issues encountered with `rocprofv2` during GEMM tuning workflows.

### What is an XCD?
On AMD CDNA3 architecture (e.g., the MI300X), the GPU is composed of multiple Accelerator Compute Dies (XCDs) stacked on top of base dies. For instance, an MI300X contains 8 XCDs. Each XCD has its own local resources, but shares memory with the broader system. Because cross-XCD traffic can introduce latency and contention, spatial and temporal locality in workload mapping is critical for achieving optimal hardware utilization and avoiding interconnect bottlenecks.

## Technique: XCD-Based PID Remapping
In Triton, the PID (Program ID) dictates the block/workgroup index in the underlying GPU execution model. A standard linear PID dispatch might scatter workgroups (which need to access adjacent or overlapping tiles of input matrices) across different XCDs. This destroys cache locality and results in excessive data fetches over the cross-XCD interconnects.

### Optimization Strategy
1. **Workgroup Swizzling / Remapping**: The standard PID is intercepted and remapped via a transformation before being used to compute the block tile indices for the GEMM.
2. **XCD Alignment**: The remapping algorithm structurally ensures that workgroups computing adjacent output tiles—which inherently share the same input matrix rows or columns—are grouped and scheduled onto the same physical XCD.
3. **L2 Cache Hit Rate**: By packing related PIDs into a single XCD, the L2 cache local to that die experiences significantly higher hit rates, avoiding redundant HBM trips.

### Memory Bounds vs. Compute Bounds
While large GEMMs are traditionally compute-bound (bottlenecked by MFMA execution), the sheer scale of the MI300X multi-die architecture means that performance is highly sensitive to memory fetch efficiency. XCD-based PID remapping reduces interconnect bandwidth saturation and maximizes effective local memory bandwidth, keeping the Matrix Cores fed.

## Profiler Reversion (`rocprofv1` vs `rocprofv2`)
The Triton GEMM tuning infrastructure (`tune_gemm.py`) relies heavily on automated profiling to sweep through block sizes, pipeline stages, and warp counts to find the optimal configuration.
- The PR reverts a prior migration to `rocprofv2` (Triton PR #613) due to a critical blocking issue (tracked in internal ticket #228) that disrupted the autotuning pipeline.
- `rocprofv1` is restored as the stable fallback for kernel execution time measurement in the GEMM tuning script, ensuring that users can continue generating optimal kernel variants while the `rocprofv2` instability is addressed upstream.

## Summary for Developers
- **Locality Matters**: When writing Triton kernels targeting CDNA3 (MI300X), implement custom PID swizzling to cluster related work within the boundaries of a single XCD.
- **Profiling Toolchain**: If you experience hangs, missing metrics, or crashes during automated autotuning with `rocprofv2`, fall back to `rocprofv1` for stable kernel execution time collections.

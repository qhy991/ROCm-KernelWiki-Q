---
id: technique-pr-triton-565
title: "Triton Read-Only Benchmark and Command Line Tuning"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, triton-rocm, optimization, memory-bound, bandwidth, hbm, occupancy]
confidence: inferred
sources: [pr-triton-565]
---

# Triton Read-Only Benchmark and Command Line Tuning

## Summary
PR #565 in the `ROCm/triton` repository introduces a read-only microbenchmark capability, coupled with command-line configuration tools. These changes permit developers to specify key kernel configuration parameters at runtime—such as data size, workgroup size (WGS), and block size—which is an essential methodology for analyzing and tuning memory-bound performance on AMD CDNA architectures.

## Intent and Context
The fundamental intent of a "read-only benchmark" in a compiler or kernel repository like Triton is to measure and isolate the **peak memory read bandwidth** of the underlying hardware (e.g., HBM on MI250X or MI300X) without the interference of compute-heavy instructions or write operations. 

By adding command-line arguments to specify `size`, `wgs` (workgroup size), and `block_size`, the author (`vgokhale`) has enabled rapid parameter sweeps. In GPU kernel development, finding the right combination of workgroup size and memory footprint per thread block is crucial for hiding memory latency and achieving maximum occupancy. 

## Technical Analysis

### Memory Bounds and HBM Bandwidth
A read-only kernel is strictly **memory-bound**. The theoretical peak performance is entirely bottlenecked by the High Bandwidth Memory (HBM) subsystem. To saturate the HBM on CDNA architectures:
1. **Sufficient Occupancy**: The workgroup size (`wgs`) must be tuned such that enough waves are in flight to hide the memory access latency.
2. **Coalesced Memory Accesses**: The memory must be loaded using vectorized and contiguous accesses (e.g., 128-bit loads) across the wavefront.
3. **Cache Line Utilization**: The block size dictates the chunk of data loaded per workgroup, which must align well with the L1/L2 cache and memory controller topology.

### Command Line Configuration (Parameter Sweeping)
Hardcoding block sizes and workgroup sizes in Triton kernels limits portability across different generations of hardware (e.g., CDNA2 vs. CDNA3). CDNA3 (MI300) might require a different balance of thread blocks to max out its higher bandwidth compared to CDNA2 (MI250). 

Exposing `size`, `wgs`, and `block_size` to the command line enables:
- **Auto-tuning**: Automated scripts can wrap the benchmark and sweep through grid and block dimensions to empirically discover the configuration that yields the highest bandwidth.
- **Micro-architecture profiling**: Engineers can test specific hypotheses about cache behaviors or TLB misses by granularly altering the access patterns and footprints.

## Optimization Technique: Empirical Tuning
While Triton's autotuner usually handles parameter sweeps (e.g., `triton.autotune`), having a standalone command-line configurable read-only benchmark is invaluable for compiler developers. They can quickly isolate whether a performance regression in a complex kernel (like GEMM or Flash Attention) is due to a fundamental drop in the compiler's ability to generate optimal memory load instructions or something else entirely.

### Hardware Considerations (CDNA2 / CDNA3 / CDNA4)
- **CDNA2 (MI250X)**: Features a dual-GCD design. Sweeping `size` and `block_size` can help identify NUMA-like effects or cross-GCD memory access penalties.
- **CDNA3 (MI300X)**: Introduces a unified memory architecture across APUs (in MI300A) or massive HBM bandwidth in discrete GPUs (MI300X). The optimal `wgs` might be larger to fully utilize the enhanced memory bandwidth.

## References
- [ROCm/triton PR #565](https://github.com/ROCm/triton/pull/565)
- Source Insight Page: [pr-triton-565](pr-triton-565)

---
id: technique-triton-649
title: "Instruction Scheduling Autotuning in Triton GEMM"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, optimization, scheduling, mfma-scheduling, mfma, gemm, triton-rocm]
confidence: inferred
sources: [pr-triton-649]
---

# Instruction Scheduling Autotuning in Triton GEMM

> [!NOTE]
> This page analyzes the architectural impact of introducing instruction scheduling (`instr.sched`) options to the Triton autotuner (`tune_gemm.py`), as introduced in ROCm/triton PR #649.

## Context and Intent

Instruction scheduling dictates the exact order and interleaving of memory operations (Global/LDS loads and stores) and compute operations (`mfma` instructions) at the compiler level. In the context of Triton for AMD GPUs, default scheduling heuristics may not always extract maximum performance because of the complex interplay between register pressure, pipeline latencies, and instruction issue limits on CDNA architectures.

By adding `instr.sched` options to the `tune_gemm.py` autotuner, the developers expose low-level scheduling strategies (e.g., prefetching depth, load/compute grouping) as hyperparameter tuning dimensions. This allows the autotuner to empirically discover the optimal sequence tailored to specific GEMM shapes, matrix layouts, and target architectures (CDNA2, CDNA3, CDNA4), rather than relying on a one-size-fits-all compiler pass.

## Architectural Optimizations 

### Latency Hiding via `mfma-scheduling`

> [!TIP]
> Optimal performance on CDNA matrix cores requires a continuous, uninterrupted stream of data from the Local Data Share (LDS) and VGPRs into the `mfma` arithmetic units.

Memory operations have significantly higher latencies compared to ALU operations. To achieve peak TFLOPS, the compiler must effectively hide these latencies through:
- **Software Pipelining:** Sinking compute instructions and hoisting memory reads so that `ds_read` (LDS load) instructions execute well before their dependent `v_mfma` instructions.
- **Grouped Issuing:** Avoiding instruction-level pipeline stalls by clustering independent memory loads and scheduling them simultaneously.

By tuning the instruction schedule, the autotuner explores different interleaving patterns, discovering the right balance of memory prefetching to prevent the Matrix Fused Multiply-Add (`mfma`) units from starving.

### Register Allocation & Occupancy Trade-offs

Different instruction schedules inherently alter the liveness of variables in Vector General Purpose Registers (VGPRs).

1. **Aggressive Prefetching:** Schedules that prefetch data early keep VGPRs live longer. This increases register pressure, potentially reducing the maximum wavefront occupancy per Compute Unit (CU).
2. **Conservative Prefetching:** Minimizes register pressure (potentially improving occupancy) but risks failing to hide memory latency completely.

Exposing `instr.sched` enables the autotuner to navigate this precise Pareto front. For certain GEMM tiles, higher occupancy is less critical than aggressive latency hiding, and vice versa.

## Memory Bound vs. Compute Bound Dynamics

GEMM workloads span a continuum depending on their arithmetic intensity (e.g., tall-and-skinny vs. large square matrices). Instruction scheduling adapts to these bounds:

> [!IMPORTANT]
> **Compute-Bound:** The autotuner favors schedules that pipeline LDS loads sufficiently far ahead of the `mfma` blocks. Even if this spikes register usage, keeping the matrix cores perfectly fed is prioritized over maximizing wave occupancy.

> [!IMPORTANT]
> **Memory-Bound:** When constrained by HBM or LDS bandwidth, optimal scheduling staggers memory requests to prevent sudden stalls and memory subsystem congestion. Spreading out global loads (`global_load_dwordx4`) and LDS writes ensures smoother throughput and maximizes the L2/L1 cache hierarchy's hit rate.

## Summary

The integration of `instr.sched` into `tune_gemm.py` elevates scheduling from a static compiler phase to a tunable dimension in the Triton search space. This capability is pivotal for bridging the performance gap between auto-generated Triton kernels and hand-written assembly libraries like `composable_kernel` or `hipBLASLt`.

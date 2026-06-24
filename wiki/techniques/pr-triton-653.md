---
id: technique-pr-triton-653
title: "Stream Pipeliner v2 Requirements: Enforcing num_stages > 0 in Triton"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - optimization
  - pipeline
  - scheduling
  - double-buffering
  - memory-bound
languages:
  - triton-rocm
confidence: inferred
sources:
  - pr-triton-653
---

# Stream Pipeliner v2 Requirements: Enforcing `num_stages > 0`

## Overview

In the [ROCm/triton PR #653](https://github.com/ROCm/triton/pull/653), performance regression tests were updated to increase `num_stages` from `0` to `2`. This change highlights a critical requirement in the modern Triton compilation pipeline for AMD ROCm: the **stream pipeliner v2** infrastructure no longer supports or expects `num_stages == 0`.

## Architectural Context and Intent

Triton employs software pipelining to hide global memory latency when fetching tile data into LDS (Local Data Share). This is essential for maximizing MFMA (Matrix Fused Multiply-Add) compute throughput on CDNA architectures. 

The `num_stages` parameter in a Triton kernel configuration specifies the depth of the software pipeline (i.e., how many stages or multi-buffers are maintained in LDS). 

### Why `num_stages == 0` Fails with Pipeliner v2

1. **Scheduling Assumptions**: The new stream pipeliner v2 explicitly constructs schedules to overlap asynchronous memory operations (global-to-shared copies) with computation. A `num_stages` of 0 indicates an un-pipelined or purely synchronous execution model with no prefetching buffers. Stream pipeliner v2 expects a structural multi-stage loop where at least the current tile and the next tile can be managed.
2. **Buffer Allocation**: Pipelining inherently requires LDS buffer allocation for multiple stages. Setting `num_stages=0` breaks the internal buffer lifecycle logic within the MLIR-based stream pipeliner, resulting in compilation failures or semantic errors during loop unrolling and software pipelining passes.

## Optimization Technique: Software Pipelining

Software pipelining transforms a standard sequential loop into an interleaved sequence of operations across different iterations. 

By setting `num_stages = 2`, the kernel implements a classic **double-buffering** scheme:
- **Stage 0**: Loads tile $i+1$ into LDS buffer A, while computing MFMA operations on tile $i$ located in LDS buffer B.
- **Stage 1**: Loads tile $i+2$ into LDS buffer B, while computing MFMA operations on tile $i+1$ located in LDS buffer A.

Higher values of `num_stages` (e.g., 3 or 4) can further hide memory latency for memory-bound or highly complex kernels, at the expense of increased LDS pressure and potentially lower occupancy.

## Memory Bounds and Occupancy Trade-offs

When migrating or tuning kernels with stream pipeliner v2:
- **Minimum Value**: `num_stages` must be set to at least `2` (or a valid positive integer supported by the pipeline) to enable meaningful overlapping of compute and memory in pipeliner v2.
- **LDS Pressure**: Each stage allocates memory in LDS. Increasing `num_stages` linearly increases the shared memory footprint of a threadblock (workgroup).
- **Occupancy Constraints**: If the combined size of the multi-buffered tiles exceeds the available LDS per CU (Compute Unit), the kernel will either fail to compile or suffer from reduced occupancy (fewer active wavefronts per CU), potentially offsetting the latency-hiding benefits.

## Best Practices
- Always explicitly define `num_stages >= 2` for Triton kernels that perform heavy matrix multiplications or reductions requiring memory latency hiding.
- Avoid relying on `num_stages=0` to disable pipelining during profiling or testing if using the stream pipeliner v2 backend, as it may result in broken execution graphs.

---
id: technique-pr-triton-655
title: "Software Pipelining: Enforcing Double Buffering with num_stages=2 in Triton"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - double-buffering
  - async-copy
  - mfma-scheduling
  - occupancy-tuning
  - lds
  - mfma
  - triton-rocm
  - optimization
  - memory-bound
confidence: inferred
sources:
  - pr-triton-655
---

# Software Pipelining and Double Buffering in Triton (num_stages=2)

## Overview
In Triton PR #655 for the AMD ROCm backend, test configurations and kernel definitions were explicitly updated to use `num_stages=2` instead of `num_stages=0`. In the Triton programming model, the `num_stages` parameter controls the depth of the software pipeline, essentially dictating how many buffers are allocated in shared memory (LDS on AMD) to overlap global memory operations with compute.

## Technical Intent & Optimization Techniques

### 1. Enabling Software Pipelining (`double-buffering`)
The `num_stages` parameter is a critical optimization knob for compiling high-performance Triton kernels:
- **`num_stages=0` or `num_stages=1`**: Disables software pipelining. The kernel executes synchronously: it issues global memory loads, waits for the loads to complete, and then performs computation. This sequential execution leads to idle Compute Units (CUs) during memory fetches and underutilized memory bandwidth during compute.
- **`num_stages=2` (Double Buffering)**: Allocates two separate buffers in the Local Data Share (LDS). This allows the Triton compiler to emit asynchronous loads (`async-copy`) to fetch data into the second buffer while the `mfma` (Matrix Fused Multiply-Add) instructions are simultaneously computing on data residing in the first buffer.

### 2. LDS Capacity and Occupancy Tuning
While higher pipeline depths (e.g., `num_stages=3` or `4`) can hide more memory latency by queueing multiple outstanding loads, they linearly increase the LDS memory footprint per thread block. 
- On AMD CDNA architectures (CDNA2, CDNA3), LDS capacity per CU is heavily contended. Exceeding certain LDS thresholds will force a reduction in the number of concurrent waves scheduled per CU (`occupancy-tuning`), which can harm overall latency hiding. 
- Setting `num_stages=2` provides the foundational latency-hiding capability (overlapping memory with compute) while keeping the LDS footprint conservative. This prevents register/LDS spilling, ensures kernels compile successfully in CI, and maintains high occupancy across various kernel shapes.

### 3. MFMA Scheduling and Async-Copy
Enforcing a non-zero stage count enables the AMD Triton compiler backend to properly interleave `v_mfma` instructions with global-to-LDS memory transfers. 
- During a matrix multiplication loop, the memory instructions for tile $N+1$ are issued concurrently with the matrix core instructions for tile $N$. 
- This is critical for both **compute-bound** kernels (where the matrix cores must be kept fed at all times) and **memory-bound** kernels (where the HBM bandwidth must be constantly saturated).

## Memory Bounds & Performance Implications
By replacing `num_stages=0` with `num_stages=2`, the memory bounds of the kernels are effectively mitigated:
- **Latency Hiding**: The stall time waiting for HBM reads is absorbed by the math instructions of the previous tile.
- **Predictable Performance**: `num_stages=0` might act as an uninitialized state or an unsupported pipeline configuration in the ROCm compiler path, leading to sub-optimal fallback code generation. Explicitly enforcing `2` guarantees that a predictable double-buffered execution model is employed, avoiding regressions in CI testing.


---
id: technique-pr-triton-660
title: "Stream-K GEMM Implementation (v0.3) in Triton"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - persistent-kernel
  - scheduling
  - optimization
  - memory-bound
  - rocm-kernel
confidence: inferred
sources:
  - pr-triton-660
---

# Stream-K GEMM Implementation (v0.3) in Triton

This page provides an architectural and code analysis of the Stream-K work-scheduling technique for GEMM kernels in AMD's Triton fork, based on the changes introduced in PR #660 (`Streamk v0.3`).

## Context and Intent

Traditional GEMM kernels partition the output matrix into tiles that are mapped directly to thread blocks (or Workgroups in ROCm). While this static mapping is straightforward, it suffers from **wave quantization (or tail effects)**. On large GPUs like the MI300X (with up to 304 Compute Units), if the number of output tiles is not a perfect multiple of the CU count, a significant portion of the GPU will remain idle during the final wave of execution. 

The intent of **Stream-K** is to treat the entire GEMM computational workload (all MAC operations across the $M, N$, and $K$ dimensions) as a continuous 1D "stream" of work. This stream is then divided evenly among all available CUs. By decoupling the work scheduling from the physical dimensions of the output matrix, Stream-K ensures near-perfect compute utilization across arbitrary matrix shapes.

## Architectural Analysis & Optimization Techniques

### 1. Persistent Kernel & Work Distribution
Stream-K fundamentally relies on a **persistent-kernel** execution model. 
- Instead of using hardware-based multi-dimensional grid launches (`tl.program_id(0)`, `tl.program_id(1)`), the Stream-K kernel calculates its work assignment dynamically.
- Each program instance computes a global iteration range it is responsible for. It may start computing part of the $K$-dimension for one $(M, N)$ tile, finish it, and seamlessly transition into the next $(M, N)$ tile within the same execution thread.

### 2. Handling the K-Dimension Splitting
A key consequence of uniform work distribution is that an output tile might be computed by multiple CUs. 
- **Full Tiles:** Most CUs will process entire K-dimensions for their assigned $(M, N)$ tiles.
- **Partial Tiles:** The boundaries of the work stream will inevitably cut through the middle of a K-dimension. The first and last tiles processed by a CU might only represent a partial sum.

### 3. Global Memory Reduction (Atomic Adds)
Because partial tiles are computed by different CUs, their results must be safely combined.
- When a CU finishes a partial sum, it cannot write directly to the final output matrix without synchronization.
- **Technique:** The partial results are typically written to a temporary workspace in global memory or accumulated using atomic operations (`tl.atomic_add`).
- **Optimization:** Triton's Stream-K script minimizes the number of atomic adds by classifying work units into "data-parallel" (no overlap) and "split-k" (overlap) phases. Atomic reduction is only paid for the boundary tiles.

## Performance and Memory Bounds

- **Compute Utilization (Pros):** Stream-K drastically improves the TFLOPS achieved on non-square, irregular, or "medium-sized" GEMM shapes (common in LLM inference, e.g., decode phase or prompt processing with odd sequence lengths). CU occupancy stays uniformly high across all waves.
- **Memory Bandwidth (Cons):** The technique inherently increases HBM traffic, moving the kernel closer to becoming **memory-bound**. The atomic reductions of partial K-sums require additional global memory reads/writes. Thus, for shapes that perfectly align with the CU count, a standard tiled GEMM often outperforms Stream-K due to zero reduction overhead. 
- **Register Pressure:** Fused Stream-K logic can increase the VGPR footprint of the kernel since the loop infrastructure has to track global stream indices, tile coordinates, and boundary conditions concurrently. Proper `tl.constexpr` usage is required to avoid spilling registers to scratch memory.

## Implementation Considerations in Triton

Writing Stream-K in `triton-rocm` requires explicitly managing coordinates that are traditionally handled by the hardware grid launcher:
1. **Linearization of Work:** `total_work = (M / BLOCK_M) * (N / BLOCK_N) * (K / BLOCK_K)`
2. **Chunking:** `work_per_program = total_work / num_programs`
3. **Index Translation:** Inside the Triton loop, translating the linear stream index back into 3D block coordinates (`m_idx, n_idx, k_idx`).
4. **Boundary Checks:** Triton's `tl.where` and masking are heavily utilized to prevent out-of-bounds memory accesses when transitioning across tile boundaries.

## Summary

The v0.3 Stream-K GEMM script in Triton represents a critical step in standardizing advanced software-based scheduling for ROCm. It bridges the performance gap for oddly shaped GEMMs by trading a slight increase in HBM traffic for guaranteed, full-device compute utilization, making it an essential technique for dynamic deep learning workloads.

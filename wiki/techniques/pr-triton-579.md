---
id: pr-triton-579
title: "Deep Analysis: Stream-K Kernel Implementations in ROCm Triton"
author: xiaohuguo2023
date: '2024-05-23'
architectures: [cdna2, cdna3, cdna4]
tags: [rocm-kernel, scheduling, optimization, compute]
kernel_types: [gemm]
languages: [triton-rocm]
techniques: [persistent-kernel]
---

# Deep Analysis: Stream-K Kernel Implementations in ROCm Triton

## Overview

PR #579 introduces new Stream-K kernel implementations for Triton on ROCm. Stream-K is a sophisticated work-partitioning algorithm that fundamentally shifts the way GEMM (General Matrix Multiply) operations are distributed across the GPU's Compute Units (CUs). By moving away from traditional spatial tiling approaches to a strictly work-based scheduling model, Stream-K ensures near-perfect load balancing and high occupancy, effectively solving the "wave quantization" or "tail effect" problems commonly seen when tile counts do not perfectly divide by the number of streaming multiprocessors/CUs.

## Architectural Context

Traditional GEMM implementations split the output matrix into discrete spatial tiles (e.g., $128 \times 128$) and assign each tile to a workgroup. On AMD CDNA architectures, a grid configuration that isn't a perfect multiple of the available CUs leads to "tail waves"—where most CUs sit idle while a few finish the remaining tiles. 

Stream-K addresses this by:
1. **Flattening the Problem Space:** The total computational work (measured in iterations or K-slices of tiles) is flattened into a single linear sequence.
2. **Equitable Distribution:** The linear sequence is divided evenly among all available CUs on the GPU. Every CU gets exactly the same amount of work, keeping the entire GPU busy until the very end.
3. **Partial Tiles and Reductions:** Because work is partitioned by arbitrary iteration boundaries, a single spatial tile's K-dimension computation might be split across multiple CUs. This necessitates mechanisms for cross-CU reduction to accumulate the final result for that tile.

## Technical Mechanisms & Optimizations

### Persistent Kernel Architecture
Stream-K inherently relies on a **persistent kernel** model. Instead of relying on the hardware scheduler to dispatch blocks, a fixed number of workgroups (matching the optimal occupancy of the device) are launched. These persistent threads fetch "chunks" of work from the linear pool. This bypasses grid launch overheads and gives the kernel fine-grained control over scheduling.

### Work Partitioning and Execution
The Stream-K algorithm divides work into two distinct phases:
1. **Data Parallel (DP) Tiles:** Tiles that perfectly map to a single CU. The CU computes the entire dot product along the K dimension. These are identical to classical GEMM tiles.
2. **Stream-K (SK) Tiles:** Tiles that straddle the boundaries of CU work allocations. Two or more CUs will compute partial sums for these tiles. 

### Synchronization and Memory Considerations
The primary complexity in Stream-K implementations lies in handling the partial sums of SK tiles:
- **Global Memory Reductions:** When an SK tile is split, the participating CUs write their partial results to a temporary workspace in global memory.
- **Fixup / Accumulation Phase:** A synchronization step is required to combine these partial sums into the final output matrix.
- **Memory-Bound Overheads:** While Stream-K optimizes compute-bound workloads by increasing utilization, the temporary workspace writes for partial tiles can introduce memory bandwidth overhead. The implementation must carefully tune the slice sizes to ensure the compute utilization gains outweigh the global memory reduction costs.

## Performance Bounds and Tuning

- **Compute vs. Memory Bound:** Stream-K shines in compute-bound scenarios where wave quantization drastically reduces overall TFLOPS. For highly memory-bound shapes, the overhead of reading/writing partial sums might negate the load balancing benefits.
- **Tuning Parameters:** In the ROCm Triton stack, parameters like `kpack` and `matrix_instr_nonkdim` are co-developed alongside Stream-K (as seen in parallel PRs) to maximize MFMA instruction throughput during the persistent execution.

## Related Features and Hardware Dependencies
- **Architectures:** Targeted for CDNA architectures (`cdna2`, `cdna3`, `cdna4`) where massive CU counts (e.g., up to 304 on MI300X) make load balancing critical.
- **MFMA Usage:** The underlying math is driven by AMD Matrix Core (`v_mfma`) instructions. Stream-K ensures these matrix cores are kept fed continuously.

## Conclusion

PR #579 lays the groundwork for high-efficiency GEMM scheduling in Triton for AMD GPUs. By implementing Stream-K, it enables developers to achieve peak TFLOPS on arbitrary matrix shapes, breaking the dependency on hardware-friendly dimensions and pushing ROCm's Triton backend closer to optimal utilization on CDNA architectures.

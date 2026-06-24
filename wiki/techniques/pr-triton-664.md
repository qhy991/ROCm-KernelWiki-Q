---
id: technique-triton-streamk-grid
title: "StreamK Grid Prediction Model in Triton"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm-kernel, optimization, scheduling, compute, occupancy]
kernel_types: [gemm]
languages: [triton-rocm]
confidence: inferred
---

# StreamK Grid Prediction Model in Triton

## Context and Intent
The Grid Prediction Model for StreamK addresses a fundamental problem in GPU scheduling for Matrix Multiplication (GEMM): the quantization effect (often referred to as the "tail effect" or "wave quantization"). 

In traditional Data-Parallel (DP) scheduling, the output matrix is divided into independent tiles (e.g., `BLOCK_M` x `BLOCK_N`). Each workgroup computes one or more of these output tiles. However, if the total number of tiles is not an exact multiple of the number of available Compute Units (CUs) on the GPU, some CUs will sit idle during the final wave of execution. On massively parallel architectures like AMD's CDNA2 (MI250X) and CDNA3 (MI300X, which has up to 304 CUs), this underutilization becomes a significant performance bottleneck for non-standard or small problem sizes.

StreamK (a technique originally introduced to solve this) redefines work distribution by dividing the *computational volume* (the total number of iterations in the K-dimension across all tiles) evenly among all participating workgroups. This allows a StreamK kernel to be launched with an arbitrary grid size. 

The intent of PR #664 in ROCm/triton is to introduce a **Grid Prediction Model** that determines the optimal number of workgroups (grid size) to launch for a given problem size, dynamically balancing the trade-off between the overhead of StreamK's cross-workgroup reductions and the increased occupancy it provides.

## Architectural Analysis & Optimization Technique

### The "Tail Effect" vs. StreamK
In a standard GEMM execution, the grid size is simply:
`Grid = (M / BLOCK_M) * (N / BLOCK_N)`

If `Grid` is 300, and the GPU has 304 CUs (MI300X), then 4 CUs are idle. If `Grid` is 400, then the first wave occupies 304 CUs, and the second wave occupies only 96 CUs, leaving 208 CUs idle while waiting for the final wave to complete.

StreamK circumvents this by decoupling the grid size from the problem dimensions. The total work is continuous and linearized:
`Total_Work = M_tiles * N_tiles * K_iterations`

With StreamK, we can launch exactly `N_CUs` (or a multiple thereof) workgroups. Each workgroup computes `Total_Work / N_CUs` iterations. Because workgroups might start or end computation in the middle of an output tile, atomic operations are used to accumulate the final results in memory.

### The Role of the Grid Prediction Model
While StreamK can theoretically use any grid size, launching exactly `1 * CUs` or `2 * CUs` workgroups isn't always optimal. The Grid Prediction Model predicts the *best* number of workgroups to launch by analyzing:
1. **Problem Size (M, N, K):** Is the problem compute-bound or memory-bound? Small GEMMs benefit highly from StreamK, while very large GEMMs naturally hide wave quantization, making standard DP scheduling more efficient due to zero atomic overhead.
2. **Hardware Characteristics (CUs & Occupancy):** The model assesses the available CUs on the target architecture. It also considers the theoretical occupancy limit (how many workgroups can fit on a single CU given VGPR and LDS usage).
3. **Data-Parallel + StreamK Hybrid (DP+SK):** Modern StreamK implementations often split the grid into a DP portion (for tiles that neatly fit into whole waves) and a StreamK portion (for the remainder). The grid model determines how many workgroups to allocate to the DP pool and how many to the StreamK pool.

## Memory Bounds and Performance Trade-offs

- **Memory Bounds:** Traditional DP GEMMs write output tiles to global memory exactly once. StreamK, however, introduces split tiles where multiple workgroups contribute to the same output tile. This requires atomic floating-point additions (or a separate reduction pass in global memory), increasing global memory traffic. The Grid Prediction Model must ensure that the performance gained from higher CU utilization outweighs the penalty of atomic memory transactions.
- **LDS and VGPR Usage:** The grid size prediction is tightly bound to register (VGPR) and Local Data Share (LDS) pressure. If a kernel requires large `BLOCK_M`/`BLOCK_N` tiles, the number of workgroups per CU might be limited to 1 or 2. The model uses this occupancy limit to calculate the exact maximum number of concurrent workgroups.
- **Performance:** For small to medium batch sizes or non-standard GEMM shapes (common in LLM inference, e.g., decoding or short context phases), the prediction model dynamically shifts the kernel to use StreamK, yielding near-perfect scaling across the GPU. For massive shapes where `Grid % CUs == 0`, the model falls back to standard DP, preserving maximum memory throughput.

## Takeaways for ROCm Kernel Developers
- **Dynamic Scheduling:** Statically compiled grid sizes are suboptimal for dynamic workloads. Implementing runtime grid prediction based on hardware telemetry (CU count) and kernel shape ensures robust performance across different CDNA generations.
- **Occupancy-Awareness:** A successful grid model must query device properties (like `hipDeviceAttributeMultiprocessorCount`) and calculate active waves per CU to perfectly size the StreamK grid.
- **Algorithmic Fallbacks:** Always maintain a heuristic threshold. The model allows Triton to gracefully degrade back to standard data-parallel scheduling when the atomic overhead of StreamK would offset the load-balancing benefits.

---
id: technique-pr-triton-619
title: "Stream-K GEMM Work Distribution in Triton"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm-kernel, optimization, scheduling, occupancy, compute]
confidence: inferred
kernel_types: [gemm]
languages: [triton-rocm]
hardware_features: [compute-unit]
sources: [pr-triton-619]
---

# Stream-K GEMM Work Distribution in Triton

## Overview

PR [#619](https://github.com/ROCm/triton/pull/619) introduces **Stream-K v0.1** to the ROCm Triton backend. Stream-K is a work-centric execution model for General Matrix Multiply (GEMM) that dramatically improves hardware utilization (occupancy) by evenly distributing MAC operations across all available Compute Units (CUs), avoiding the "tail quantization" (wave quantization) effect typical in data-parallel tile execution.

## Intent and Architectural Significance

In a standard tile-based GEMM kernel, the output matrix $C$ is divided into discrete tiles. Each thread block (or wave, mapped to a CU) computes exactly one tile of $C$ by multiplying a row panel of $A$ and a column panel of $B$. 

This data-centric approach suffers from **wave quantization**:
- If the number of tiles is not perfectly divisible by the number of CUs (e.g., 130 tiles on a 120-CU CDNA2 GPU like MI250X), the first wave occupies all 120 CUs perfectly. 
- The second wave processes the remaining 10 tiles, leaving 110 CUs idle. 
- This leads to severe underutilization for small-to-medium sized matrices or non-standard shapes.

**Stream-K** shifts the paradigm to **work-centric execution**. Instead of mapping CUs to discrete output tiles, Stream-K computes the total number of block-level inner-loop iterations across the entire GEMM and divides them equally among the available CUs.

> [!NOTE]
> Stream-K introduces a paradigm shift from data-centric to work-centric scheduling, directly targeting hardware utilization over strict spatial output separation.

## Technique Details

### Fractional-Tile Execution

By dividing the total work equally, a CU may start its execution in the middle of computing one output tile, finish it, and then proceed to compute the beginning of the next output tile. This means multiple CUs will contribute to the *same* output tile (fractional tiles).

The execution model in Triton is adapted as follows:
1. **Work Assignment**: The 1D launch grid is sized based on the physical CU count (e.g., `num_programs = num_cus`). Each `program_id` calculates its continuous chunk of inner-loop iterations: `total_iters / num_cus`.
2. **Boundary Crossing**: As the kernel iterates over its assigned chunk, it checks if it has crossed a tile boundary. If it does, it saves the current partial result for the outgoing tile and begins accumulating for the new tile.
3. **Partial Accumulation**: Since an output tile's $K$-dimension might be split across multiple CUs, the CUs cannot directly write the final result to the output matrix. They must either:
   - Use atomic additions in global memory.
   - Write partial sums to a temporary global workspace (HBM), which are then reduced in a separate step or via atomic accumulation.

### ROCm/CDNA Optimization Considerations

For AMD CDNA architectures (CDNA2, CDNA3), applying Stream-K requires careful management of global memory bandwidth, as the partial sums require additional read/write operations to HBM compared to standard GEMM where accumulation happens entirely in VGPRs. 

- **Workspace Management**: The Triton implementation provisions a secondary workspace buffer where partial tiles are staged.
- **Compute vs Memory Bound**: Stream-K is most effective for compute-bound GEMMs with quantization issues. If the GEMM is strictly memory-bound, the overhead of writing and reading partial tiles in HBM can offset the benefits of better CU utilization.

> [!TIP]
> Stream-K is most effective for GEMM sizes that cause wave quantization (e.g., when the number of output tiles is not a multiple of the CU count). For perfectly aligned sizes, standard tile-based GEMM may be slightly faster due to the absence of partial tile reduction overhead.

## Memory and Performance Bounds

### Advantages
- **Optimal Occupancy**: Guarantees that all CUs are utilized equally, achieving near 100% active wave occupancy even for irregular matrix sizes.
- **Predictable Performance**: Execution time becomes linearly proportional to the total FLOPs, removing performance "cliffs" associated with matrix dimensions that are slightly larger than a wave multiple.

### Trade-offs and Memory Implications
- **HBM Traffic Overhead**: Writing partial tiles to global memory consumes extra bandwidth. The number of partial tiles is bounded by the number of CUs. For very small matrices, this overhead is noticeable.
- **Synchronization Overhead**: The cross-CU reduction of partial tiles requires global synchronization or atomic operations. 

> [!WARNING]
> The cross-CU reduction of partial tiles requires additional HBM bandwidth. This technique works best when the kernel is predominantly compute-bound, as the latency of extra memory operations can be hidden.

## Applicability
- Best used for Large Language Model (LLM) inference (e.g., decoding stages) and parameter-efficient fine-tuning (like LoRA) where batch sizes are small and GEMM dimensions do not align well with the massive CU counts of modern accelerators like the MI300X (304 CUs).

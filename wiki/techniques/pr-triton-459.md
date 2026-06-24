---
id: technique-pr-triton-459
title: "Autotuning Flash Attention in ROCm Triton"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, optimization, memory-bound, scheduling]
kernel_types: [flash-attention]
languages: [triton-rocm]
confidence: inferred
sources: [pr-triton-459]
---

# Autotuning Flash Attention in ROCm Triton

## Overview
PR #459 in `ROCm/triton` ("Add autotuning for FA") introduces automated configuration tuning (autotuning) for Flash Attention (FA) kernels within the `perf-kernels` suite of the ROCm Triton backend. Flash Attention is a highly parameterized, memory-bound kernel whose optimal configuration varies dramatically based on sequence lengths, head dimensions, and the target CDNA architecture (e.g., MI250X vs. MI300X).

## Architectural and Code Analysis

### 1. Triton Autotuner Integration
Triton's `@triton.autotune` decorator systematically explores a predefined configuration space to minimize kernel execution time. This PR introduces a specific set of configurations tailored to ROCm CDNA architectures for Flash Attention forward and backward passes. 

The tuning space generally explores:
- `BLOCK_M`: Block size along the sequence length dimension for the Query (Q).
- `BLOCK_N`: Block size along the sequence length dimension for the Key/Value (K, V).
- `num_stages`: The number of software pipeline stages. ROCm CDNA architectures can often utilize deeper pipelines to hide HBM latency via asynchronous loads.
- `num_warps`: The number of wavefronts (64 threads per wave on AMD) allocated per thread block.

### 2. Optimization Techniques & Memory Bounds
Flash Attention is primarily **memory-bound**, constrained by the HBM bandwidth necessary to stream large KV caches and Q tensors. Autotuning automatically discovers the optimal HBM-to-LDS trade-offs:

1. **LDS (Local Data Share) capacity vs. HBM Bandwidth**: Larger blocks reuse data more effectively, minimizing global memory round-trips. However, larger blocks consume more LDS. Over-allocating LDS reduces the number of concurrent thread blocks (occupancy) on a Compute Unit (CU).
2. **Register Pressure and Pipelining**: Deeper pipelines (`num_stages`) require more VGPRs (Vector General Purpose Registers) to hold in-flight HBM loads. If the VGPR limit is exceeded, the compiler either reduces occupancy or spills registers to memory, severely degrading performance. The autotuner finds the delicate balance between high `num_stages` and safe VGPR usage.
3. **MFMA Macro-tile Alignment**: Specific block dimensions align better with the underlying CDNA matrix core instructions (e.g., `v_mfma_f32_16x16x16bf16`). The autotuner effectively favors configurations where `BLOCK_M` and `BLOCK_N` are optimal multiples of MFMA dimensions, ensuring the matrix cores are fully saturated without padding overheads.

### 3. Impact Across CDNA Generations
Because CDNA hardware specifications evolve rapidly, hard-coded configurations fail to port optimally across generations. Autotuning solves this automatically:
- **CDNA2 (MI250X)**: Tuning tends to favor fewer `num_stages` and slightly smaller `BLOCK` sizes to fit within tighter LDS and register bounds while maintaining reasonable occupancy.
- **CDNA3 (MI300X) / CDNA4 (MI350X)**: With expanded LDS, massive global memory bandwidth improvements, and the Dual-CMA (Compute Matrix Accelerator) architecture, the autotuner naturally shifts towards deeper pipelines (higher `num_stages`) and larger `BLOCK_M`/`BLOCK_N`. This ensures that the dense computational power of the matrix cores does not starve waiting on memory.

## Implementation Details
The decision to implement autotuning within `perf-kernels` illustrates that Flash Attention is too complex for a one-size-fits-all static configuration block. Instead of relying on static shapes derived from NVIDIA hardware heuristics, this PR evaluates ROCm-tailored configurations on the host before selecting the optimal set. The tuning cache keys rely on `Q`, `K`, `V` sequence lengths and head sizes to ensure that changing input shapes dynamically triggers re-tuning or efficiently fetches the best-known configuration.

## Summary
By leveraging Triton's autotuner, PR #459 abstracts away the immense complexity of manual performance engineering on ROCm. It guarantees that Flash Attention can transparently adapt its blocking, pipelining, and wave allocation strategies to maximally utilize the HBM bandwidth and MFMA computational throughput of any underlying CDNA architecture.

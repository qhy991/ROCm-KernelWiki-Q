---
id: wiki-technique-pr-triton-685
title: "Triton MoE GEMM Implementation and Optimization"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, optimization, memory-bound, tiling]
confidence: inferred
---

# Triton MoE GEMM Implementation and Optimization

Based on ROCm/triton PR #685 ("Tianxing/moe gemm"), this page provides an architectural and code-level analysis of Triton-based Mixture of Experts (MoE) GEMM kernels optimized for AMD CDNA architectures. 

## Context and Intent

Mixture of Experts (MoE) models (like Mixtral or Grok) route different tokens to a subset of expert feed-forward networks (FFNs). Naively executing a distinct GEMM kernel for each expert introduces severe kernel launch overhead and poor GPU utilization, particularly for small or highly uneven batch sizes per expert. 

The introduction of a specialized MoE GEMM kernel in Triton addresses this inefficiency by computing the outputs for multiple experts in a single fused kernel or via a Grouped GEMM paradigm. The intent is to maximize throughput by efficiently batching irregular token-to-expert mappings while sustaining high Matrix Core (MFMA) utilization across the GPU.

## Architectural Analysis & Techniques

### Grouped GEMM & Token Routing
MoE GEMMs conceptually map to a Grouped GEMM where each group represents an expert's weights. In Triton, this typically relies on:
1. **Token Sorting and Permutation**: A preprocessing step sorts tokens based on their assigned expert, generating contiguous segments of tokens for each expert.
2. **Pointer Indirection (`tl.load` / `tl.store`)**: The Triton kernel computes the start offset of each expert's tokens and weights. Expert weights form a batched tensor `(num_experts, in_features, out_features)`.
3. **Block Scheduling**: Workgroups (Threadblocks) are assigned to compute output tiles. Since the number of tokens per expert varies, the grid configuration often requires mapping blocks to specific token-expert segments, dynamically load-balancing the workload across compute units (CUs).

### MFMA Instruction Utilization
On CDNA architectures (CDNA2/3/4), the Triton compiler maps block-level matrix multiplications (`tl.dot`) to underlying `v_mfma` (Matrix Fused Multiply-Add) instructions. For MoE GEMMs, tuning block sizes (e.g., `BLOCK_M`, `BLOCK_N`, `BLOCK_K`) is crucial:
* Because `M` (the number of tokens assigned to an expert) can be small and highly variable, smaller `BLOCK_M` sizes might be favored to avoid padding waste.
* Conversely, keeping `BLOCK_N` and `BLOCK_K` large enough is essential to saturate the dual-CMA (Compute Matrix Accelerator) units on CDNA3 and matrix cores on other CDNA generations.

## Performance and Memory Bounds

### The Memory-Bound Regime
MoE GEMM workloads frequently operate in the **memory-bound** regime rather than the compute-bound regime:
* **Low Arithmetic Intensity**: Due to token routing, the number of active tokens per expert in a single batch is often small (e.g., 32 to 128). Consequently, the kernel spends more time streaming the massive expert weight matrices from HBM than performing matrix multiplications.
* **HBM Bandwidth Saturation**: Fetching `(num_experts * hidden_size * intermediate_size)` parameters heavily taxes memory bandwidth. Optimizations must focus on maximizing read throughput.

### Optimization Strategies in Triton
1. **Vectorized Loads**: Triton aggressively utilizes wide vector loads (e.g., 128-bit `global_load_dwordx4` or larger, as hardware allows) to maximize memory throughput when fetching activations and weights.
2. **Double Buffering and Async Copy**: Leveraging pipelined `tl.dot` and `tl.load` operations. The ROCm Triton backend lowers these to asynchronous global-to-LDS transfers, overlapping the fetching of the next tile's weights with the MFMA computations of the current tile.
3. **LDS Bank Conflict Mitigation**: Applying memory swizzling when laying out blocks of A and B in Local Data Share (LDS) to ensure unhindered MFMA reads.
4. **Occupancy Tuning**: Balancing VGPR (Vector General Purpose Register) and LDS usage. High register pressure per thread limits the number of active wavefronts per CU, which diminishes the GPU's ability to hide memory latency—a critical penalty for memory-bound MoE scenarios.

## Reproducibility and Benchmarking considerations

When evaluating MoE GEMM performance on ROCm:
* Compare against batched or grouped GEMMs available in `hipBLASLt`.
* Profile utilizing `rocprof` to measure Global Memory Bandwidth (GB/s) vs. Peak Theoretical, and TFLOPS vs. Peak Theoretical. 
* Monitor the **"tail effect"**—where overall execution time is gated by the expert with the largest token assignment, leaving other CUs idle. Advanced load-balancing techniques (like Stream-K scheduling or persistent kernels) may be required if severe load imbalance is observed.

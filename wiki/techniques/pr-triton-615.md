---
id: technique-pr-triton-615
title: "Autotuning Space Expansion for Flash Attention"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
hardware_features: [lds, wavefront, mfma]
kernel_types: [flash-attention]
languages: [triton-rocm]
techniques: [occupancy-tuning, async-copy]
tags: [optimization, rocm-kernel, memory-bound, compute]
confidence: inferred
sources: [pr-triton-615]
---

# Autotuning Space Expansion for Flash Attention

## Overview
This technique focuses on expanding the autotuning search space (the set of configurations or "keys") for Flash Attention kernels within the ROCm Triton compiler. By providing a wider array of tuning parameters—such as block sizes, number of warps, and pipeline stages—the compiler can discover highly optimized configurations specifically tailored to AMD CDNA architectures. 

**Source PR:** [#615: Increase CI timeout](https://github.com/ROCm/triton/pull/615)

## Context & Intent
In the ROCm Triton backend, performance optimization heavily relies on an `autotuner` that empirically evaluates multiple tiling and scheduling strategies at runtime to select the most performant variant.

The primary intent inferred from PR #615 is that the developers broadened the autotuning parameter keys for the Flash Attention implementation. As a direct consequence of exploring this expanded configuration space, the continuous integration (CI) tests experienced significant compilation and execution time increases, resulting in timeouts. This highlights a fundamental trade-off in JIT-compiled languages like Triton: **deeper search spaces yield higher kernel performance at the cost of exponentially increased offline/JIT compilation and tuning times.**

## Architectural and Performance Analysis

### Memory Bounds and LDS Limits
Flash Attention is inherently designed to minimize global memory traffic by keeping intermediate attention score matrices in fast, on-chip Local Data Share (LDS). 
- **LDS Sizing:** On CDNA2 and CDNA3, adjusting tile dimensions (`BLOCK_M`, `BLOCK_N`) via autotuning directly affects LDS pressure. If a tile configuration overflows the available LDS per Compute Unit (CU), it drastically hurts occupancy. A broader autotuning space enables finding the exact threshold where LDS is fully utilized without bottlenecking occupancy.
- **Memory-Bound Regimes:** For short context lengths, Flash Attention is often memory-bandwidth bound. Finding the optimal `num_stages` (software pipelining depth) is critical for hiding asynchronous global-to-LDS memory copies (`async-copy`).

### Compute and VGPR Constraints
For longer sequences or specific matrix dimensions, the kernel becomes compute-bound.
- **Wavefront Parallelism:** Expanding `num_warps` (wavefront count per thread block) helps the autotuner discover the optimal instruction-level and thread-level parallelism configuration. 
- **Register Pressure:** Larger block configurations increase Vector General Purpose Register (VGPR) pressure. High VGPR usage reduces the number of concurrent wavefronts (occupancy). Autotuning empirically finds the sweet spot between block size efficiency and occupancy drop caused by VGPR limits.

## Optimization Strategies

When adapting autotuning spaces for ROCm kernels:

1. **Prune the Search Space:** Avoid brute-force autotuning. Implement early-exit heuristics to discard configurations that mathematically exceed hardware limits (e.g., configurations that demand >64KB LDS or >512 VGPRs per wave).
2. **Align with MFMA Instructions:** Restrict tiling keys (`BLOCK_M`, `BLOCK_N`, etc.) to multiples of native AMD Matrix Fused Multiply-Add (MFMA) instruction sizes (e.g., 16x16 or 32x32) to prevent wasted compute lanes.
3. **Handle CI Overheads:** As seen in this PR, expanding autotuning keys will bloat testing time. Consider separating extensive autotuning tests into nightly runs rather than per-commit CI pipelines to maintain developer velocity.

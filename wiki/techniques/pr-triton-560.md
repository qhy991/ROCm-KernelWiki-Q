---
id: technique-pr-triton-560
title: "Triton Attention Tuning: Supporting Head Size <= 256 via Autotuning"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
hardware_features: [lds]
techniques: [occupancy-tuning, double-buffering, async-copy]
kernel_types: [attention, flash-attention]
languages: [triton-rocm]
tags: [rocm, rocm-kernel, optimization, memory-bound, occupancy, lds]
confidence: inferred
sources: [pr-triton-560]
---

# Triton Attention Tuning: Supporting Head Size <= 256

## Intent & Context

In Transformer architectures, the dimension of the attention heads ($d$ or `head_size`) significantly dictates both computational intensity and the memory footprint of Flash Attention kernels. Historically, head sizes of 64 or 128 were standard, but newer models increasingly push for larger dimensions like $d=256$ to improve model expressivity per head.

Supporting head sizes up to 256 in OpenAI Triton for ROCm targets (CDNA architectures) introduces profound register and Local Data Share (LDS) pressure. PR #560 introduces specific autotune configurations aimed at `head_size <= 256`, achieving similar performance parity to standard $d=128$ setups through careful tuning of tile sizes and pipelining stages.

## Architectural Deep Dive

### LDS Memory Bounds

On AMD CDNA architectures like CDNA2 (MI250X) and CDNA3 (MI300X), a single Compute Unit (CU) has a fixed capacity of LDS. In a standard Flash Attention implementation, the SRAM (LDS) must hold blocked tiles of the Query ($Q$), Key ($K$), and Value ($V$) matrices. 

When scaling from $d=128$ to $d=256$:
* The memory requirement per row doubles.
* At half-precision (FP16 or BF16), a single tile of $Q$ using `BLOCK_M=128` and $d=256$ requires $128 \times 256 \times 2 = 64$ KB. 
* This immediately saturates the typical per-CU LDS limits (e.g., 64KB on CDNA2 per CU) before even allocating space for $K$, $V$, or multi-stage pipeline buffers used by Triton's asynchronous copy semantics (`num_stages`).

### Optimization Techniques

To sustain performance equivalent to $d=128$ without causing LDS allocation failures or catastrophic occupancy drops, the new autotuning config balances the parameter space:

1. **Tile Dimension Scaling (`BLOCK_M` and `BLOCK_N`)**:
   By adding targeted autotuning configs, the kernel explores smaller block sizes for large head dimensions. A configuration like $64 \times 256$ halves the LDS footprint per pipeline stage compared to $128 \times 256$, allowing the kernel to successfully compile and execute without exceeding shared memory bounds.

2. **Pipeline Stage Adjustments (`num_stages`)**:
   Triton uses software pipelining to overlap global memory reads with matrix multiplications. 
   * Higher `num_stages` (e.g., 3 or 4) effectively triple or quadruple the LDS requirements for buffered $K$ and $V$ tiles. 
   * For $d=256$, the autotuner likely curtails `num_stages` to 2 (classic double-buffering) to constrain the total working set inside the available shared memory.

3. **Occupancy vs. Register Pressure**:
   Matrix multiplications compile down to `v_mfma` instructions on AMD GPUs. Doubling the inner K-dimension per iteration naturally demands more vector registers (VGPRs) for accumulators. The new configuration finds a sweet spot for `num_warps`, minimizing register spilling while maintaining enough active wavefronts to hide memory access and arithmetic latencies.

## Performance Profile

The author notes that performance for $d \le 256$ is now comparable to $d=128$:
* **Compute vs. Memory Bound**: For small head sizes, Attention frequently becomes memory bound due to the massive global memory traffic. As $d$ scales to 256, the arithmetic intensity (math operations per byte of memory read) increases linearly. 
* By resolving the LDS bottlenecks and properly scheduling loads via autotuning, the kernel maintains high MFMA pipeline utilization on CDNA architectures, ensuring that the theoretical compute advantages of a larger head size translate into matching real-world efficiency (TFLOPS) without degradation.

## Summary of Code Implications

The core change in Triton PR #560 lies in expanding the `@triton.autotune` configurations within the attention kernel implementation. Predefined `triton.Config` decorators dictate how the compiler navigates the kernel optimization space for large $d$:
* **`BLOCK_M` and `BLOCK_N`**: Reduced to fit the wider 256 dimension into LDS.
* **`num_warps`**: Tweaked to sustain occupancy under elevated register load.
* **`num_stages`**: Capped to maintain double-buffering inside limited shared memory.

This eliminates fallback scenarios to unoptimized parameters and effectively enables large-head models to run seamlessly on AMD infrastructure.

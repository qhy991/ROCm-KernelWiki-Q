---
id: technique-pr-triton-469
title: "Triton ROCm: 64x4 and 4x64 Tile Sizes via MFMA 4x4"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - mfma
  - tiling
  - triton-rocm
  - optimization
confidence: inferred
sources:
  - pr-triton-469
---

# Triton ROCm: 64x4 and 4x64 Tile Sizes via MFMA 4x4

## Overview

PR #469 in the `ROCm/triton` repository introduces support for two highly skewed Matrix Fused Multiply-Add (MFMA) block sizes: **64x4** and **4x64**. These shapes are specifically supported by mapping them down to the underlying `v_mfma_..._4x4` instructions available on AMD CDNA architectures.

This addition extends the Triton compiler's ability to efficiently handle matrix multiplication operations where one of the spatial dimensions (M or N) is extremely small.

## Architectural Context: MFMA 4x4

On AMD CDNA GPUs, matrix operations are accelerated using MFMA instructions. While the hardware offers various sizes for these instructions natively (such as `32x32`, `16x16`, and `4x4`), high-level compilers like Triton group these hardware-level operations into larger logical tiles or blocks to balance computation with memory throughput (Global to LDS, and LDS to VGPR).

The `v_mfma_..._4x4` class of instructions computes a 4x4 output matrix across a wavefront (64 threads). When Triton maps a `64x4` or `4x64` tile operation, it systematically unrolls or chains multiple `4x4` MFMA instructions to cover the broader tile dimension. 
For instance, a `64x4` tile requires $16 \times$ `4x4` MFMA instructions in the M-dimension per logical block iteration.

## Motivation & Use Cases

Historically, block sizes in highly optimized kernels default to squarish shapes (like `32x32`, `64x64`, or `128x128`) to maximize data reuse from LDS memory. However, highly skewed matrix operations appear frequently in modern ML workloads:

1.  **Generative AI Decoding:** In auto-regressive decoding phases (e.g., token generation in LLMs), operations typically run with a sequence length of 1, resulting in very small dimensions for the Query or KV projections.
2.  **Attention Heads:** Multi-Head Attention components with a small head dimension (like 64, or varying intermediate dimensions) often require asymmetric matrix multiplication.
3.  **Vector-Matrix Products (GEMV-like workloads):** When $M \le 4$, treating a GEMM as a set of standard squarish tiles vastly underutilizes the matrix cores due to padding and overhead.

By providing explicit compiler support for `64x4` and `4x64` shapes, the ROCm Triton backend natively allows efficient mapping of these dimensions to the matrix cores without excessive padding or wasteful calculation of dummy elements.

## Compiler Mechanism

When Triton emits the intermediate representation (IR) for these skewed shapes, the AMD backend converts the operations as follows:

1.  **Tiling Resolution:** The high-level tensor operation (e.g., `tl.dot`) explicitly infers or specifies `64x4` or `4x64` layouts.
2.  **Instruction Mapping:** The `amd_mfma` lowering pass targets the 4x4 MFMA variants.
3.  **Register Allocation:** Because the output dimension is extremely narrow on one side, it drastically reduces the VGPR (Vector General Purpose Register) pressure per element compared to large square tiles, enabling higher occupancy.
4.  **Memory Access:** Memory accesses from global to LDS and LDS to VGPR are reshaped to match the long but thin strides.

## Benefits

-   **Reduced Wasteful Computation:** Directly computing the required narrow dimensions without padding to larger tile boundaries (e.g., upcasting to a 32x32 matrix shape).
-   **Improved Occupancy:** Less register usage for small-dimension accumulation allows more waves to reside on a compute unit.
-   **Tailored Performance for Decoding Phase:** Crucial for improving tokens-per-second in LLM inference, where latency for small sequence dimensions is the primary bottleneck.

> [!TIP]
> When tuning a Triton kernel on ROCm for decoding passes, explicitly test tile configurations of `[64, 4]` or `[4, 64]` instead of falling back to default values. This often provides significant performance uplifts by routing correctly through `mfma_4x4`.

---
id: technique-pr-triton-478
title: "Refining Tolerance in Checking GEMM Correctness for FP8"
type: wiki-technique
architectures:
  - cdna3
tags:
  - fp8
  - rocm
  - quantization
  - programming
kernel_types:
  - gemm
languages:
  - triton-rocm
confidence: inferred
sources:
  - pr-triton-478
---

# Refining Tolerance in Checking GEMM Correctness for FP8

## Background and Motivation

When validating GEMM (General Matrix Multiply) kernels in Triton targeting AMD ROCm architectures (particularly CDNA3 / MI300 which features native FP8 support), correctness is verified by comparing the hardware's GEMM output against a high-precision software reference. 

However, validating ultra-low precision formats like `fp8e4m3` (specifically the `fp8e4m3b8` variant used in ROCm, which has 4 exponent bits and 3 mantissa bits) presents severe numerical challenges. Previous testing configurations in the Triton test suite exhibited unacceptably large differences between the Triton output and the reference implementation, forcing the use of loose tolerance bounds.

This high variance was traced to a specific numerical edge case: **subnormal (denormal) values**.

## The Subnormal Problem in FP8

In the `fp8e4m3` format, the dynamic range is extremely narrow. Subnormal numbers occur when the exponent field evaluates to zero but the mantissa is non-zero. 

Due to the tiny 3-bit mantissa, operating on or converting subnormal values introduces massive relative rounding errors. In a standard GEMM operation ($C = A \times B$), values are multiplied and then accumulated across the $K$ dimension. If the random input generator for the matrices produces subnormal values, these outsized rounding errors accumulate across thousands of dot products, leading to catastrophic precision loss that causes the kernel output to dramatically diverge from the higher-precision baseline.

## Implementation Details

To properly validate the underlying MFMA (Matrix Fused Multiply-Add) kernel without masking true bugs behind a large error tolerance, PR #478 introduces the following testing technique:

1. **Subnormal Avoidance in Input Generation**: The test framework's random input generator was modified to strictly avoid creating subnormal values when generating matrices in the `fp8e4m3b8` format. The input range is bounded to ensure that all generated elements fall within the "normal" representable range.
2. **Refined Error Tolerance**: With the primary source of numerical instability eliminated, the correctness checking logic can employ a significantly reduced (tighter) tolerance. 

## Architectural & Practical Context

- **CDNA3 Native FP8**: Testing the exact behavior of CDNA3's FP8 matrix core instructions (via `v_mfma_..._fp8_fp8` instructions) requires isolating hardware behavior from expected algorithmic rounding errors. By keeping inputs in the normal range, developers can reliably test if the hardware and Triton's compiler bindings are working correctly.
- **Deep Learning Workloads**: In practical LLM training and inference workloads, FP8 operations are almost always paired with **tensor-wise or block-wise scaling factors** specifically calculated to ensure the vast majority of weights and activations lie within the normal range of the FP8 format. Subnormal values are practically avoided because they destroy signal fidelity. Therefore, testing without subnormals actually models the real-world operating conditions of an FP8 GEMM kernel more accurately than purely unconstrained random generation.

## Summary

By intelligently constraining the generated random inputs to bypass the severe precision loss inherent to FP8 subnormals, the Triton test suite can enforce tight correctness bounds, ensuring rigorous verification of ROCm's FP8 MFMA hardware and kernel bindings.

---
id: technique-pr-triton-472
title: "Testing Infrastructure and MFMA Configurations in Triton (PR #472)"
type: wiki-technique
architectures:
  - cdna1
  - cdna2
  - cdna3
hardware_features:
  - mfma
tags:
  - rocm
  - hardware
  - fp8
  - bf16
kernel_types:
  - gemm
languages:
  - triton-rocm
  - python
confidence: verified
sources:
  - pr-triton-472
---

# Analysis of MFMA Configurations in Triton (PR #472)

## Overview
Triton PR #472 introduces Python scripts for generating MLIR lit tests corresponding to Matrix Fused Multiply-Add (MFMA) instructions on AMD's CDNA architectures. While primarily a testing infrastructure update, the generator scripts implicitly document critical mapping constraints and configuration parameters for using MFMA operations across CDNA1, CDNA2, and CDNA3 GPUs.

## Architectural Capabilities

The PR encodes specific configurations regarding supported dimensions, data types, and hardware capabilities:

### K-Width Encodings by Architecture
The `k_width` parameter determines how many elements along the K dimension of the GEMM are packed into a single operand word. This varies depending on the hardware architecture and datatype:

*   **CDNA1 (`gfx908`):**
    *   `f16`: 4
    *   `bf16`: 2 (Upgraded to 4 in CDNA2)
    *   `i8`: 4
    *   `f32`: 1
*   **CDNA2 (`gfx90a`):**
    *   `f16`: 4
    *   `bf16`: 4
    *   `i8`: 4
    *   `f32`: 1
*   **CDNA3 (`gfx940`):**
    *   `f8` formats (`f8E4M3FNUZ`, `f8E5M2FNUZ`): 8
    *   `f16` / `bf16`: 4
    *   `i8`: 8 (except when `m_dim` and `n_dim` are both 4, where `k_width = 4`)
    *   `f32`: 1

### Datatype Support
*   **FP8 / BF8 Support:** The `f8E4M3FNUZ` and `f8E5M2FNUZ` 8-bit floating-point formats are explicitly supported on **CDNA3** and above. The test generator explicitly marks operations using these datatypes as unsupported for `cdna_version < 3`.

### MFMA Tile Dimensions
Triton maps matrix operations to explicit tile configurations (`m_dim`, `n_dim`) for the `triton_gpu.mfma` operation attributes:
*   Standard dimensions include `32x32`, `16x16`, `64x4`, `4x64`, and `4x4`.
*   Operations using `f8` datatypes are deemed unsupported when the minimal tile dimension (`min(m_dim, n_dim)`) is 4.

## MLIR Generation and Transformation

The test generation highlights the two critical passes for MFMA support in Triton's AMDGPU backend:

1.  **AccelerateAMDMatmul Pass:** Transforms standard `blocked` layouts into `mfma` layouts (`triton_gpu.mfma` and `triton_gpu.dot_op` attributes) tailored for the target CDNA generation. This determines the optimal tile shape and K-width mapping shown above.
2.  **TritonGPU to ROCDL Conversion:** Maps the high-level `tt.dot` operations with MFMA layouts down to explicit LLVM IR intrinsic calls (e.g., `rocdl.mfma.f32.32x32x16.fp8.fp8`). The `generate_mfma_variants.py` script systematically tests over 20 distinct ROCDL operations spanning matrix core versions 1 through 3.

## Optimization Implications
While this PR does not introduce runtime optimizations, the codified matrices in these scripts serve as a reference implementation for compiler developers. Understanding these `k_width` constraints is crucial when authoring kernels targeting optimal MFMA instruction throughput, as matching the `k_width` exactly avoids extraneous layout conversions and register shuffling overhead.

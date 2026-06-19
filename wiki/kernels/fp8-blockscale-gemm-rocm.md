---
id: kernel-fp8-blockscale-gemm-rocm
title: FP8 and Block-Scale GEMM on ROCm
type: wiki-kernel
architectures: [cdna3, cdna4]
tags: [gemm, fp8, fp4, quantization, block-scale, cdna4, composable_kernel, hipblaslt]
confidence: source-reported
kernel_types: [gemm]
languages: [hip-cpp, ck-dsl]
techniques: [ck-tile-programming, mfma-scheduling, register-tiling, vectorized-load]
hardware_features: [mfma, scaled-mfma, block-scale, lds]
related:
  - hw-scaled-mfma
  - hw-mfma-matrix-core
  - kernel-moe-grouped-gemm-cdna4
  - kernel-streamk-splitk-gemm-rocm
sources:
  - doc-cdna4-isa
  - doc-cdna4-whitepaper
  - pr-composable_kernel-3543
  - pr-composable_kernel-3603
  - pr-composable_kernel-3620
  - pr-composable_kernel-3629
  - pr-hipblaslt-52
  - pr-hipblaslt-5
  - pr-transformerengine-537
  - pr-transformerengine-535
  - pr-transformerengine-613
  - pr-transformerengine-605
  - pr-transformerengine-568
  - pr-transformerengine-630
  - pr-transformerengine-627
  - pr-transformerengine-571
reproducibility: concept
---

# FP8 and Block-Scale GEMM on ROCm

FP8 GEMM on ROCm spans CDNA3 FP8 paths and CDNA4 block-scaled F8/F6/F4 MFMA. The recurring design question is where quantized values are unpacked, where scale factors are loaded, and whether the kernel can keep the scaled representation until the MFMA instruction boundary.

## Kernel Shape

```text
A/B tensors:
  packed low-precision values
  scale metadata per block, row, or column group

inner loop:
  load packed tile and scales
  transform or preshuffle into MFMA-friendly layout
  issue FP8 or scaled MFMA
  accumulate in FP32 or BF16-compatible accumulator path
```

On CDNA4, block-scale support makes native FP4/FP8 kernels more attractive because the scaling metadata is part of the matrix-core path instead of a separate dequantization epilogue.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `doc-cdna4-isa` | Instruction-level reference for scaled MFMA |
| `doc-cdna4-whitepaper` | CDNA4 architecture context for block-scaled compute |
| `pr-composable_kernel-3543` | BF8 stream-K generator support |
| `pr-composable_kernel-3603` | A4W4 block-scale GEMM support |
| `pr-composable_kernel-3620` | ABQuant block-scale GEMM optimization |
| `pr-composable_kernel-3629` | Preshuffle quantization for AB block-scale GEMM |
| `pr-hipblaslt-52` | CDNA3 FP8 support in hipBLASLt |
| `pr-hipblaslt-5` | MI355X FP8 GEMM closed-loop case study |
| [`pr-transformerengine-537`](../../sources/prs/transformerengine/PR-537.md) | MXFP4 training recipe: fused cast+transpose, layout shuffle, AITER GEMM dispatch, and weight caching |
| [`pr-transformerengine-535`](../../sources/prs/transformerengine/PR-535.md) | Shape-based MXFP4 GEMM tuning selection and quantizer-copy fix |
| [`pr-transformerengine-613`](../../sources/prs/transformerengine/PR-613.md) | CK MXFP8 grouped GEMM integration for gfx1250; useful as RDNA4 integration evidence, not CDNA4 performance evidence |
| [`pr-transformerengine-605`](../../sources/prs/transformerengine/PR-605.md) | Dev-branch gfx1250 MXFP8 scale pre-swizzling and production-like GEMM tests |
| [`pr-transformerengine-568`](../../sources/prs/transformerengine/PR-568.md) | Original gfx1250-branch MXFP8 scale pre-swizzling implementation |
| [`pr-transformerengine-630`](../../sources/prs/transformerengine/PR-630.md) | Open NN/NT transpose workaround for gfx1250 MXFP8 GEMM layouts |
| [`pr-transformerengine-627`](../../sources/prs/transformerengine/PR-627.md) | Open K-dimension restriction relaxation for gfx1250 MXFP8 GEMM |
| [`pr-transformerengine-571`](../../sources/prs/transformerengine/PR-571.md) | FP4 swizzle instruction compatibility for gfx1250; PR reports no gfx950 performance difference |

## Database Use

Index this page with `dtype=fp8`, `dtype=fp4`, `quantization=block_scale`, and `hardware=scaled_mfma`. It should be linked to MoE, attention projection, and fused GEMM pages because they reuse the same low-precision matmul substrate.

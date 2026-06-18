---
id: kernel-hipblaslt-fused-gemm-rocm
title: hipBLASLt Fused GEMM and Quantization on ROCm
type: wiki-kernel
architectures: [cdna3, cdna4]
tags: [gemm, fp8, fused-kernel, quantization, hipblaslt, cross-repo, inference]
confidence: source-reported
kernel_types: [gemm, activation]
languages: [hip-cpp, triton-rocm, python]
techniques: [vectorized-load, mfma-scheduling, persistent-kernel, occupancy-tuning]
hardware_features: [mfma, scaled-mfma, block-scale, lds]
related:
  - kernel-fp8-blockscale-gemm-rocm
  - kernel-streamk-splitk-gemm-rocm
  - kernel-moe-grouped-gemm-cdna4
sources:
  - pr-hipblaslt-52
  - pr-hipblaslt-364
  - pr-aiter-3285
  - pr-aiter-3422
  - pr-aiter-3457
  - pr-vllm-44626
  - pr-rocm_libraries-7520
reproducibility: concept
---

# hipBLASLt Fused GEMM and Quantization on ROCm

hipBLASLt and adjacent inference libraries collect many of the production-facing GEMM fusions: FP8 quantization, activation epilogues, preshuffled weights, Split-K variants, and model-specific tuning configs.

## Kernel Shape

```text
GEMM core:
  low-precision A/B tile loads
  MFMA or scaled MFMA accumulation

fused work:
  quantize or dequantize
  amax/scale update
  optional activation or normalization-like epilogue
  layout conversion for the next operator
```

The fused approach trades a more complex kernel for fewer global-memory round trips and fewer launches, which is valuable in autoregressive inference.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `pr-hipblaslt-52` | CDNA3 FP8 enablement |
| `pr-hipblaslt-364` | Fused FP8 quantization kernels with amax/scale and cast/transpose |
| `pr-aiter-3285` | GLM FP8 GEMM configs for gfx950 |
| `pr-aiter-3422` | GLM FP8 blockscale GEMM and FMoE tunings for gfx942 |
| `pr-aiter-3457` | Fused Split-K zero-init for FP8 A8W8 blockscale GEMMs |
| `pr-vllm-44626` | Preshuffled FP8 GEMM for per-channel attention weights |
| `pr-rocm_libraries-7520` | hipBLASLt FP8 solution-selection compatibility fix |

## Retrieval Cues

Retrieve this page for hipBLASLt FP8 GEMM, fused quantization, amax scale update, preshuffled FP8 weights, Split-K fused epilogues, or production inference GEMM tuning on ROCm.


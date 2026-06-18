---
id: kernel-fp8-flash-attention-rocm
title: FP8 FlashAttention on ROCm
type: wiki-kernel
architectures: [cdna3, cdna4]
tags: [attention, flash-attention, fp8, quantization, fused-kernel, optimization]
confidence: source-reported
kernel_types: [attention, flash-attention]
languages: [hip-cpp, ck-dsl, triton-rocm, python]
techniques: [ck-tile-programming, mfma-scheduling, vectorized-load, occupancy-tuning]
hardware_features: [mfma, scaled-mfma, block-scale, lds]
related:
  - kernel-flash-attention-rocm
  - kernel-fp8-blockscale-gemm-rocm
  - kernel-paged-prefill-attention-rocm
sources:
  - doc-flash-attention-rocm
  - pr-flash-attention-116
  - pr-flash-attention-119
  - pr-flash-attention-140
  - pr-flash-attention-154
  - pr-flash-attention-160
  - pr-composable_kernel-3633
  - pr-composable_kernel-3635
reproducibility: concept
---

# FP8 FlashAttention on ROCm

FP8 FlashAttention combines the memory-saving online-softmax structure with low-precision Q/K/V or intermediate data paths. It is especially relevant for inference and training runs that want attention bandwidth reduction without switching the entire model to a lower-precision GEMM path.

## Kernel Shape

```text
forward:
  load FP8 or quantized Q/K/V tiles
  apply scale metadata as required by layout
  compute QK and softmax in higher precision
  compute P * V and store output

backward:
  preserve enough scaling and softmax metadata
  compute dQ, dK, dV with FP8-aware conversion points
```

The exact placement of scale application is the key implementation choice. Applying scales too early increases memory traffic; applying them too late can complicate MFMA operand layout and numerical behavior.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `doc-flash-attention-rocm` | ROCm FlashAttention source context |
| `pr-flash-attention-116` | FP8 forward support |
| `pr-flash-attention-119` | FP8 backward support |
| `pr-flash-attention-140` | FP8 fused-kernel work |
| `pr-flash-attention-154` | FP8 documentation fixes |
| `pr-flash-attention-160` | FP8 performance tuning |
| `pr-composable_kernel-3633` | CK FP8 block-scale FMHA revert evidence |
| `pr-composable_kernel-3635` | CK FP8 block-scale FMHA re-landing evidence |

## Retrieval Cues

Retrieve this page for FP8 attention, FP8 FMHA, low-precision KV cache, FP8 backward attention, or fused FP8 FlashAttention on AMD GPUs.


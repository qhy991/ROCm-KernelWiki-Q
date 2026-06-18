---
id: kernel-triton-flash-attention-rocm
title: Triton FlashAttention on ROCm
type: wiki-kernel
architectures: [cdna3, cdna4]
tags: [attention, flash-attention, triton-rocm, optimization, training]
confidence: source-reported
kernel_types: [attention, flash-attention]
languages: [triton-rocm, python]
techniques: [vectorized-load, occupancy-tuning, wave-reduction]
hardware_features: [mfma, lds, dpp, wavefront]
related:
  - lang-triton-rocm
  - kernel-flash-attention-rocm
  - kernel-paged-prefill-attention-rocm
sources:
  - doc-flash-attention-rocm
  - pr-flash-attention-112
  - pr-flash-attention-113
  - pr-flash-attention-114
  - pr-flash-attention-122
  - pr-flash-attention-128
reproducibility: concept
---

# Triton FlashAttention on ROCm

The Triton ROCm FlashAttention path is useful when a project wants Python-level kernel iteration while still targeting AMD CDNA GPUs. The main tradeoff is that some hardware-specific control is delegated to Triton's compiler and backend lowering instead of being expressed directly in HIP or CK.

## Kernel Shape

```text
Triton program:
  block pointers for Q, K, V, O
  loop over K/V blocks
  dot -> online softmax -> dot with V
  split backward into dK/dV and dQ variants when needed
```

The ROCm-specific work usually appears in test coverage, backend pinning, launch parameter tuning, and head-dimension coverage rather than in large handwritten assembly blocks.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `doc-flash-attention-rocm` | ROCm FlashAttention repository context |
| `pr-flash-attention-112` | CK Tile FAv3 backward test and API usage update |
| `pr-flash-attention-113` | FAv3 backward follow-up changes |
| `pr-flash-attention-114` | FAv3 backward enablement for BF16 head_size=64 |
| `pr-flash-attention-122` | Backward Triton implementation with separated dK/dV and dQ kernels |
| `pr-flash-attention-128` | Triton commit update for ROCm compatibility |

## Database Use

Index this page under `language=triton_rocm`, `operator=flash_attention`, and `phase=backward` as well as `phase=forward`. It complements the CK/HIP FlashAttention page rather than replacing it.


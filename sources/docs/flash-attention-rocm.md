---
id: doc-flash-attention-rocm
title: ROCm Flash Attention Repository
type: source-doc
architectures: [cdna2, cdna3, cdna4]
tags: [flash-attention, attention, mfma]
date: '2026-01-01'
url: https://github.com/ROCm/flash-attention
source_category: upstream-code
kernel_types: [attention]
languages: [hip-cpp, ck-dsl]
confidence: verified
---

# ROCm Flash Attention

AMD-maintained FlashAttention port with multiple backends (CK, CK Tile, composable FMHA instances) optimized for CDNA GPUs.

## Integration Notes

- CK Tile FMHA is the primary path on MI300X/MI350X
- Backward passes require careful workspace sizing and atomic reduction modes
- Head dimension and dtype combinations map to different tile configs

## Related Wiki Pages

- [Flash Attention on ROCm](../../wiki/kernels/flash-attention-rocm.md)

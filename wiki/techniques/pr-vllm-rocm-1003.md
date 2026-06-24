---
id: technique-moe-stride-padding-gfx11x
title: "Applying Cache Cliff Stride Padding to W4A16 MoE Experts"
type: wiki-technique
confidence: verified
architectures:
  - rdna3
kernel_types:
  - moe
tags:
  - w4a16
  - cache-cliff
  - stride-padding
sources:
  - pr-vllm-rocm-1003
---

# Applying Cache Cliff Stride Padding to W4A16 MoE Experts

## MoE Memory Layout Nuances
While standard unquantized GEMMs can pad the trailing dimension, Mixture of Experts (MoE) weights have a complex 3D shape `[E, N, K//8]`. For hybrid W4A16 precision, the row strides can inadvertently land on power-of-2 alignments (e.g., exactly 1024 Bytes) with the expert stride landing precisely at 1 MB, triggering the catastrophic RDNA3 cache collision.

## The Solution
A padded buffer `[E, N, K//8 + 32]` is allocated, meaning an offset of 32 INT32 values (128 Bytes) is added per row. 
1. **Prefill (Triton)**: Triton's `fused_moe_kernel_gptq_awq` natively queries `B.stride(x)`, absorbing the padding dynamically without arithmetic changes.
2. **Decode (HIP)**: The `wvSplitK` HIP kernel dynamically computes the stride as `expert_stride_w / M`. Since the expert memory footprint expands, the derived stride correctly accounts for the padding.

Zero extra FLOPs are spent, and MoE-dominated prefill times drop significantly.

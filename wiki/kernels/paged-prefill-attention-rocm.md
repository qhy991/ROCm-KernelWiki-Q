---
id: kernel-paged-prefill-attention-rocm
title: Paged Prefill Attention on ROCm
type: wiki-kernel
architectures: [cdna3, cdna4]
tags: [attention, flash-attention, inference, memory, fp8, composable_kernel]
confidence: source-reported
kernel_types: [attention, flash-attention]
languages: [hip-cpp, ck-dsl]
techniques: [ck-tile-programming, vectorized-load, double-buffering, occupancy-tuning]
hardware_features: [mfma, lds, wavefront]
related:
  - kernel-flash-attention-rocm
  - kernel-fp8-flash-attention-rocm
  - technique-vectorized-load
sources:
  - doc-flash-attention-rocm
  - pr-composable_kernel-3545
  - pr-composable_kernel-3568
  - pr-composable_kernel-3657
  - pr-flash-attention-103
  - pr-aiter-3583
reproducibility: concept
---

# Paged Prefill Attention on ROCm

Paged prefill attention is the inference-oriented attention path where KV cache blocks are addressed through page tables rather than one contiguous sequence tensor. The kernel has to preserve FlashAttention's streaming softmax structure while handling page-size and KV-layout constraints.

## Kernel Shape

```text
for each query block:
  resolve KV page table entries
  load K/V tiles from paged cache layout
  compute QK with MFMA
  run online softmax
  compute softmax(QK) V
  write output for the prefill tokens
```

The page size and KV-cache layout affect vectorization. Small pages improve allocator flexibility but can reduce contiguous memory access; vectorized layouts improve bandwidth but require more careful descriptor math.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `doc-flash-attention-rocm` | Upstream FlashAttention ROCm context |
| `pr-composable_kernel-3545` | Batch prefill pipeline with page_size=1 linear layout |
| `pr-composable_kernel-3568` | Page size 16 support for batch prefill |
| `pr-composable_kernel-3657` | Vectorized-layout KV cache performance optimization |
| `pr-flash-attention-103` | Page attention in variable-length forward path |
| `pr-aiter-3583` | FP8 sparse paged prefill attention layout |

## Retrieval Cues

Retrieve this page for vLLM-style paged attention, prefill, page tables, KV cache, sparse paged attention, or page-size-specific attention kernels on ROCm.


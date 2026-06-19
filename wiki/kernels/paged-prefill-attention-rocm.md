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
  - pr-composable_kernel-3733
  - pr-composable_kernel-3745
  - pr-vllm-12348
  - pr-vllm-40745
  - pr-vllm-46148
  - pr-vllm-46076
  - pr-sglang-28700
  - pr-sglang-28712
  - pr-flash-attention-103
  - pr-flash-attention-157
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

## Correctness Boundaries

Paged attention kernels have correctness hazards that do not appear in contiguous FlashAttention:

| Boundary | Failure mode | Evidence |
|----------|--------------|----------|
| Page-table extent | V prefetch can read one-past-end `page_idx` if physical-page lookup is not clamped to valid KV length | `pr-composable_kernel-3733` |
| Dtype contract | BF16/FP16 Q with FP8 K/V is not the same as an all-FP8 `fp8bf16` path | `pr-composable_kernel-3745` |
| Page layout | Vectorized KV cache improves bandwidth but requires multi-dimensional page-index/gather handling | `pr-composable_kernel-3657` |
| Fused prepare + cache write | Decode fusion can make QK norm and RoPE inseparable from the paged KV-cache store contract | `pr-sglang-28700` |
| Sparse MiniMax page helpers | Sparse prefill/decode attention needs explicit page/index helper kernels before full model integration | `pr-sglang-28712` |
| FP8 KV scale semantics | FP8 K/V cache does not imply an FP8 query; query scale guards should only apply when the query operand is quantized | `pr-vllm-46148` |
| DCP-local sparse metadata | Sparse MLA/paged backends need rank-localized index metadata before launching under decode-context parallelism | `pr-vllm-46076` |

For serving systems, treat dtype and page layout as part of the kernel contract. A dispatcher should reject unsupported mixed-dtype requests explicitly rather than silently falling back to a nearby all-BF16 or all-FP8 path.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `doc-flash-attention-rocm` | Upstream FlashAttention ROCm context |
| `pr-composable_kernel-3545` | Batch prefill pipeline with page_size=1 linear layout |
| `pr-composable_kernel-3568` | Page size 16 support for batch prefill |
| `pr-composable_kernel-3657` | Vectorized-layout KV cache performance optimization |
| `pr-composable_kernel-3733` | Clamp paged KV lookups to avoid one-past-end V prefetch |
| `pr-composable_kernel-3745` | Explicit unsupported contract for mixed Q plus FP8 K/V batch prefill |
| `pr-vllm-12348` | vLLM custom paged-attention kernel using `mfma16x16x16` on MI300X |
| `pr-vllm-40745` | Head-size guard for ROCm paged attention when `head_size < 128` |
| `pr-vllm-46148` | Open ROCm attention compatibility fix scoping `q_scale` validation to FP8-query cases, not all FP8-KV cases |
| `pr-vllm-46076` | Open DCP integration for `FLASHINFER_MLA_SPARSE`, emphasizing sparse attention index localization |
| `pr-sglang-28700` | Open Qwen3.5 decode proposal fusing QK RMSNorm, 3D mRoPE, and paged KV-cache store into one AITER HIP op |
| `pr-sglang-28712` | Open MiniMax-M3 sparse attention foundation with decode/prefill sparse ops and page-index helpers |
| `pr-flash-attention-103` | Page attention in variable-length forward path |
| `pr-flash-attention-157` | FlashAttention v3 enablement with FP8 and paged attention |
| `pr-aiter-3583` | FP8 sparse paged prefill attention layout |

## Retrieval Cues

Retrieve this page for vLLM-style paged attention, prefill, page tables, KV cache, sparse paged attention, or page-size-specific attention kernels on ROCm.

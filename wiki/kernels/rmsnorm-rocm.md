---
id: kernel-rmsnorm-rocm
title: RMSNorm and Normalization Kernels on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [rmsnorm, reduction, memory, fused-kernel, triton-rocm, rocm]
confidence: source-reported
kernel_types: [rmsnorm, reduction]
languages: [triton-rocm, python, hip-cpp]
techniques: [wave-reduction, vectorized-load, occupancy-tuning]
hardware_features: [dpp, lds, wavefront]
related:
  - kernel-reduction-softmax-rocm
  - kernel-hipblaslt-fused-gemm-rocm
  - lang-triton-rocm
  - technique-occupancy-tuning
sources:
  - pr-transformerengine-615
  - pr-vllm-44976
  - pr-sglang-27745
  - pr-sglang-28700
  - pr-sglang-28712
  - pr-composable_kernel-3681
reproducibility: concept
---

# RMSNorm and Normalization Kernels on ROCm

RMSNorm is usually smaller than GEMM or attention, but it sits on the critical path of LLM inference and training. On ROCm it is often implemented as a bandwidth-sensitive reduction plus scale operation, then fused with quantization, RoPE, or a downstream GEMM contract.

## Kernel Shape

```text
for each token row:
  load hidden vector
  compute sum(x * x) with wave/block reduction
  rsqrt(mean + epsilon)
  scale by gamma
  optionally quantize or write into a producer-owned buffer
```

RMSNorm performance is dominated by memory movement, reduction layout, and launch count. A single standalone RMSNorm kernel may be fast in isolation but still lose to a fused norm+quant path if it forces another HBM round trip.

## Fusion Contracts

| Fusion | Why it matters | Evidence |
|--------|----------------|----------|
| RMSNorm + quant | The producer already writes the GEMM input or output buffer, so it can satisfy layout/zero-init requirements | `pr-vllm-44976` |
| Triton RMSNorm | Python-level kernel iteration is useful for normalization because the logic is compact and shape-dependent | `pr-transformerengine-615` |
| Norm/RoPE/FP8 dispatch | CDNA3 and CDNA4 can differ in FP8 max constants and shared-memory assumptions | `pr-sglang-27745` |
| QK RMSNorm + mRoPE + KV store | Decode serving can remove multiple launch-bound prepare kernels by fusing norm, rotary embedding, and cache write into one guarded HIP op | `pr-sglang-28700` |
| MiniMax JIT norm kernels | New sparse MoE models often land RMSNorm/QK-norm JIT kernels before full model wiring, so the kernel surface can be validated independently | `pr-sglang-28712` |
| CK model-sensitive path | Small casts or dtype conversions can be visible in model-sensitive normalization kernels | `pr-composable_kernel-3681` |

## ROCm Tuning Notes

1. **Use wave reductions deliberately.** A 64-lane wavefront changes the reduction tree relative to CUDA warp-sized kernels.
2. **Avoid unnecessary dtype round trips.** Normalization often feeds quantized GEMM or attention; keep scale and cast placement aligned with the consumer.
3. **Fuse only when ownership is clear.** vLLM's Split-K zero-init fusion is safe because the producer is explicitly rewritten to pre-zero the downstream GEMM output buffer.
4. **Dispatch on architecture when constants differ.** SGLang's gfx942/gfx950 dispatch is a reminder that FP8 format constants are part of the kernel contract.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `pr-transformerengine-615` | TransformerEngine ROCm Triton RMSNorm optimization |
| `pr-vllm-44976` | RMSNorm/quant producer fusion for AITER block-scale FP8 Split-K zero-init |
| `pr-sglang-27745` | Runtime CDNA3/CDNA4 dispatch for FP8-related fused kernels |
| `pr-sglang-28700` | Open AITER fusion proposal for QK RMSNorm + 3D mRoPE + paged KV-cache store |
| `pr-sglang-28712` | Open MiniMax-M3 additive JIT kernel foundation with RMSNorm and QK-norm/RoPE kernels |
| `pr-composable_kernel-3681` | CK Tile RMSNorm cast cleanup in a model-sensitive path |

## Retrieval Cues

Retrieve this page for RMSNorm, normalization, norm+quant fusion, reduction kernels, Triton normalization, FP8 quant producers, or Split-K zero-init fusion on ROCm.

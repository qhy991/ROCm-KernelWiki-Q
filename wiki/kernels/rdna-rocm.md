---
id: kernel-rdna-rocm
title: RDNA ROCm Kernels (gfx11/gfx12)
type: wiki-kernel
architectures: [rdna3, rdna4]
tags: [rdna, rocm, gemm, attention, inference, optimization]
confidence: source-reported
kernel_types: [gemm, attention, flash-attention, grouped-gemm]
languages: [hip-cpp, triton-rocm, python, assembly]
techniques: [mfma-scheduling, vectorized-load, occupancy-tuning, register-tiling]
hardware_features: [mfma, wavefront, lds]
related:
  - kernel-flash-attention-rocm
  - kernel-triton-flash-attention-rocm
  - kernel-gemm-rocm
  - kernel-moe-grouped-gemm-cdna4
  - lang-triton-rocm
  - lang-amd-assembly
sources:
  - pr-flash-attention-178
  - pr-flash-attention-184
  - pr-composable_kernel-3598
  - pr-composable_kernel-3619
  - pr-vllm-40977
  - pr-vllm-45559
  - pr-transformerengine-578
  - pr-rocwmma-611
reproducibility: concept
---

# RDNA ROCm Kernels (gfx11/gfx12)

RDNA kernel work in this wiki is a related but distinct track from CDNA. RDNA targets desktop/workstation and local-serving GPUs such as gfx11/gfx12, while CDNA pages focus on MI-series datacenter GPUs. Some APIs are shared, but performance assumptions and supported instruction paths can diverge.

## Main Themes

| Theme | RDNA-specific lesson | Evidence |
|-------|----------------------|----------|
| FlashAttention enablement | CK Tile needed gfx11/gfx12 build support, BF16 conversion handling, and guarded backward paths before broader enablement | `pr-flash-attention-178`, `pr-flash-attention-184` |
| WMMA contraction exposure | CK exposes RDNA WMMA paths for FP16/BF16 contractions through hipTensor-facing APIs | `pr-composable_kernel-3598` |
| Grouped GEMM | RDNA4 WMMA grouped GEMM supports fixed-NK multi-ABD examples with bias/GELU-style elementwise operations | `pr-composable_kernel-3619` |
| Quantized local serving | vLLM routes tiny decode shapes to HIP skinny GEMM and prefill shapes to Triton fused-dequant GEMM | `pr-vllm-40977` |
| SWMMAC exploration | GFX12 skinny GEMM and rocWMMA SWMMAC work are active/open or closed-prototype directions, not settled CDNA-style primitives | `pr-vllm-45559`, `pr-rocwmma-611` |

## Avoid These Overgeneralizations

1. **Do not map gfx950 conclusions to gfx1201.** CDNA4 block-scale/scaled-MFMA pages often discuss MI350/MI355; RDNA4 PRs may target gfx1200/gfx1201/gfx1250 and should keep that scope.
2. **Do not treat open PRs as released behavior.** vLLM W4A16 and SWMMAC skinny GEMM PRs are useful trend evidence but should be cited as open unless merged.
3. **Do not assume CK backward coverage is complete.** FlashAttention RDNA support evolved from build enablement with guards to an open PR removing local backward skips.
4. **Do not mix WMMA and MFMA vocabulary casually.** RDNA pages often say WMMA/SWMMAC, while CDNA pages usually discuss MFMA and scaled-MFMA.

## Serving-Oriented Kernel Routing

RDNA local-serving work often splits the same operator by shape:

```text
if decode batch is tiny:
  use HIP skinny GEMM or SWMMAC-specialized path
else:
  use Triton fused-dequant GEMM or ordinary batched path
```

That differs from many CDNA datacenter pages, where the main question is saturating large MI-series matrix cores across high-throughput batches.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `pr-flash-attention-178` | RDNA gfx11/gfx12 build support with backward guards |
| `pr-flash-attention-184` | Open PR removing RDNA CK backward guards after CK update |
| `pr-composable_kernel-3598` | gfx11/gfx12 FP16/BF16 WMMA contraction exposure for hipTensor |
| `pr-composable_kernel-3619` | RDNA4 WMMA grouped GEMM fixed-NK multi-ABD path |
| `pr-vllm-40977` | Open W4A16 hybrid linear routing for gfx11/gfx12 decode/prefill |
| `pr-vllm-45559` | Open GFX12 SWMMAC skinny GEMM direction for N=5..8 decode |
| `pr-transformerengine-578` | gfx1250 MXFP8 grouped GEMM integration in ROCm TransformerEngine |
| `pr-rocwmma-611` | Closed experimental SWMMAC rocWMMA prototype with compiler caveats |

## Retrieval Cues

Retrieve this page for RDNA ROCm kernels, gfx11, gfx12, gfx1201, gfx1250, SWMMAC, RDNA FlashAttention, W4A16 local serving, RDNA WMMA, or RDNA4 grouped GEMM.

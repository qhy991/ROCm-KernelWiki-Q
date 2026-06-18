---
id: kernel-reduction-softmax-rocm
title: Reduction and Softmax Kernels on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [reduction, softmax, dpp, bpermute, memory, optimization]
confidence: source-reported
kernel_types: [reduction, softmax]
languages: [hip-cpp, ck-dsl]
techniques: [wave-reduction, vectorized-load, occupancy-tuning]
hardware_features: [dpp, bpermute, wavefront, lds]
related:
  - hw-dpp-cross-lane
  - hw-wavefront
  - kernel-flash-attention-rocm
  - kernel-paged-prefill-attention-rocm
sources:
  - doc-ck-readme
  - doc-ck-structure
  - blog-matrix-cores-cdna
  - pr-composable_kernel-3564
reproducibility: concept
---

# Reduction and Softmax Kernels on ROCm

Reduction and softmax are small compared with GEMM, but they often define the latency and numerical behavior of attention, normalization, and fused epilogues. On ROCm, these kernels rely heavily on wavefront-level cross-lane movement and careful memory layout.

## Kernel Shape

```text
row reduction:
  each wave owns one or more rows
  load vectorized elements
  reduce max/sum through DPP or bpermute
  optionally write row statistics

softmax:
  max reduction
  exp and sum reduction
  normalize and store
```

For attention, softmax is usually fused into the streaming QK loop so the kernel stores only compact state instead of a full attention matrix.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `doc-ck-readme` | CK operator families and kernel programming context |
| `doc-ck-structure` | Locations for primitives and examples |
| `blog-matrix-cores-cdna` | CDNA wavefront and matrix-core background |
| `pr-composable_kernel-3564` | Batched GEMM-softmax-GEMM descriptor fix |

## Database Use

Index this page under `operator=softmax`, `operator=reduction`, `technique=wave_reduction`, and `hardware=dpp`. It should be linked as a dependency of attention and normalization recipes.


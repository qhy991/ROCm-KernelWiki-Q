---
id: kernel-gemm-mfma-rocm
title: MFMA GEMM on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [gemm, mfma, lds, composable_kernel, hipblaslt, optimization]
confidence: source-reported
kernel_types: [gemm]
languages: [hip-cpp, ck-dsl]
techniques: [ck-tile-programming, mfma-scheduling, double-buffering, register-tiling]
hardware_features: [mfma, lds, wavefront]
related:
  - hw-mfma-matrix-core
  - hw-lds
  - technique-mfma-scheduling
  - technique-double-buffering
sources:
  - doc-ck-readme
  - doc-ck-structure
  - blog-matrix-cores-cdna
  - pr-composable_kernel-3544
  - pr-composable_kernel-3583
  - pr-composable_kernel-3611
reproducibility: concept
---

# MFMA GEMM on ROCm

Dense GEMM is the baseline kernel family behind ROCm attention, convolution, and MoE paths. On CDNA GPUs, the main compute primitive is MFMA, while CK and hipBLASLt provide the host-side selection, tiling, and epilogue machinery.

## Kernel Shape

The common ROCm GEMM shape is a block-level M x N tile with a wave-level MFMA microkernel:

```text
for each CTA tile (Mblk, Nblk):
  stage A[Mblk, Kstep] and B[Kstep, Nblk] through LDS
  for each Kstep:
    issue v_mfma_* instructions
    interleave LDS/vector loads with accumulator updates
  apply epilogue and store C[Mblk, Nblk]
```

The tile engine and older XDL paths differ in metaprogramming surface, but both optimize the same pressure points: LDS layout, VGPR accumulator count, wave scheduling, and global-memory vectorization.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `doc-ck-readme` | CK library scope and GEMM as a first-class operation |
| `doc-ck-structure` | Repository locations for device GEMM, tile engine, and examples |
| `blog-matrix-cores-cdna` | MFMA programming model and CDNA matrix-core background |
| `pr-composable_kernel-3544` | Shared GridwiseGemm XDL base class cleanup |
| `pr-composable_kernel-3583` | GEMM multi-D fixes in the tile engine |
| `pr-composable_kernel-3611` | Interwave pipeline support for basic GEMM pipelines |

## Retrieval Cues

Use this page when a task asks for ROCm GEMM, MFMA scheduling, CK GEMM, XDL GEMM, or a baseline matmul implementation before adding quantization, grouping, attention, or convolution-specific behavior.


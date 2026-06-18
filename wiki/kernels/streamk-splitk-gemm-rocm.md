---
id: kernel-streamk-splitk-gemm-rocm
title: Stream-K and Split-K GEMM on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [gemm, scheduling, persistent-kernel, optimization, composable_kernel, hipblaslt]
confidence: source-reported
kernel_types: [gemm]
languages: [hip-cpp, ck-dsl]
techniques: [persistent-kernel, mfma-scheduling, occupancy-tuning, wave-reduction]
hardware_features: [mfma, lds, gws, wavefront]
related:
  - technique-persistent-kernel
  - kernel-gemm-mfma-rocm
  - kernel-fp8-blockscale-gemm-rocm
sources:
  - pr-composable_kernel-3559
  - pr-composable_kernel-3625
  - pr-composable_kernel-3653
  - pr-composable_kernel-3662
  - pr-rocm_libraries-7980
reproducibility: concept
---

# Stream-K and Split-K GEMM on ROCm

Stream-K and Split-K are scheduling strategies for GEMM shapes where ordinary CTA tiling leaves compute units imbalanced or the K dimension is large enough to profit from parallel reduction.

## Kernel Shape

```text
Split-K:
  multiple CTAs compute partial C tiles over disjoint K ranges
  a second reduction step combines partial accumulators

Stream-K:
  CTAs claim work units from a stream of K tiles
  scheduling smooths tail effects and irregular tile counts
```

These paths are especially relevant for block-scaled GEMM and inference workloads where shape distributions are broad rather than a single square GEMM.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `pr-composable_kernel-3559` | CK Tile Stream-K reduction test handling |
| `pr-composable_kernel-3625` | Alignment fix in Stream-K workspace buffers |
| `pr-composable_kernel-3653` | Split-K support for block-scale GEMM B-quant mode |
| `pr-composable_kernel-3662` | Stream-K tile-engine test config generation |
| `pr-rocm_libraries-7980` | hipBLASLt/TensileLite cleanup around legacy StreamK modes |

## Retrieval Cues

Retrieve this page for tasks mentioning Split-K, Stream-K, persistent GEMM, tail-tile scheduling, workspace reductions, or large-K parallelization.


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
  - pr-rocm_libraries-7636
  - pr-rocm_libraries-8442
  - pr-rocm_libraries-8615
  - pr-vllm-44976
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

## TensileLite Subtile and Stream-K Scheduling

Recent hipBLASLt work has moved Stream-K from a single scheduling label toward a family of codegen-time scheduler choices:

| Path | What changes | Why it matters |
|------|--------------|----------------|
| Tail-loop Subtile | Allows Subtile kernels when `K % 32 == 0` rather than requiring `K % DepthU == 0` | More real GEMM shapes can use the optimized Subtile path |
| Dynamic queue Stream-K | Workgroups fetch tile work from queues instead of static ownership only | Better tail-tile balance for irregular problem sizes |
| Single-hop work stealing | A drained queue can steal one neighboring structural-extra tile | Reduces idle workgroups without changing the default code path |
| initD/GR overlap | Accumulator zeroing moves into the Subtile preloop | Hides VGPR initialization under global reads |

This is a useful retrieval distinction: **Split-K** is a parallel reduction strategy over K, while **Stream-K/Subtile** is also a scheduling and code-generation problem. Tuning can involve queue semantics, tail-loop legality, prologue placement, and solution-selection heuristics, not only tile dimensions.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `pr-composable_kernel-3559` | CK Tile Stream-K reduction test handling |
| `pr-composable_kernel-3625` | Alignment fix in Stream-K workspace buffers |
| `pr-composable_kernel-3653` | Split-K support for block-scale GEMM B-quant mode |
| `pr-composable_kernel-3662` | Stream-K tile-engine test config generation |
| `pr-rocm_libraries-7980` | hipBLASLt/TensileLite cleanup around legacy StreamK modes |
| `pr-rocm_libraries-7636` | Initial Subtile tail-loop support for gfx950 BF16/MXFP4 shapes |
| `pr-rocm_libraries-8442` | Optional single-hop Stream-K work stealing for dynamic queues |
| `pr-rocm_libraries-8615` | Overlap Subtile accumulator initialization with global reads |
| `pr-vllm-44976` | ROCm AITER FP8 Split-K requires explicit zero-init contract, then fuses it upstream |

## Retrieval Cues

Retrieve this page for tasks mentioning Split-K, Stream-K, persistent GEMM, tail-tile scheduling, workspace reductions, or large-K parallelization.

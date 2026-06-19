---
id: pattern-compute-bound-mfma-amd
title: Compute-Bound MFMA Pattern on AMD GPUs
type: wiki-pattern
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [mfma, compute, optimization, scheduling, occupancy, rocm]
confidence: source-reported
techniques: [mfma-scheduling, register-tiling, occupancy-tuning, double-buffering]
hardware_features: [mfma, wavefront, lds, dual-cma]
related:
  - hw-mfma-matrix-core
  - hw-wavefront
  - technique-mfma-scheduling
  - technique-register-tiling
  - technique-occupancy-tuning
  - kernel-gemm-mfma-rocm
sources:
  - doc-cdna4-isa
  - doc-cdna4-whitepaper
  - pr-composable_kernel-3709
  - pr-composable_kernel-3603
  - pr-composable_kernel-3620
reproducibility: concept
---

# Compute-Bound MFMA Pattern on AMD GPUs

A kernel is compute-bound when matrix-core issue rate, accumulator movement, or epilogue math dominates global-memory traffic. For ROCm kernels this usually means the hot loop is built around `v_mfma_*` or CDNA4 scaled-MFMA instructions, and the optimization target shifts from "load fewer bytes" to "keep MFMA instructions issued with useful operands."

## Diagnosis

| Signal | Interpretation |
|--------|----------------|
| HBM bandwidth is below peak but MFMA utilization is high | Memory is probably not the limiting resource |
| Increasing vectorized load width does not improve time | Operand delivery is already adequate |
| Larger tiles improve reuse but reduce occupancy sharply | Register/LDS pressure is now the tradeoff |
| Epilogue fusion changes runtime significantly | Non-MFMA instructions are competing with the math pipeline |

## Optimization Loop

1. **Choose an MFMA-native tile shape.** Match the M/N/K blocking to supported operand types and accumulator layout rather than forcing a CUDA Tensor Core tile shape onto AMD wavefronts.
2. **Keep accumulator fragments in registers.** Use register tiling to avoid round-trips through LDS, but watch VGPR pressure and spill risk.
3. **Interleave memory and MFMA.** Double-buffer global-to-LDS and LDS-to-register movement so operand staging does not drain the MFMA issue queue.
4. **Move format conversion to the boundary.** For FP8/FP4/block-scale paths, avoid dequantizing earlier than needed; packed values and scale metadata should survive until the MFMA-facing path when possible.
5. **Fuse only cheap epilogues.** Bias, activation, and scale application are attractive, but large epilogues can turn a compute-bound GEMM into a mixed pipeline bottleneck.

## CDNA4 Block-Scale Variant

CDNA4 adds scaled-MFMA paths for F8/F6/F4 style operands. The compute-bound pattern changes in two ways:

- The kernel must schedule scale metadata loads and packed-value unpacking alongside matrix-core instructions.
- The best layout may be the one that minimizes format shuffling at the MFMA boundary, even if it is less convenient for the producer or consumer tensor.

CK block-scale ABQuant work illustrates this shift: PRs around A4W4 support and ABQuant optimization focus less on generic GEMM structure and more on where packed FP4/FP8 values, preshuffled B data, and scale tensors reside.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `doc-cdna4-isa` | MFMA and scaled-MFMA instruction reference |
| `doc-cdna4-whitepaper` | CDNA4 matrix-core and block-scale architecture context |
| `pr-composable_kernel-3709` | MX GEMM support for non-preshuffled and RCR layouts |
| `pr-composable_kernel-3603` | A4W4 FP4 block-scale GEMM support |
| `pr-composable_kernel-3620` | ABQuant block-scale GEMM optimization evidence |

## Retrieval Cues

Retrieve this page for compute-bound GEMM, MFMA utilization, accumulator tiling, FP4/FP8 block-scale math, epilogue fusion tradeoffs, or when memory-bound optimization stops improving a ROCm kernel.

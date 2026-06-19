---
id: technique-occupancy-tuning
title: Occupancy Tuning on ROCm
type: wiki-technique
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [occupancy-tuning, occupancy, vgpr, lds, wavefront, optimization, rocm]
confidence: source-reported
techniques: [occupancy-tuning, register-tiling, double-buffering]
hardware_features: [wavefront, lds, mfma]
related:
  - hw-wavefront
  - hw-lds
  - hw-mfma-matrix-core
  - technique-register-tiling
  - technique-double-buffering
  - pattern-memory-bound-amd
sources:
  - doc-hip-programming-guide
  - doc-cdna4-isa
  - pr-composable_kernel-3729
  - pr-composable_kernel-3709
reproducibility: concept
---

# Occupancy Tuning on ROCm

Occupancy tuning is the tradeoff between how many wavefronts can live on a compute unit and how much work each wavefront can keep in registers and LDS. On AMD GPUs, the useful question is not simply "maximize occupancy"; it is whether additional resident waves hide memory and pipeline latency better than larger tiles, deeper prefetch, or more VGPR-resident data.

## What Limits Occupancy

| Limiter | Why it matters | Common symptom |
|---------|----------------|----------------|
| VGPRs per thread | Large accumulator fragments, staging registers, and epilogue temporaries reduce resident waves | MFMA-heavy kernel has high theoretical math rate but low active waves |
| LDS per workgroup | Double-buffered tiles and padded swizzled layouts reduce workgroups per CU | Memory latency returns when K tiles get larger |
| Workgroup size | More waves per workgroup increase parallelism but may consume more scheduling slots | Good bandwidth at small shapes, poor tail behavior |
| Instruction mix | MFMA, LDS, vector memory, and scalar control need balanced scheduling | Pipeline bubbles or long scoreboard stalls |

## ROCm-Specific Heuristics

1. **Start from wavefront granularity.** AMD wavefronts have 64 lanes, so CUDA warp-level intuitions need adjustment. A block that looks like "four warps" in CUDA terms may map to fewer independent scheduling units after the ROCm tiling is rewritten.
2. **Measure VGPR pressure before increasing tile size.** Register tiling improves reuse, but every accumulator fragment and vectorized-load buffer can reduce resident waves.
3. **Treat LDS as both bandwidth and capacity.** Padding and swizzling can remove bank conflicts, but the extra bytes may reduce workgroups per CU.
4. **Tune occupancy together with prefetch depth.** A low-occupancy kernel can still run well if double-buffering or asynchronous loads keep MFMA instructions fed.
5. **Use architecture-specific fallbacks.** CDNA4 scaled-MFMA and block-scale paths often need scale metadata and packed-value conversion registers that a CDNA3 FP16/BF16 GEMM path does not need.

## Decision Pattern

```text
if kernel is memory-bound:
  add resident waves until HBM/LDS latency is hidden
  reduce tile size or staging buffers if occupancy is the limiter
else if kernel is MFMA-bound:
  prefer enough occupancy to hide MFMA latency, not maximum occupancy
  spend registers on accumulator reuse and epilogue fusion
else if kernel has irregular pages/groups:
  preserve occupancy for load imbalance and tail tiles
  avoid tile shapes that only work for full waves
```

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `doc-hip-programming-guide` | HIP execution and occupancy concepts |
| `doc-cdna4-isa` | Instruction-level constraints for MFMA, LDS, and CDNA4 paths |
| `pr-composable_kernel-3729` | FMHA async transpose-load tuning for MI355-style attention shapes |
| `pr-composable_kernel-3709` | MX GEMM layout variants where packed low-precision data changes occupancy tradeoffs |

## Retrieval Cues

Retrieve this page for occupancy, waves per CU, VGPR pressure, LDS capacity, tile-size tradeoffs, MFMA latency hiding, or why a ROCm kernel slows down after increasing register tiling.

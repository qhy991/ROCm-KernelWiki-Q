---
id: technique-pr-hipblaslt-44626
title: "PR Insight: hipblaslt #44626 - [ROCm][AITER] Use pre-shuffled FP8 GEMM for Quark "
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - optimization
  - rocm-kernel
confidence: inferred
sources:
  - pr-hipblaslt-44626
---

# Analysis of PR #44626 in hipblaslt

## Summary
This PR (`[ROCm][AITER] Use pre-shuffled FP8 GEMM for Quark per-channel attention weights`) introduces changes to the hipblaslt repository. It has been automatically analyzed for optimizations related to AMD ROCm CDNA architectures.

## Technical Details
- **Hardware focus**: General CDNA improvements.
- **Kernel types**: Inferred from changes (e.g. GEMM, Attention).
- **Techniques**: Contains enhancements that improve HBM bandwidth utilization and register allocation.

> Note: This is an automatically generated insight page based on PR metadata.

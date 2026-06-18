---
id: technique-pr-hipblaslt-330
title: "PR Insight: hipblaslt #330 - [WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx9"
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
  - pr-hipblaslt-330
---

# Analysis of PR #330 in hipblaslt

## Summary
This PR (`[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE`) introduces changes to the hipblaslt repository. It has been automatically analyzed for optimizations related to AMD ROCm CDNA architectures.

## Technical Details
- **Hardware focus**: General CDNA improvements.
- **Kernel types**: Inferred from changes (e.g. GEMM, Attention).
- **Techniques**: Contains enhancements that improve HBM bandwidth utilization and register allocation.

> Note: This is an automatically generated insight page based on PR metadata.

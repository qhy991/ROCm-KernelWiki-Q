---
id: technique-pr-hipblaslt-3
title: "PR Insight: hipblaslt #3 - [Hipblaslt] [Subtiling] Add non-uniform partition "
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
  - pr-hipblaslt-3
---

# Analysis of PR #3 in hipblaslt

## Summary
This PR (`[Hipblaslt] [Subtiling] Add non-uniform partition size to Logical Scheduler`) introduces changes to the hipblaslt repository. It has been automatically analyzed for optimizations related to AMD ROCm CDNA architectures.

## Technical Details
- **Hardware focus**: General CDNA improvements.
- **Kernel types**: Inferred from changes (e.g. GEMM, Attention).
- **Techniques**: Contains enhancements that improve HBM bandwidth utilization and register allocation.

> Note: This is an automatically generated insight page based on PR metadata.

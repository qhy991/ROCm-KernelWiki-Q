---
id: technique-pr-hipblaslt-366
title: "PR Insight: hipblaslt #366 - opt(gemm): add AITER MXFP4 preshuffle fast path"
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
  - pr-hipblaslt-366
---

# Analysis of PR #366 in hipblaslt

## Summary
This PR (`opt(gemm): add AITER MXFP4 preshuffle fast path`) introduces changes to the hipblaslt repository. It has been automatically analyzed for optimizations related to AMD ROCm CDNA architectures.

## Technical Details
- **Hardware focus**: General CDNA improvements.
- **Kernel types**: Inferred from changes (e.g. GEMM, Attention).
- **Techniques**: Contains enhancements that improve HBM bandwidth utilization and register allocation.

> Note: This is an automatically generated insight page based on PR metadata.

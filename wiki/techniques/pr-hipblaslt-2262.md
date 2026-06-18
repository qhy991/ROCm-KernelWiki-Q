---
id: technique-pr-hipblaslt-2262
title: "PR Insight: hipblaslt #2262 - [AIROCMLIR-71] Add gemm+gemm and conv+gemm support"
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
  - pr-hipblaslt-2262
---

# Analysis of PR #2262 in hipblaslt

## Summary
This PR (`[AIROCMLIR-71] Add gemm+gemm and conv+gemm support to quickTuningGen.py`) introduces changes to the hipblaslt repository. It has been automatically analyzed for optimizations related to AMD ROCm CDNA architectures.

## Technical Details
- **Hardware focus**: General CDNA improvements.
- **Kernel types**: Inferred from changes (e.g. GEMM, Attention).
- **Techniques**: Contains enhancements that improve HBM bandwidth utilization and register allocation.

> Note: This is an automatically generated insight page based on PR metadata.

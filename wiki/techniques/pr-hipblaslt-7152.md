---
id: technique-pr-hipblaslt-7152
title: "PR Insight: hipblaslt #7152 - Add fused_gemm_benchmark.py: fused two-GEMM SwiGLU"
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
  - pr-hipblaslt-7152
---

# Analysis of PR #7152 in hipblaslt

## Summary
This PR (`Add fused_gemm_benchmark.py: fused two-GEMM SwiGLU kernel benchmark`) introduces changes to the hipblaslt repository. It has been automatically analyzed for optimizations related to AMD ROCm CDNA architectures.

## Technical Details
- **Hardware focus**: General CDNA improvements.
- **Kernel types**: Inferred from changes (e.g. GEMM, Attention).
- **Techniques**: Contains enhancements that improve HBM bandwidth utilization and register allocation.

> Note: This is an automatically generated insight page based on PR metadata.

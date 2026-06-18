---
id: doc-ck-readme
title: Composable Kernel README
type: source-doc
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [ck-dsl, composable_kernel, gemm]
date: '2026-01-01'
url: https://github.com/ROCm/composable_kernel/blob/develop/README.md
source_category: upstream-code
languages: [ck-dsl]
confidence: verified
---

# Composable Kernel (CK)

CK is AMD's open-source performance library for tensor operations on CDNA/RDNA GPUs. It provides:

- Templated GPU kernels with MFMA-centric tiling
- Instance factories for GEMM, conv, reduction, attention, MoE
- CK Tile DSL for composable tile algorithms

## When to Use CK

| Scenario | Recommendation |
|----------|----------------|
| Production GEMM/attention on MI300X | Use CK/CK Tile instances or wrappers |
| Custom fused kernel with MFMA | Extend CK Tile pipeline |
| Quick HIP prototype | Start HIP, migrate hot path to CK Tile |

## Related Wiki Pages

- [CK DSL](../../wiki/languages/ck-dsl.md)

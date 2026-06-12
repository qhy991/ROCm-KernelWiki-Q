---
id: doc-ck-structure
title: Composable Kernel Repository Structure
type: source-doc
architectures: [cdna2, cdna3, cdna4]
tags: [ck-dsl, composable-kernel, gemm, attention]
date: '2026-01-01'
url: https://github.com/ROCm/composable_kernel/blob/develop/include/ck/README.md
source_category: upstream-code
techniques: [ck-tile-programming]
languages: [ck-dsl]
confidence: verified
---

# Composable Kernel Layout

Composable Kernel (CK) organizes high-performance GPU kernels in layered abstractions:

- `include/ck/tensor_operation/gpu/` — device GEMM/conv/reduce instances
- `include/ck_tile/` — newer tile programming model (CK Tile)
- `example/` — runnable references for GEMM, attention, MoE, grouped GEMM
- `library/` — pre-instantiated kernel instances used by frameworks

## Practical Navigation

1. Start from the closest `example/ck_tile/` or `example/` demo for your kernel type
2. Trace into `device/` instance headers for block/warp tiling choices
3. Use CK Tile when writing new kernels; legacy CK for existing instance catalogs

## Related Wiki Pages

- [CK DSL](../../wiki/languages/ck-dsl.md)
- [CK Tile Programming](../../wiki/techniques/ck-tile-programming.md)

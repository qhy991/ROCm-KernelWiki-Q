---
id: kernel-ck-tile-gemm-rocm
title: CK Tile GEMM on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [gemm, ck-tile, composable_kernel, programming-model, tiling, optimization]
confidence: source-reported
kernel_types: [gemm]
languages: [hip-cpp, ck-dsl]
techniques: [ck-tile-programming, mfma-scheduling, vectorized-load, occupancy-tuning]
hardware_features: [mfma, lds, wavefront]
related:
  - technique-ck-tile-programming
  - technique-vectorized-load
  - kernel-gemm-mfma-rocm
sources:
  - doc-ck-readme
  - doc-ck-structure
  - pr-composable_kernel-3554
  - pr-composable_kernel-3571
  - pr-composable_kernel-3643
  - pr-composable_kernel-3672
reproducibility: concept
---

# CK Tile GEMM on ROCm

CK Tile is the lower-level programming model in Composable Kernel for building GEMM-like kernels from explicit tile transforms, pipelines, and tensor views. It is the most useful source family when extracting reusable operator-development patterns instead of only one finished library call.

## Development Pattern

```text
problem layout -> tile distribution -> LDS/global descriptors
               -> pipeline policy -> block/wave tile shape
               -> epilogue and type conversion
```

The tile model exposes enough detail to reason about register tiling, LDS staging, vector width, and wave tile compatibility. This makes it a good external database source for generating kernel templates, not just operator descriptions.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `doc-ck-readme` | CK scope and supported operator families |
| `doc-ck-structure` | Codebase structure for tile-engine examples and instances |
| `pr-composable_kernel-3554` | Basic tile-engine CI maintenance |
| `pr-composable_kernel-3571` | API reference cleanup around CK Tile |
| `pr-composable_kernel-3643` | Supported warp tile updates in the tile engine |
| `pr-composable_kernel-3672` | Intrinsic argument compatibility fixes |

## Database Use

Index this page under `programming_model=ck_tile`, `backend=composable_kernel`, and `operator=gemm`. It should be retrieved before more specialized pages such as FP8 block-scale GEMM, split-K GEMM, or paged attention when a generator needs the tile-engine vocabulary.


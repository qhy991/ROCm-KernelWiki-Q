---
id: kernel-hipsparselt-spmm-rocm
title: hipSPARSELt SpMM (Sparse × Dense) on ROCm
type: wiki-kernel
architectures: [cdna3]
tags: [gemm, spmm, sparse, hipsparselt, structured-sparsity, bbs, tuning, memory-bound]
confidence: source-reported
kernel_types: [gemm, spmm]
languages: [hip-cpp, assembly]
techniques: [shape-based-kernel-selection]
hardware_features: [mfma, lds]
related:
  - kernel-cdna4-hipblaslt-scaled-mfma-gemm
  - kernel-ck-tile-gemm-rocm
sources:
  - pr-rocm-libraries-8216
  - pr-rocm-libraries-8217
  - pr-rocm-libraries-8219
  - pr-rocm-libraries-8265
  - pr-rocm-libraries-8404
  - pr-rocm-libraries-8405
  - pr-rocm-libraries-8754
  - pr-rocm-libraries-8750
reproducibility: concept
---

# hipSPARSELt SpMM (Sparse × Dense) on ROCm

hipSPARSELt is AMD's library for **structured-sparsity** linear algebra — primarily
SpMM (sparse × dense matrix multiply) and SDDMM on 2:4-pruned matrices. Unlike rocSPARSE
(general sparse formats, CSR/BSR/COO, solvers), hipSPARSELt targets the narrow but
high-value 2:4 block-sparsity pattern used to compress LLM weights. Crucially, it does
**not** write its own kernels from scratch: it reuses the **Tensile** codegen + tuning
machinery from hipBLASLt, just with a sparse-operand solution family.

## Shared Tensile Backend

hipSPARSELt's SpMM kernels live under `spmm/Tensile/Logic/asm_full/aquavanjaram/gfx942/GridBased/`.
`aquavanjaram` is the internal codename for the gfx942 / MI300 family. The solution YAMLs
are produced by the same Tensile/TensileLite pipeline as hipBLASLt's dense GEMM, so the
naming grammar is identical — only the sparse-operand segments differ.

`pr-8754` made this explicit by **removing the legacy standalone HIP kernel launcher**
(`spmm/hip/{kernel_launcher,kernel_arguments,hip_solution_adapter}.cpp`), leaving Tensile
as the sole backend. The hipSPARSELt ↔ hipBLASLt code paths now share their codegen root.

## Solution Naming: SPA / SPB (Sparse Operand)

Reading `aquavanjaram_Cijk_Alik_Bljk_BBS_BH_Bias_FDMN_SPA_AS_SAV.yaml`:

| Segment | Meaning |
|---------|---------|
| `aquavanjaram` | gfx942 / MI300 family codename |
| `Cijk_Alik_Bljk` | Output C(i,j), reduction over k; A row-k layout, B col-jk layout |
| `BBS` / `HHS` | Precision family (same BF16 variants as hipBLASLt) |
| `BH` | Block hierarchy |
| `Bias` | Epilogue bias |
| `FDMN` | Sparse-domain marker (sparse value/index handling) |
| **`SPA` / `SPB`** | **Which operand is sparse** — A (M×K, left) or B (K×N, right) |
| `AS` / `A_S` | Sparse storage/compression layout |
| `SAV` | Scale-A-Vector |

The `SPA` vs `SPB` split is the architecturally interesting axis: **LLM inference with
sparse weights almost always wants `SPB`** (the weight matrix B is the pruned one), so the
SPB solution libraries receive the heaviest tuning. The 2026-06 PR wave
(`pr-8216/8217/8219/8265/8404/8405`) retunes the gfx942 BBS/HHS × {TN,NN,NT,TT} × {SPA,SPB}
GridBased logic — the full transpose × sparse-side matrix.

## Why It Is Memory-Bound

A 2:4-sparse SpMM does half the FMAs of the dense GEMM but still must load the sparse
index structure (which lanes are non-zero per 4-element group) on top of the values. The
arithmetic intensity therefore drops, and the kernel becomes **memory-bandwidth bound** —
the dominant cost is streaming the compressed weight, not MFMA throughput. This is why
hipSPARSELt invests in per-shape GridBased solution selection rather than a single
hand-optimized kernel: the right tiling/pipeline depends on the sparsity layout and the
dense-opponent shape.

## Evidence Map

| Source | What it contributes |
|--------|---------------------|
| `pr-rocm-libraries-8216` | gfx942 BBS/HHS TN SPA/SPB GridBased logic; reveals full solution-naming grammar |
| `pr-rocm-libraries-8217` | gfx942 BBS/HHS NN SPA/SPB tuning |
| `pr-rocm-libraries-8219` | gfx942 BBS/HHS NT SPA tuning |
| `pr-rocm-libraries-8265` | gfx942 BBS/HHS NT SPB tuning |
| `pr-rocm-libraries-8404` | gfx942 BBS/HHS TT SPA tuning |
| `pr-rocm-libraries-8405` | gfx942 BBS/HHS TT SPB tuning |
| `pr-rocm-libraries-8754` | Removes legacy HIP kernel launcher; Tensile becomes sole backend |
| `pr-rocm-libraries-8750` | FFM + SpMM test coverage (test_categories) |

## Database Use

Index this page under `operator=spmm`, `backend=hipsparselt`, `sparsity=2:4`,
`arch=cdna3`. Retrieve it for queries about structured-sparse GEMM, sparse LLM weight
inference, or the hipSPARSELt↔hipBLASLt shared Tensile lineage. Pair with
`kernel-cdna4-hipblaslt-scaled-mfma-gemm` when the question is "how do ROCm GEMM-family
libraries share infrastructure" — both decode the same Tensile solution grammar with
different operand specializations (scaled-MFMA there, sparse here).

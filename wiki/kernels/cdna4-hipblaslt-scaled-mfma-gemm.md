---
id: kernel-cdna4-hipblaslt-scaled-mfma-gemm
title: CDNA4 (gfx950/MI350) Scaled-MFMA GEMM in hipBLASLt
type: wiki-kernel
architectures: [cdna4]
tags: [gemm, hipblaslt, fp8, fp4, block-scale, scaled-mfma, optimization, tuning, stream-k, subtile]
confidence: source-reported
kernel_types: [gemm]
languages: [hip-cpp, assembly]
techniques: [double-buffering, subtile, stream-k, block-scale, shape-based-kernel-selection]
hardware_features: [scaled-mfma, block-scale, mfma]
related:
  - kernel-ck-tile-gemm-rocm
  - kernel-gemm-mfma-rocm
sources:
  - pr-rocm-libraries-8568
  - pr-rocm-libraries-8780
  - pr-rocm-libraries-8845
  - pr-rocm-libraries-8918
  - pr-rocm-libraries-9161
  - pr-rocm-libraries-8973
  - pr-rocm-libraries-8449
  - pr-rocm-libraries-9056
reproducibility: concept
---

# CDNA4 (gfx950/MI350) Scaled-MFMA GEMM in hipBLASLt

On CDNA4 (gfx950 / MI350 / MI355), hipBLASLt ships GEMM solutions that exploit the
**scaled-MFMA** matrix cores — hardware block-scaling that natively accepts FP8 and
microscaled (MXFP4) inputs with per-block scale factors, avoiding the dequant round-trip
to FP16/BF16. The library is driven by **Tensile/TensileLite**, which generates and tunes
per-arch solution YAML files. This page decodes the solution naming, the precision/scale
mode matrix, and the Split-K / subtile scheduling strategy visible in the 2026-06 → 2026-07
monorepo tuning wave.

## Solution Naming Convention

hipBLASLt/Tensile solution files follow a dense positional grammar. Reading
`gfx950_Cijk_Alik_Bljk_F8BS_BH_BiasSB_HAS_SAB_SAV_UserArgs.yaml`:

| Segment | Meaning |
|---------|---------|
| `Cijk_{Ailk\|Alik}_{Bljk\|Bjlk}` | Output C indexed (i,j); reduction over k; A/B matrix layouts (e.g. A row-major in k, B column-major in jk) |
| `BBS\|HHS\|HSS\|SS\|BSS` | BF16 precision / scale-grouping family |
| `F8BS\|F8B8BS\|B8F8BS\|F8F8S` | FP8 block-scaled family (which operand carries the block scale) |
| `BH` | Block hierarchy (cooperative tile distribution) |
| `Bias{S\|SB\|SH}` | Epilogue bias with its own scale type |
| `HAS` | Hardware Accumulator Scale — routes the block scale through the MFMA unit instead of an epilogue multiply |
| `SAB\|SAV\|SABV` | Where scales are applied (A / B / Vec) |

The `HAS` + `SAB/SAV` combination is the CDNA4 differentiator: it is what makes
scaled-MFMA faster than an equivalent FP8 kernel with a software epilogue rescale.

## Precision / Scale Mode Matrix

| Mode | Use case | Evidence |
|------|----------|----------|
| BF16 (`BBS`/`HHS`/`SS`) | Baseline dense training/inference | `pr-8568`, `pr-8449` |
| FP8 block-scale (`F8BS`, `F8B8BS`) | Low-precision inference, LLM decode | `pr-8845`, `pr-8918` |
| MXFP4 `matmul_subtile_fp4_scaleAlphaVec` | Lowest-bit LLM weight/activation | `pr-8780` |

The MXFP4 path runs on **subtile** kernels with asymmetric 128×192 / 192×128 tiles and
relu/gelu/swish epilogues; `pr-8780` records a release-day accuracy regression on those
exact tiles, confirming the subtile MXFP4 path is the active frontier.

## Split-K Strategy: SK3 → SK5

`pr-8568` switches gfx950 device-library equality kernels from **SK3 to SK5** in
default-OFF mode. SK5-OFF still executes the SK3 code path at runtime, but the YAML
plumbing enables future **hybrid-mode scheduling** (mixing split-K and non-split-K
strategies per shape). This is a runtime-dispatch preparation, not a perf switch itself.

## Subtile + Global Prefetch (gl2)

`pr-9161` adds **prefetch gl2** (global level-2) support to the subtile kernel in
TensileLite's `Subtile/{InstructionEmitter,Kernel,LogicalScheduler}` and
`KernelWriterAssembly`. This is the double-buffering / software-pipelining lever for
subtile GEMMs — overlapping the next global load with current MFMA. Bench coverage lands
on gfx1250 subtile BF16, indicating the optimization is being forward-ported to the next
CDNA generation.

## Tuning Cadence

The 2026-06 → 2026-07 wave shows per-device tuning drops keyed by device ID
(`gfx950_id75a3`, `id75a8`) — hipBLASLt retunes the Equality/BBS/F8BS solution libraries
per MI350 SKU. `pr-8973` ("mi350P performance gap") and `pr-9056` ("Tuning Equality for
gfx950_id75a3") are representative of the continuous perf-gap closure loop.

## Evidence Map

| Source | What it contributes |
|--------|---------------------|
| `pr-rocm-libraries-8568` | SK3→SK5 split-K strategy migration; reveals full BF16/FP8 solution YAML matrix |
| `pr-rocm-libraries-8780` | MXFP4 subtile `scaleAlphaVec` tile shapes (128×192/192×128) and epilogue set |
| `pr-rocm-libraries-8845` | gfx950 id75a3 BBS/F8BS Equality library tuning drop |
| `pr-rocm-libraries-8918` | gfx950 id75a3 F8BS / F8B8BS TN Equality updates |
| `pr-rocm-libraries-9161` | Subtile gl2 prefetch (double-buffering) in TensileLite |
| `pr-rocm-libraries-8973` | MI350 performance-gap fix (retuning loop) |
| `pr-rocm-libraries-8449` | gfx950 HHS NN TN tuning |
| `pr-rocm-libraries-9056` | gfx950 id75a3 Equality perf tuning |

## Database Use

Index this page under `operator=gemm`, `backend=hipblaslt`, `arch=cdna4`,
`precision∈{fp8,mxfp4}`. Retrieve it when a query concerns low-precision GEMM on MI350,
block-scaled MFMA, or hipBLASLt/Tensile solution selection. It should precede per-shape
benchmark pages, which would attach concrete `performance_claims` to the solution
families described here.

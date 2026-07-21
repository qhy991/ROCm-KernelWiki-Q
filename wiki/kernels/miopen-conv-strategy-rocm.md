---
id: kernel-miopen-conv-strategy-rocm
title: MIOpen Convolution Kernel Strategy on ROCm
type: wiki-kernel
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [conv, miopen, composable_kernel, winograd, fusion, optimization, tuning, runtime-dispatch]
confidence: source-reported
kernel_types: [conv]
languages: [hip-cpp]
techniques: [shape-based-kernel-selection, runtime-dispatch]
hardware_features: [mfma, lds]
related:
  - kernel-ck-tile-gemm-rocm
  - kernel-cdna4-hipblaslt-scaled-mfma-gemm
sources:
  - pr-rocm-libraries-8618
  - pr-rocm-libraries-7637
  - pr-rocm-libraries-6439
  - pr-rocm-libraries-8722
  - pr-rocm-libraries-8988
  - pr-rocm-libraries-8426
  - pr-rocm-libraries-9050
reproducibility: concept
---

# MIOpen Convolution Kernel Strategy on ROCm

MIOpen is AMD's DNN primitive library (the cuDNN analog). The key thing to understand
about its convolution path is that **MIOpen is a solver dispatcher, not a single kernel**:
at runtime it selects among multiple convolution solver families — some hand-written, some
generated, some imported from Composable Kernel (CK) — via a Find/GetWorkspaceSize
heuristic backed by a per-architecture tuning database. A query about "MIOpen conv
performance" is really a question about which solver won the selection for that shape.

## Multi-Source Conv Solvers

| Solver family | Origin | Where it wins |
|---------------|--------|---------------|
| Winograd | MIOpen hand-written | Small-filter conv (3×3), forward; extended to gfx120x and depthwise NHWC |
| `ConvOclDirectFwd` | MIOpen direct conv (OpenCL→HIP port) | Direct forward conv, older archs |
| CK `GroupedConv` | Imported from Composable Kernel | Grouped convolution |

The interesting architectural fact is the **third row**: MIOpen does not statically link
CK — it `dlopen`s a per-architecture runtime library and dispatches to CK's grouped-conv
solver at runtime.

## MIOpen → CK Runtime Integration

`pr-8618` makes MIOpen **eagerly load** `libMIOpenCKGroupedConv_<arch>.so` during
`Handle` construction (`miopenCreate`) instead of lazily on the first CK call. The stated
reason is a classic one: the lazy `dlopen` showed up as an unpredictable mid-run stall the
first time a workload reached the CK solver (via `GetWorkspaceSize`/`Find`). Moving the
one-time load to handle creation makes the cost land at the expected setup point.

This reveals the full call chain for grouped conv on ROCm:

```text
hipDNN  →  MIOpen Handle  →  dlopen libMIOpenCKGroupedConv_<arch>.so
                              →  CK GroupedConv solver  →  MFMA
```

`pr-8988` / `pr-9105` (always register CK grouped-conv solvers in the host) and
`pr-9050` (disable CK grouped conv for gfx1250, later reverted by `pr-9491`) show this
integration is actively tuned per-arch — CK grouped-conv is the default on supported
architectures but gated where it regresses.

## OpenCL → HIP Migration

MIOpen historically shipped a dual backend: OpenCL kernels (`.cl`) and HIP kernels
(`.cpp`), each with its own per-gfx tuning database (`.fdb.txt.bz2`). `pr-7637` ports the
`ConvOclDirectFwd` direct-convolution solver from OpenCL to HIP — deleting
`MIOpenConvDirUni.cl` (774 lines) and adding `MIOpenConvDirUniHip.cpp`. This is part of a
sustained effort to **remove the OpenCL backend dependency** from MIOpen, consolidating on
HIP so that all solvers share one compilation path and one tuning DB format.

## Winograd

`pr-6439` (Add gfx120x support to Winograd) and `pr-8722` (fix depthwise NHWC Winograd
conv) show the Winograd solver remains actively maintained for small-filter forward conv,
extended to new architectures and layout variants. Winograd is the hand-optimized path for
the shape class where its reduced multiplication count beats direct/CK approaches.

## Evidence Map

| Source | What it contributes |
|--------|---------------------|
| `pr-rocm-libraries-8618` | MIOpen eager-loads `libMIOpenCKGroupedConv_<arch>.so` at handle creation; reveals hipDNN→MIOpen→CK chain |
| `pr-rocm-libraries-7637` | `ConvOclDirectFwd` OpenCL→HIP port; reveals dual-backend history and `.fdb.txt.bz2` tuning DBs |
| `pr-rocm-libraries-6439` | Winograd extended to gfx120x |
| `pr-rocm-libraries-8722` | Depthwise NHWC Winograd conv fix |
| `pr-rocm-libraries-8988` | Always register CK grouped-conv solvers in host |
| `pr-rocm-libraries-8426` | Softmax kernel NaN fix (beta=0) — sibling non-conv primitive |
| `pr-rocm-libraries-9050` | Per-arch gating of CK grouped conv (gfx1250) |

## Database Use

Index this page under `operator=conv`, `backend=miopen`, `solver∈{winograd,direct,ck}`.
Retrieve it for queries about MIOpen convolution, the MIOpen↔Composable Kernel runtime
integration, or ROCm's layered library architecture (hipDNN→MIOpen→CK). It is the natural
bridge between the CK tile-engine pages (`kernel-ck-tile-gemm-rocm`) and the higher-level
DNN frontend.

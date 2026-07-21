---
id: technique-tensilelite-codegen
title: GEMM Kernel Code Generation (TensileLite + Stinkytofu)
type: wiki-technique
architectures: [cdna3, cdna4]
tags: [hipblaslt, tensilelite, codegen, code-generation, programming-model, stinkytofu, assembly-emission]
confidence: source-reported
techniques: [code-generation, instruction-selection, assembly-emission]
hardware_features: [mfma, lds]
kernel_types: [gemm]
related:
  - kernel-cdna4-hipblaslt-scaled-mfma-gemm
  - technique-ck-tile-programming
sources:
  - pr-rocm-libraries-7540
  - pr-rocm-libraries-8349
  - pr-rocm-libraries-6944
  - pr-rocm-libraries-7053
  - pr-rocm-libraries-8162
  - pr-rocm-libraries-8284
  - pr-rocm-libraries-8303
  - pr-rocm-libraries-7858
reproducibility: concept
---

# GEMM Kernel Code Generation (TensileLite + Stinkytofu)

The single most important thing to know about hipBLASLt's GEMM kernels: **they are not
hand-written**. The `.s` / `.cpp` device kernels under `projects/hipblaslt/` are emitted by
a two-stage code generator — **TensileLite** (Python) describes and schedules the kernel,
and **Stinkytofu** lowers it to GCN assembly. The solution YAML files decoded in
`kernel-cdna4-hipblaslt-scaled-mfma-gemm` are the *input* to this pipeline, not the output.
A huge fraction of hipBLASLt's recent monorepo PRs are not kernel tweaks — they are
**codegen features** (new scheduling primitives, new instruction emissions).

## TensileLite — Python Code Generator

Layout under `projects/hipblaslt/tensilelite/Tensile/`:

```
Common/             GlobalParameters, RequiredParameters, ValidParameters
Components/
  Subtile/          LogicalScheduler, ScheduleTypes, InstructionEmitter, Kernel
  MAC_F32.py        MFMA accumulate + epilogue (e.g. VOPD dual-issue)
SolutionStructs/    Solution abstraction
KernelWriter.py
KernelWriterAssembly.py
```

The flow is **YAML solution → parameter binding → LogicalScheduler schedule →
InstructionEmitter → KernelWriter → HIP/asm text**. Adding a new optimization usually
means adding a parameter in `Common/`, a scheduling case in `Subtile/LogicalScheduler`,
and an emission rule in `KernelWriter*` — not editing a kernel.

### LogicalScheduler

`pr-7540` (stage-tagged TypeAlias for the LogicalScheduler pipeline) and `pr-8162`
(StreamK=5 hybrid kernel + **Tile Scheduling tri-state**) show the scheduler is where
pipeline stages and tile-split policy are decided. The scheduler emits an ordered
instruction stream; it is the analogue of an LLVM scheduler, but operating on the
tile/MFMA vocabulary directly.

### Emitted Optimizations

Codegen features that appear as PRs (each is a new thing the generator can emit, not a
kernel edit):

| Feature | PR | What it emits |
|---------|----|---------------|
| Global Read/Write | `pr-6944` | Wide global loads/stores |
| PrefetchAcrossPersistent | `pr-7053` | Prefetch in persistent-kernel mode |
| GL2Prefetch (32-bit inc) | `pr-8303` | Global-L2 prefetch with reduced SGPR pressure |
| DirectToLds metadata | `pr-8284` | CDNA4 direct-global-to-LDS path |
| VOPD `v_dual_fmac_f32` | `pr-8349` | VALU dual-issue for f32 MAC (in `MAC_F32.py`) |
| StreamK=5 hybrid | `pr-8162` | Tri-state tile scheduling |

The VOPD case (`pr-8349`) is illustrative: it touches `Common/{Global,Required,Valid}Parameters.py`,
`Components/MAC_F32.py`, and `SolutionStructs/Solution.py` — i.e. it registers a parameter,
adds the emission logic, and exposes it on the Solution. No kernel file is hand-edited.

## Stinkytofu — Assembly Backend

Stinkytofu is the lowering backend that takes TensileLite's IR and produces GCN assembly.
It works in terms of a **control-flow graph** and register state:

- `pr-7858`: support **function-call CFG** — the assembler can represent call/return
  structure, not just straight-line code.
- `pr-7844`: `s_wait_alu` insertion + half-aware **RegKey** — correct wait-state placement
  and register liveness tracking.
- `pr-8096`: `InitCIterWmma` **RegionClonePass** — marker-driven region cloning for WMMA
  init loops.
- `pr-8266`: rebuild barriers — barrier synchronization correctness.

So the division of labor is: TensileLite decides *what* to compute and in *what order*;
Stinkytofu makes that emit *correct, scheduled* assembly with proper waits and register
allocation.

## Why This Matters

Three implications for anyone reading hipBLASLt PRs:

1. **Most "perf" PRs are codegen PRs.** A title like "Add prefetch gl2 support for subtile
   kernel" means the *generator* can now emit gl2 prefetch — it applies across all subtile
   solutions, not one kernel.
2. **The solution YAML is the tuning surface.** Per-shape tuning is done by selecting and
   parameterizing YAML solutions, not by writing new kernels. The codegen runs offline/at
   build to produce the device library.
3. **CK is the hand-written contrast.** Composable Kernel (`technique-ck-tile-programming`)
   is the C++ template library where kernels *are* authored by humans. hipBLASLt and CK
   represent the two ROCm strategies: auto-generated (TensileLite) vs hand-templated (CK).

## Evidence Map

| Source | What it contributes |
|--------|---------------------|
| `pr-rocm-libraries-7540` | LogicalScheduler pipeline + ScheduleTypes (scheduling structure) |
| `pr-rocm-libraries-8349` | VOPD `v_dual_fmac` emission via MAC_F32 + parameter registration |
| `pr-rocm-libraries-6944` | Global Read/Write codegen feature |
| `pr-rocm-libraries-7053` | PrefetchAcrossPersistent codegen feature |
| `pr-rocm-libraries-8162` | StreamK=5 hybrid + tri-state tile scheduling |
| `pr-rocm-libraries-8284` | DirectToLds metadata (CDNA4) |
| `pr-rocm-libraries-8303` | GL2Prefetch with 32-bit increment (SGPR pressure) |
| `pr-rocm-libraries-7858` | Stinkytofu function-call CFG support |

## Database Use

Index this page under `technique=code-generation`, `backend=hipblaslt/tensilelite`,
`stage∈{schedule,emit,assemble}`. Retrieve it when a query concerns how hipBLASLt kernels
are produced, what a "tensilelite" or "stinkytofu" PR means, or the auto-generated vs
hand-written kernel contrast. It explains the PR-family that dominates the hipBLASLt
monorepo activity.

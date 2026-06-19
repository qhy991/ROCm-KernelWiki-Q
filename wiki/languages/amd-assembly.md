---
id: lang-amd-assembly
title: AMDGPU Assembly for ROCm Kernels
type: wiki-language
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [assembly, isa, rocm, programming, optimization]
confidence: source-reported
languages: [assembly, hip-cpp]
related:
  - hw-mfma-matrix-core
  - hw-dpp-cross-lane
  - hw-lds
  - kernel-conv-rocm
sources:
  - doc-cdna4-isa
  - pr-miopen-3702
  - pr-rocwmma-611
reproducibility: snippet
---

# AMDGPU Assembly for ROCm Kernels

Handwritten AMDGPU assembly appears when a library needs exact control over MFMA issue, VGPR/AGPR usage, LDS access, or solver-specific instruction sequences. In this wiki it is most visible in convolution and low-level matrix-core evidence.

## When Assembly Shows Up

| Use case | Reason |
|----------|--------|
| Winograd convolution solvers | Algorithm-specific data movement and instruction scheduling are difficult to express portably |
| Matrix-core microkernels | MFMA shape, accumulator layout, and operand packing may need exact ISA control |
| Compatibility hotfixes | Assembler macro behavior and metadata versions can change across ROCm releases |

## Maintenance Notes

MIOpen's Winograd Rage solver is a good example. `pr-miopen-3702` integrates two `.s` kernels plus generated `.inc` metadata and explicitly tests assembler behavior around `.altmacro` changes in ROCm 6.4. That is the maintenance cost of assembly: the code can be fast and precise, but the build path and assembler semantics become part of the kernel contract.

RDNA4 SWMMAC prototypes add another caution. `pr-rocwmma-611` is closed and depends on modified compiler support, so it is useful as research evidence for instruction-level exploration, not as a merged rocWMMA user feature.

## Retrieval Cues

Retrieve this page for `.s` kernels, AMDGPU ISA, MFMA assembly, assembler metadata, `.altmacro`, MIOpen Winograd, or handwritten ROCm kernels.

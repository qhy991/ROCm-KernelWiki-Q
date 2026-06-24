---
id: technique-pr-triton-498
title: "Triton Compilation Pipeline Debugging and IR Parsing"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, triton-rocm, optimization, programming]
confidence: inferred
sources:
  - pr-triton-498
---

# Triton Compilation Pipeline Debugging and IR Parsing

## Context and Motivation

Understanding how a high-level Triton kernel maps to AMD GPU Assembly (ISA) is crucial for performance tuning. Triton relies on MLIR for compilation, utilizing several intermediate representations (IR) to incrementally lower the kernel code. A typical compilation pipeline produces a large, monolithic text dump containing the IR state before and after various passes. When analyzing why a kernel suffers from register spilling, sub-optimal memory accesses, or poor instruction scheduling, developers must inspect these intermediate dumps.

PR [ROCm/triton#498](https://github.com/ROCm/triton/pull/498) introduces an IR dump parsing script that takes a combined dump of Triton IR (TTIR), Triton GPU IR (TTGIR), LLVM IR, and Assembly, and splits it into individual files per compiler pass. This significantly improves the ergonomics of debugging and optimizing Triton kernels for AMD CDNA architectures.

## Triton Compilation Pipeline on ROCm

Triton lowers Python-like kernels into GPU-executable code through the following fundamental stages:

1. **AST to TTIR (Triton IR):**
   The Python Abstract Syntax Tree (AST) is lowered to Triton's hardware-agnostic MLIR dialect (TTIR). Optimizations at this stage focus on high-level mathematical simplifications and control-flow flattening.

2. **TTIR to TTGIR (Triton GPU IR):**
   The TTIR is lowered to Triton GPU IR (TTGIR). This is the stage where the `triton-rocm` backend introduces block-level parallelism and distributed memory representations. This pass handles layout optimization, shared memory (LDS) allocations, and tile layouts.
   - *Optimization Insight:* Inspecting TTGIR helps identify whether matrix layouts are mapped efficiently to MFMA instructions or if expensive cross-lane operations (like `dpp` or `gws`) are implicitly being generated.

3. **TTGIR to LLVM IR:**
   The MLIR is further lowered to the AMDGPU dialect of LLVM IR. Target-specific intrinsics for ROCm (such as `llvm.amdgcn.mfma` for matrix cores and `llvm.amdgcn.ds` for LDS operations) are introduced here.
   - *Optimization Insight:* If `bank-conflict-padding` or `swizzling` is not working as expected, this IR layer reveals the exact pointer arithmetic and index mapping used for LDS access.

4. **LLVM IR to Assembly (ISA/GCN):**
   The LLVM compiler backend performs instruction selection, register allocation, and instruction scheduling, resulting in the final GPU assembly for CDNA architectures.
   - *Optimization Insight:* Excessive VGPR usage and register spilling, or poor scheduling of MFMA instructions intermixed with memory operations, become evident in the final ISA dump.

## Utilizing the IR Parsing Tool

When debugging a kernel, environment variables such as `MLIR_ENABLE_DUMP=1` can be used to generate a monolithic log of the IR before and after every pass. Without tooling, locating the exact transformation that introduced an inefficiency is difficult due to the sheer volume of output.

The parser script splits this monolithic text dump into a sequenced directory of files, where each file represents a single MLIR or LLVM pass. This approach enables developers to:
- Use standard `diff` tools to track the evolution of a specific loop, pointer operation, or block layout.
- Pinpoint exactly which pass performed suboptimal loop unrolling or failed to utilize vectorized loads.
- Correlate excessive VGPR allocations in the final assembly with the specific LLVM pass that failed to optimize variable lifetime ranges.

## Impact on Optimization Workflows

Isolating individual compilation passes is foundational for diagnosing `memory-bound` kernels or register-pressured applications on `cdna3` and `cdna4`. It allows compiler engineers and kernel optimization experts to build streamlined analysis workflows that:
- Track how the `swizzling` patterns evolve in TTGIR passes.
- Validate that hardware-specific mappings target features accurately without regressing performance.
- Evaluate the efficacy of experimental backend passes targeting new CDNA features (such as `scaled-mfma`) rapidly through a clean, diffable representation.

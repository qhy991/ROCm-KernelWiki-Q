---
id: wiki-technique-triton-608
title: "Triton Flash Attention: Constexpr SM Scale & ASM Optimizations"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, optimization, vgpr, occupancy-tuning, memory-bound, isa]
kernel_types: [flash-attention, attention]
languages: [triton-rocm, assembly]
confidence: inferred
---

# Triton Flash Attention: Constexpr SM Scale and ASM Improvements

## Overview

This technique analyzes optimizations applied to Flash Attention (FA) kernels within the ROCm Triton backend (referenced in [ROCm/triton PR #608](https://github.com/ROCm/triton/pull/608)). The primary improvements focus on the Softmax scaling phase of the Attention computation by promoting runtime tensor variables to compile-time constants, supplemented by targeted minor assembly-level refinements.

## Architectural & Code Analysis

### 1. Constexpr Softmax (SM) Scale Multiplication

In standard Scaled Dot-Product Attention, the intermediate query-key dot product ($Q \times K^T$) is multiplied by a scaling factor—commonly $1/\sqrt{d}$ where $d$ is the head dimension—before executing the Softmax operation. 

**Intent:**  
Traditionally, this scale factor is fetched from memory or passed as a runtime argument via a scalar or vector register. By setting the SM scale multiplication to be a `constexpr` (compile-time constant) in the kernel generator, the compiler can deeply optimize the math logic.

**Optimization Technique Applied:**  
- **Constant Folding:** The Triton/LLVM compiler embeds the constant scaling factor directly into the instruction stream as an immediate operand. 
- **Elimination of Loads/Registers:** Prevents the allocation of Vector General Purpose Registers (VGPRs) or Scalar General Purpose Registers (SGPRs) that would otherwise be required to hold the scale factor.

**Memory Bounds & Occupancy Impact:**  
- Flash attention on AMD CDNA architectures is highly sensitive to register pressure. High VGPR allocation directly limits the number of active wavefronts that can reside concurrently on a single Compute Unit (CU).
- By saving even a single VGPR (by not storing the scale parameter), the kernel can sometimes hit a tipping point that increases wavefront occupancy (e.g., from 4 to 5 active waves per SIMD unit).
- **Occupancy tuning** directly mitigates memory-bound bottlenecks by ensuring there are enough active threads to hide the latency of High Bandwidth Memory (HBM) fetches during the block loads of $Q$, $K$, and $V$ tiles.

### 2. Minor Assembly (ASM) Improvements

The PR pairs this change with "minor asm improvements," indicating direct tweaks to the generated AMDGCN ISA or the use of specific Triton inline assembly blocks (`tl.inline_asm_elementwise`).

**Intent:**  
To streamline the compute pipeline, reduce instruction count, and avoid stalls.

**Optimization Technique Applied:**  
- **Instruction Selection:** Optimizing how multiplication and scaling are lowered. For instance, using optimized `v_mul_f32` or fused instructions if applicable, avoiding inefficient operand combinations.
- **Pipeline Stalls Mitigation:** Hand-tuned inline assembly allows kernel authors to avoid subtle compiler heuristics that might introduce Read-After-Write (RAW) data hazards or unnecessary `v_mov` instructions.

## Key Takeaways for Triton/ROCm Developers

1. **Compile-time Constants Unlock Occupancy:** Always prefer compile-time constants (`constexpr` or `tl.constexpr`) over runtime variables for scaling factors and dimensional sizes. This acts as a catalyst for downstream compiler optimizations like constant folding and aggressive loop unrolling.
2. **VGPR Thriftiness:** On architectures like CDNA2 and CDNA3, VGPRs are precious. Treat them as a scarce resource; moving static constants out of VGPRs is a direct path to higher occupancy and better memory-latency hiding.

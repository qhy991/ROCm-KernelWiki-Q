---
id: technique-pr-triton-474
title: "LDS Swizzling for Transposed Dot Operands in Triton"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [swizzling, lds, gemm, triton-rocm, memory-bound, optimization]
confidence: inferred
sources: [pr-triton-474]
---

# LDS Swizzling for Transposed Dot Operands

## Overview

In matrix multiplication (GEMM) workloads, particularly when compiled via the Triton framework for AMD ROCm GPUs, managing Local Data Share (LDS) memory access patterns is critical for performance. PR #474 in the ROCm/triton repository addresses a specific performance bottleneck: severe bank conflicts when accessing transposed dot operands from LDS.

By enabling **LDS swizzling** (SMEM swizzling in Triton terminology) for these transposed operands, the memory access footprint is redistributed across the 32 LDS banks, eliminating bank conflicts and dramatically improving kernel throughput. The author reported a 10% performance improvement on internal models as a direct result of this change.

## The Problem: Transposed Access Patterns

When a matrix operand (e.g., Matrix B in a GEMM) is loaded into LDS, it is naturally laid out in a contiguous, row-major, or column-major format. If the subsequent `dot` (matrix multiplication) operation expects the operand in a transposed layout, threads within a wavefront will read the data in an opposite, strided order compared to the memory layout.

On CDNA architecture GPUs (such as CDNA2, CDNA3, and CDNA4):
- The Local Data Share (LDS) is divided into 32 independent memory banks.
- Each bank serves one address per clock cycle.
- A strided read—where the stride is a multiple of the number of banks (e.g., reading a column from a row-major tile)—causes multiple threads in the wavefront to request addresses mapping to the same bank.
- This creates an **LDS bank conflict**, forcing the hardware to serialize the memory requests and severely degrading the effective LDS bandwidth. 

Given that GEMM inner loops are heavily memory-bound by LDS read throughput to feed the MFMA (Matrix Fused Multiply-Add) units, these conflicts stall the compute pipeline.

## The Solution: Memory Swizzling

To resolve the bank conflicts associated with transposed access, the Triton compiler can apply a **swizzling** transformation to the LDS addresses.

### How Swizzling Works
Swizzling alters the mapping of logical addresses to physical LDS banks using an XOR-based permutation:
1. Instead of storing row `i` and column `j` linearly, the physical column (or bank index) is calculated by XORing the row index with the column index (e.g., `bank_id = (col ^ (row % swizzle_factor)) % 32`).
2. This non-linear mapping ensures that contiguous logical columns, which were previously mapping to the same bank, are scattered across different banks.
3. When the wavefront later performs the strided read (reading along the column), the requests map to distinct LDS banks, allowing all 32 threads to be serviced in a single cycle.

### Implementation in Triton
PR #474 extends Triton's AMD backend to explicitly enable this swizzling strategy for SMEM allocations tied to transposed dot operands. The compiler automatically applies the necessary XOR logic during both the load (global to shared) and the dot (shared to registers) phases.

## Hardware Implications

- **Architectures**: This optimization is applicable to AMD CDNA architectures (CDNA2, CDNA3, CDNA4), all of which utilize highly banked LDS designs to feed massive matrix engines (MFMA/Dual-CMA).
- **Compute Units**: By alleviating LDS stalls, the compute units can maintain higher occupancy and better instruction issue rates, ensuring that the MFMA instructions are fed with operands at maximum throughput.

## Sources
- [ROCm/triton PR #474: Enable swizzling SMEM for transposed dot operand](https://github.com/ROCm/triton/pull/474)

---
id: technique-pr-triton-546
title: "DotSlicing: AMDReorderInstructionPass Fix in Triton"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags:
  - optimization
  - rocm-kernel
  - scheduling
  - pipeline
confidence: inferred
sources:
  - pr-triton-546
---

# DotSlicing: AMDReorderInstructionPass Fix in Triton

## Overview
This technique document details a critical bug fix in Triton's AMD backend instruction scheduler (`AMDReorderInstructionPass`) concerning the reordering of `tt.load` instructions associated with sliced `tt.dot` operations (DotSlicing). When loads have multiple arguments (e.g., masks, secondary pointers, or cache modifiers), naive reordering algorithms may miscalculate dependencies or corrupt the instruction stream, leading to either compiler crashes or incorrect scheduling behavior.

## Architectural and Scheduling Context

### The Role of `AMDReorderInstructionPass`
In the Triton compilation pipeline for AMD GPUs (CDNA architectures), achieving high instruction-level parallelism (ILP) and effectively hiding memory latency are paramount. The `AMDReorderInstructionPass` acts as an instruction scheduler at the IR level. Its primary goal is to hoist global and shared memory loads (`tt.load`) as early as possible so that their latency overlaps with arithmetic operations—especially matrix multiplications (`tt.dot` or MFMA instructions).

### DotSlicing in Triton
DotSlicing refers to the transformation where a larger matrix multiplication (`tt.dot`) is sliced or chunked into smaller, more granular `tt.dot` operations. This is often done to:
1. Improve VGPR (Vector General Purpose Register) allocation and avoid register spilling.
2. Better pipeline memory loads (LDS/global) with compute (MFMA).
3. Accommodate specific hardware block sizes required by matrix core instructions.

When a `tt.dot` is sliced, its corresponding operand loads must also be sliced and reordered effectively.

## The Reordering Bug and Fix

### The Problem: Multi-Argument Loads
A standard `tt.load` in Triton can take several arguments beyond just the source pointer. For instance, it can take:
- **`ptrs`**: The base addresses.
- **`mask`**: A boolean tensor masking out-of-bounds loads.
- **`other`**: Default values for masked-out elements.
- **`cache` / `evict` / `is_volatile`**: Modifiers for memory hierarchy behavior.

During DotSlicing, if the `AMDReorderInstructionPass` attempts to reorder these loads but assumes they possess a simplified signature (i.e., treating them as single-argument or failing to track dependencies of the `mask` and `other` arguments), it can break the dependency graph. 
- **Incorrect Hoisting**: A load might be hoisted above the computation of its `mask` or `other` value, resulting in invalid memory accesses or corrupted execution.
- **Lost Operands**: Iteration logic assuming fixed operand positions could drop or misassign arguments during the reorder.

### The Solution
The pass logic was updated to correctly identify, preserve, and schedule *all* operands of a multi-argument load instruction during the reordering phase of DotSlicing. 
1. **Full Dependency Tracking**: Ensuring that all arguments to the load (`mask`, `other`) have their defining instructions properly tracked, preventing the load from being scheduled before its dependencies are available.
2. **Robust Operand Iteration**: Safe handling of the full parameter list of `tt.load` when generating the reordered IR.

## Performance Implications
While this is primarily a correctness fix, a robust instruction scheduler ensures that complex masked loads can be safely pipelined alongside `tt.dot` slices. This maximizes memory-to-compute overlap, directly impacting kernel bandwidth utilization and minimizing stalling on CDNA architectures.

## References
- PR Metadata: `pr-triton-546`
- Repository: `ROCm/triton`

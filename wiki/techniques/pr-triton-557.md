---
id: technique-pr-triton-557
title: "Triton: Enabling Atomic Memory Orderings for AMD GPUs"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [synchronization, memory, rocm, rocm-kernel]
languages: [triton-rocm]
confidence: inferred
sources: [pr-triton-557]
---

# Triton: Enabling Atomic Memory Orderings for AMD GPUs

## Overview
Pull Request [#557](https://github.com/ROCm/triton/pull/557) in the ROCm Triton repository enhances the code generation for atomic operations on AMD architectures. Prior to this change, LLVM atomic instructions generated for AMD GPUs via Triton were hardcoded with fixed memory ordering semantics (e.g., `monotonic` or `acq_rel`). This PR rectifies this by correctly mapping Triton's internal `MemSemantic` directly to `LLVM::AtomicOrdering`, thereby enabling developers to precisely control atomic visibility across threads, warps, and blocks.

## Architectural and Code Analysis

### Prior Limitations
Before PR #557, memory orderings for atomic operations in `TritonGPUToLLVM` were static:
- **`AtomicCASOpConversion`**: Hardcoded to `LLVM::AtomicOrdering::acq_rel` (Acquire/Release).
- **`AtomicRMWOpConversion`**: Hardcoded to `LLVM::AtomicOrdering::monotonic` (Monotonic).

Using static memory ordering constraints inherently limits both performance and correctness:
1. Hardcoding `monotonic` for `RMW` operations can cause subtle memory consistency bugs if a kernel requires cross-workgroup synchronization. Under `monotonic` ordering, other memory operations can be reordered around the atomic, which is fatal for spinlocks or serialized barriers.
2. Hardcoding `acq_rel` for `CAS` enforces strict ordering fences, which may penalize performance if the developer only requires relaxed (monotonic) semantics.

### LLVM Memory Ordering Mapping
The PR introduces a direct mapping function from Triton's `MemSemantic` enum to LLVM's `AtomicOrdering`:

| Triton `MemSemantic` | `LLVM::AtomicOrdering` | Description |
|----------------------|------------------------|-------------|
| `RELAXED`            | `monotonic`            | No synchronization or ordering constraints imposed on other memory operations, only guarantees atomicity of the operation itself. |
| `ACQUIRE`            | `acquire`              | Subsequent memory operations cannot be reordered before the atomic load. |
| `RELEASE`            | `release`              | Prior memory operations cannot be reordered after the atomic store. |
| `ACQUIRE_RELEASE`    | `acq_rel`              | Combines `acquire` and `release`, acting as a bidirectional memory barrier. |

### Implementation in Code Generation
This mapping is dynamically applied during the lowering of Triton's dialect into LLVM IR within `TritonGPUToLLVM/LoadStoreOpToLLVM.cpp`:
1. **RMW Operations (`LLVM::AtomicRMWOp`)**: The instruction creation now consumes `llvmMemOrdering` instead of hardcoding `monotonic`.
2. **CAS Operations (`LLVM::AtomicCmpXchgOp`)**: The success ordering consumes `llvmMemOrdering` instead of hardcoding `acq_rel`. The failure ordering conservatively defaults to `monotonic` per typical LLVM backend recommendations.

## Performance and Reliability Impact
1. **Correctness**: The primary driver for this PR was fixing unit tests involving `atomic_cas` (compare-and-swap). By expanding the unit test scope from 64 to 2000 work groups (`num_ctas=2000`), the synchronization edge-cases that expose memory reordering bugs became statistically inevitable. Explicit memory ordering enforcement resolves these race conditions.
2. **Performance (Flexibility)**: Developers optimizing specialized kernels (e.g., custom reduction kernels, serialized global locks, persistent kernels with global wave sync) can now confidently use relaxed orderings to squeeze out memory latency, or strict orderings to guarantee correctness.
3. **Hardware Semantics**: On AMD CDNA architectures (MI200, MI300 series), atomic visibility across the L2 cache (across CUs) often requires precise memory fences (e.g., `s_waitcnt` with appropriate counters or `buffer_wbinvl1_vol` equivalents lowered by LLVM). Passing the correct LLVM memory ordering ensures the backend emits the optimal sequence of fences surrounding the `buffer_atomic` or `flat_atomic` instructions.

## Key Takeaway
For kernel engineers writing Triton for ROCm, this update restores the standard behavior of semantic atomics. You should explicitly declare `SEM="acq_rel"` (or whichever semantic fits the algorithm) in synchronization structures like custom mutexes and software barriers to ensure correct cross-CU data visibility, particularly when scaling to full GPU occupancy.

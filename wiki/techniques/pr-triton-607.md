---
id: technique-pr-triton-607
title: "Stream-K: Non-Atomic Multiple Buffer Implementation"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, rocm-kernel, triton, memory, synchronization, compute, memory-bound]
kernel_types: [gemm]
languages: [triton-rocm]
confidence: inferred
sources: [pr-triton-607]
---

# Stream-K: Non-Atomic Multiple Buffer Implementation

## 1. Overview

Stream-K is an advanced workload-partitioning strategy for Matrix Multiplication (GEMM) that ensures all Compute Units (CUs) are fully utilized. By distributing the K-dimension iterations evenly across all available hardware resources, Stream-K minimizes tail-effect inefficiencies typically seen in standard tile-based GEMMs.

However, when multiple CUs process non-overlapping segments of the K-dimension for the *same* output tile, their partial accumulations must be summed together. The standard approach relies on `atomic_add` operations in global memory to combine these partial results. PR [#607](https://github.com/ROCm/triton/pull/607) in ROCm/triton replaces this atomic-based accumulation with a **multiple buffer implementation**, removing hardware atomics from the critical path.

## 2. Architectural Intent and Bottleneck Analysis

### The `atomic_add` Bottleneck
In AMD CDNA architectures (CDNA2, CDNA3, CDNA4), issuing concurrent `atomic_add` requests to the same global memory addresses from multiple CUs triggers hardware contention:
1. **Cache Coherency Traffic:** Atomics force frequent cache line invalidations and memory system locks, degrading the efficiency of the L2 cache.
2. **Read-Modify-Write (RMW) Latency:** Each atomic addition effectively becomes a serialized sequence of operations, stalling wave execution and preventing latency hiding.
3. **Non-Determinism:** Floating-point addition is non-associative. When multiple threads use atomic additions, the order of accumulation is non-deterministic, leading to numerical inconsistencies across runs.

### The Multiple Buffer Strategy
By inferring from the PR description ("multiple buffer implementation to replace atomic_add"), the new approach introduces a lock-free paradigm:
1. **Workspace Allocation:** Instead of pointing all participating threads to the final output tile, the kernel dynamically assigns unique, disjoint temporary buffers in global memory (or HBM) for each CU's partial results.
2. **Independent Writes:** Each CU computes its fragment of the K-dimension and writes its partial accumulation directly into its dedicated buffer using high-bandwidth vector stores, avoiding RMW entirely.
3. **Deterministic Reduction:** A secondary reduction phase—either handled by the last arriving CU via a global counter or as a separate fused kernel dispatch—fetches these partial buffers and performs a deterministic, sequential (or tree-based) reduction into the final output matrix.

## 3. Performance and Memory Bounds

### Compute and Memory Bandwidth Optimizations
- **Increased Memory Throughput:** By shifting from `atomic_add` to standard vectorized store operations (`global_store`), the implementation achieves significantly higher memory write bandwidth.
- **Improved ALU Utilization:** Without waiting on atomic serialization stalls, the Matrix Fused Multiply-Add (`mfma`) units can sustain higher throughput and closer-to-peak utilization.

### Memory Overhead Trade-offs
- **Increased Workspace Memory:** This technique trades memory capacity for performance. Storing $N$ partial accumulations requires allocating $O(N \times \text{tile\_size})$ additional global memory. For extremely large matrices partitioned across hundreds of CUs, this could exert pressure on HBM capacity.
- **Secondary Read Overhead:** The final reduction requires reading back the partial buffers from HBM before writing the ultimate result. This introduces a slight latency penalty, but on modern CDNA architectures, the massive memory bandwidth (e.g., 5.3 TB/s on MI300X) easily absorbs this linear read overhead compared to the severe penalties of atomic collisions.

## 4. Applicability and Extensions

This technique is vital for kernels like **GEMM** and **Grouped GEMM**, specifically where dynamic load balancing (like Stream-K) forces overlapping writes. It is highly applicable for large-scale training and inference setups on AMD MI250/MI300 series GPUs (CDNA2/CDNA3) where deterministic results are required for rigorous debugging and numerical stability (e.g., in LLM training).

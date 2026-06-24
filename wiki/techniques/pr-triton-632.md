---
id: technique-pr-triton-632
title: "PR Insight: StreamK Atomics Replacement via Spinlocks and Multiple Buffers"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
kernel_types:
  - gemm
languages:
  - triton-rocm
tags:
  - optimization
  - rocm-kernel
  - synchronization
  - memory
  - memory-bound
confidence: inferred
sources:
  - pr-triton-632
---

# Replacing StreamK Atomics with Spinlocks and Multiple Buffers

## Overview
In the context of the ROCm/triton repository, PR #632 addresses an optimization for the **StreamK GEMM kernel**. The core improvement replaces traditional global memory `atomic_add` operations with a custom synchronization mechanism utilizing a **spinning lock** combined with a **multiple buffer method**.

## Context: The StreamK Accumulation Problem
StreamK is an advanced workload distribution strategy for GEMM operations designed to achieve near-perfect utilization across all available Compute Units (CUs) by allowing workgroups to compute fractional output tiles. Because multiple workgroups (or blocks) might compute partial results for the exact same output tile in the `C` matrix, these partial results must be reduced (summed) together.

Historically, this reduction is performed using global `atomic_add` operations. However, hardware atomic additions can introduce several performance and correctness bottlenecks:
- **Data Type Support:** Native hardware `atomic_add` may lack support or optimal throughput for modern mixed-precision data types (e.g., FP16, BF16, or FP8). Software fallbacks (like atomic Compare-And-Swap loops) for these types can be extremely slow.
- **Contention:** Heavy contention on identical memory addresses from multiple CUs leads to memory request serialization, underutilizing high-bandwidth memory (HBM).
- **Non-Deterministic Ordering:** Floating-point addition is not associative. Using `atomic_add` means the accumulation order is non-deterministic, which can lead to slight numerical variations across different runs.

## Technique: Spinlocks and Multiple Buffers
To bypass the limitations of hardware `atomic_add`, the optimization introduces a software-managed reduction mechanism:

1. **Spinning Lock (Spinlock):** A synchronization primitive (typically implemented via lightweight `atomic_cas` or `atomic_xchg` operations) is used to control access to the shared accumulation workspace.
2. **Multiple Buffer Method:** Instead of directly competing to write to a single destination via atomics, the workgroups leverage multiple discrete buffers or a staged accumulation workspace. Once a workgroup finishes computing its partial tile, it acquires the spinlock, performs a standard vectorized load-add-store sequence from the workspace, and releases the lock.

> [!TIP]
> **Vectorized Memory Access**
> By protecting the critical section with a lock, the actual reduction (read-modify-write) can be performed using wide, vectorized loads and stores rather than scalar atomic operations. This significantly improves memory throughput.

### Advantages
- **Broad Data Type Compatibility:** Because the addition occurs in registers rather than via hardware memory controllers, any data type supported by the ALUs can be accumulated without waiting for native memory atomic support.
- **Reduced Contention Impact:** While the lock serializes the critical section, the time spent in the critical section is minimized by vectorized memory operations, reducing the overall stall time compared to serialized atomic requests.

## Performance and Memory Bounds
- **Memory Bandwidth:** This technique explicitly optimizes HBM usage. By switching to vectorized memory accesses inside a locked critical section, the kernel can achieve higher effective memory bandwidth utilization compared to scalar atomics.
- **Synchronization Overhead:** The primary overhead shifts from memory atomic serialization to lock contention. For small numbers of overlapping partial tiles, the spinlock overhead is negligible.
- **Register Usage:** The lock implementation and vectorized load-add-store sequence may marginally increase VGPR (Vector General-Purpose Register) usage, but this is typically outweighed by the performance gains in accumulation.

## Applicability
This optimization is particularly relevant for CDNA architectures where maximizing HBM bandwidth and managing global memory contention are critical for high-performance GEMM kernels written in Triton.

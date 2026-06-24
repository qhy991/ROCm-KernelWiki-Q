---
id: wiki-technique-pr-triton-682
title: "AMD Performance Cherry Picks (Triton PR 4925)"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, optimization, memory-bound, scheduling]
hardware_features: [mfma, lds]
techniques: [mfma-scheduling, bank-conflict-padding, async-copy]
kernel_types: [gemm, attention]
languages: [triton-rocm]
confidence: inferred
sources: [pr-triton-682]
---

# AMD Performance Cherry Picks in Triton

## Context & Intent
ROCm/triton PR #682, titled `[CP] AMD Performance cherry picks`, merges significant performance optimizations originally developed in upstream Triton (PR #4925). The primary intent of these cherry picks is to bring targeted compiler optimizations to the ROCm backend, enhancing how Triton generates code for AMD CDNA architectures (CDNA2, CDNA3, and CDNA4). 

Given the focus on "Performance cherry picks," this update aims to bridge the gap in hardware utilization between AMD GPUs and their NVIDIA counterparts by fine-tuning low-level code generation tailored to AMD's specific execution model.

## Optimization Techniques
This PR incorporates several critical techniques designed to maximize throughput on AMD matrix cores and memory hierarchy:

1. **MFMA Instruction Scheduling (`mfma-scheduling`)**
   The ROCm Triton backend relies on MFMA (Matrix Fused Multiply-Add) instructions. A key performance optimization involves rescheduling these instructions to aggressively overlap with global and local memory operations. By keeping the MFMA units continuously fed, the compiler can effectively hide memory latency during compute-bound operations like GEMMs and Flash Attention.

2. **LDS Bank Conflict Avoidance (`bank-conflict-padding`)**
   AMD's Local Data Share (LDS) is highly susceptible to bank conflicts, which can drastically reduce memory bandwidth. The cherry picks likely include improved address swizzling (XOR-based transformations) and padding strategies when allocating shared memory. This ensures that vectorized threads accessing the LDS can do so without serializing their requests.

3. **Asynchronous Data Movement (`async-copy`)**
   Optimizing the pipeline between Global Memory and LDS is crucial. Utilizing asynchronous copy operations allows the hardware to prefetch tile data in the background. This bypasses the need to stage data in Vector General Purpose Registers (VGPRs), reducing register pressure and enabling higher occupancy per Compute Unit.

## Memory Bounds & Performance Implications
For memory-bound kernels, the bottleneck lies primarily in HBM (High Bandwidth Memory) utilization and LDS throughput.
- **Bandwidth Saturation:** By refining vectorized loads and ensuring aligned memory accesses, these optimizations allow the kernels to saturate the HBM bus more consistently.
- **Occupancy Tuning:** Reductions in VGPR usage through efficient async copies mean more wavefronts can be active simultaneously. Increased occupancy is vital for hiding the latency of global memory reads in low arithmetic-intensity scenarios.
- **Compute Efficiency:** For compute-bound kernels, the improved MFMA scheduling ensures that the mathematical throughput is limited only by the theoretical peak of the matrix cores, rather than being stalled by instruction dependencies or memory fetches.

## Conclusion
The cherry-picking of upstream PR #4925 into the ROCm Triton backend represents a significant step in maturing the AMD compiler pipeline. By addressing both compute scheduling (MFMA) and memory bounds (LDS/HBM), these optimizations enable users writing Triton kernels to achieve near native-level performance on CDNA hardware without manual assembly or intrinsic tuning.

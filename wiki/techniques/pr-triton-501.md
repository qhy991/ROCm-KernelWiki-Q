---
id: technique-pr-triton-501
title: "Stream Pipelining with Transpose in ROCm Triton"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [swizzling, double-buffering, pipeline, optimization, rocm-kernel, memory-bound]
hardware_features: [lds, mfma]
kernel_types: [gemm, attention, flash-attention]
languages: [triton-rocm]
confidence: source-reported
sources: [pr-triton-501]
---

# Stream Pipelining with Transpose in ROCm Triton

## Overview
Triton PR [#501](https://github.com/ROCm/triton/pull/501) enables support for software pipelining (stream pipelining) on dot operations that involve transpositions—most notably `tt.dot(Q, tt.trans(K))`. Crucially, it resolves a significant performance degradation caused by incorrect LDS (Local Data Share) swizzling layouts generated when an operand was transposed.

## Context: Software Pipelining and Swizzling
To hide global memory latency on CDNA architectures, Triton applies a software pipelining pass that double-buffers (or multi-buffers) memory from global memory into LDS. Computations (like matrix multiplication using `v_mfma` instructions) read these operands from LDS into VGPRs.

To prevent **LDS bank conflicts** when a wavefront accesses shared memory, Triton applies "swizzling" (typically an XOR-based transformation of the address bits). The swizzling function must be carefully tuned to the specific access pattern of the consumer.

In FlashAttention and many transformer-based models, computing $Q \times K^T$ requires loading $K$ and transposing it on the fly. In Triton, this is represented as `tt.dot(Q, tt.trans(K))`.

## The Architectural Bottleneck: Pre-Transpose vs. Post-Transpose Layouts
Prior to this fix, the stream pipeliner evaluated the swizzling pattern based on the **pre-transpose layout** of the tensor. 

1. **The Conflict:** By optimizing the swizzling code for the pre-transpose state, the layout favored the global-to-LDS write pattern but ignored the transpose operation.
2. **MFMA Read Degradation:** When the MFMA instructions later read the data from LDS to VGPRs, they traversed the data in the transposed orientation. Because the swizzle pattern was incorrect for this traversal, multiple threads within the wavefront hit the same LDS banks simultaneously.
3. **Hardware Impact:** On CDNA architectures, an LDS bank conflict forces the memory controller to serialize requests. This stalls the wavefront, starves the matrix cores (MFMA) of operands, and severely degrades achievable TFLOPS, effectively turning a compute-bound GEMM into an LDS-bandwidth-bound operation.

## Resolution
The PR updates the stream pipelining logic to compute the shared memory layout and issue swizzling code based on the **post-transpose layout** whenever `tt.trans` is fed into a `tt.dot`. 

- **Consumer-Aware Swizzling:** By shifting the swizzling strategy to match the post-transpose shape, the LDS layout aligns perfectly with the access pattern of the MFMA read instructions.
- **Conflict-Free Reads:** The `v_mfma` operand loads (from LDS to VGPRs) become bank-conflict-free, operating at peak LDS bandwidth.
- **Pipeline Efficiency:** With LDS read stalls eliminated, the software pipeline can correctly overlap global memory fetches with matrix math without artificial pipeline stalls.

## Takeaways for Kernel Developers
- **On-the-fly Transpose Performance:** Developers utilizing `triton-rocm` can rely on `tt.dot(A, tt.trans(B))` without incurring catastrophic LDS bank conflicts. 
- **Global Memory Layout:** It is no longer necessary to pre-transpose tensors in global memory prior to kernel launch just to avoid swizzling bugs inside the Triton compiler. The stream pipeliner naturally accommodates the layout transformation in shared memory.

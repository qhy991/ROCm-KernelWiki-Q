---
id: technique-pr-triton-497
title: "Variable Length (Varlen) Flash Attention v2 API in Triton"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - flash-attention
  - triton-rocm
  - inference
  - memory-bound
  - optimization
  - rocm-kernel
confidence: inferred
sources:
  - pr-triton-497
---

# Variable Length (Varlen) Flash Attention v2 API in Triton

## Overview

PR #497 in ROCm/triton introduces the Flash Attention v2 "varlen" (variable length) API specifically tailored for forward pass and inference workloads on AMD CDNA architectures. In modern LLM serving scenarios (e.g., continuous batching in vLLM), batches contain sequences of varying lengths. The traditional approach requires padding shorter sequences to match the longest sequence in the batch, wasting substantial computational resources and memory bandwidth. The `varlen` API circumvents this by operating directly on concatenated unpadded sequences, guided by a cumulative sequence length array (`cu_seqlens`).

## Architectural & Performance Analysis

### Memory Bound Optimization

Flash Attention is fundamentally designed to mitigate memory bandwidth bottlenecks by keeping the attention matrix $QK^T$ in Local Data Share (LDS) and performing tiling. However, when dealing with padded batches, standard kernels still load padding tokens from High Bandwidth Memory (HBM), compute their attention scores, and write out meaningless padded outputs.

The `varlen` approach provides a profound optimization for memory-bound scenarios:
1. **Elimination of Padded Loads/Stores:** By passing `cu_seqlens_q` and `cu_seqlens_k` to the Triton kernel, the address generation logic strictly bounds loads to valid tokens. This maximizes HBM effective bandwidth usage.
2. **LDS Capacity Efficiency:** CDNA compute units (CUs) have finite LDS capacity. Bypassing padding tokens means every LDS tile loaded contains mathematically useful data, inherently boosting the arithmetic intensity of the kernel and avoiding unnecessary capacity misses.

### Compute Optimization (MFMA Utilization)

On architectures like CDNA2 (MI250), CDNA3 (MI300), and CDNA4, matrix multiplications are accelerated using MFMA (Matrix Fused Multiply-Add) instructions.
- In padded configurations, MFMA cycles are wasted computing zeros or masked elements.
- With the `varlen` API, the Triton kernel iterates linearly over the exact block sizes specified by `cu_seqlens`. This continuous sequence scheduling means the wave instructions (including register tiling and MFMA instructions) are completely saturated with active computation.

### Indexing and Pointer Arithmetic

The transition to `varlen` requires intricate changes to the Triton pointer arithmetic. Instead of calculating offsets via fixed grid dimensions:
`offset = (batch_idx * seq_len * head_dim) + (seq_idx * head_dim) + head_idx`

The block pointers must dynamically shift using the `cu_seqlens` array:
`token_offset = cu_seqlens[batch_idx] + seq_idx`

This indirection introduces a slight overhead in scalar ALU operations for address calculation, but yields a massive net positive by completely omitting the dense matrix multiplications for padding elements.

## Implementation Scope

As originally merged, this implementation provides:
- **Supported:** Forward pass (fwd) / inference execution.
- **Pending/Unsupported:** Causal masking, bias additions, and dropout. Causal masking with variable lengths requires a dynamic masking bounds check per batch element, which adds further complexity to the inner loop of the Flash Attention kernel and was left for subsequent iterations.

## Impact on LLM Serving

The integration of this API into ROCm's Triton fork is highly strategic for production deployments. Engines like vLLM rely heavily on `varlen` to achieve high-throughput inference by packing variable-length sequences tightly into the same batch. This PR bridges the gap, allowing Triton-compiled ROCm Flash Attention kernels to perform on par with highly optimized custom HIP implementations for continuous batching workloads.

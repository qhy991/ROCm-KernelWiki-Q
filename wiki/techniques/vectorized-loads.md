---
id: technique-vectorized-load
title: Vectorized Global Memory Loads
type: wiki-technique
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [memory, optimization, vectorization]
confidence: source-reported
techniques: [vectorized-load]
hardware_features: [lds]
kernel_types: [gemm, attention, conv]
related: [pattern-memory-bound-amd, hw-lds]
sources: [blog-amdgpu-kernel-opt, doc-mi300x-workload]
reproducibility: snippet
---

# Vectorized Global Memory Loads

Use wide vector loads (128-bit `float4` / `uint4`) to maximize HBM throughput on memory-bound kernels.

## HIP Example

```cpp
// 128-bit coalesced load per thread
float4 vec = *reinterpret_cast<const float4*>(&input[idx * 4]);
```

## Checklist

- Ensure 16-byte alignment of base pointers
- Prefer consecutive thread indices mapping to consecutive addresses
- Combine with LDS staging for reuse-heavy tiles

## Related

- [Memory-Bound AMD Pattern](../patterns/memory-bound-amd.md)

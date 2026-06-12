---
id: technique-bank-conflict-padding
title: LDS Bank Conflict Padding
type: wiki-technique
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [lds, memory, optimization]
confidence: verified
techniques: [bank-conflict-padding]
hardware_features: [lds, wavefront]
kernel_types: [gemm, attention, reduction]
related: [hw-lds, pattern-memory-bound-amd]
sources: [doc-cdna4-isa, blog-amdgpu-kernel-opt]
reproducibility: snippet
---

# LDS Bank Conflict Padding

Add one element of padding per row in LDS arrays so 64-thread wavefront accesses map to distinct banks.

## Problem

With 32 banks and 64-thread wavefronts, strided row access can serialize LDS ports when multiple lanes hit the same bank.

## Fix

```cpp
// BAD: power-of-two width often causes conflicts with 64 lanes
__shared__ float tile[64][64];

// GOOD: pad inner dimension by 1
__shared__ float tile[64][65];
float v = tile[row][col];  // same indexing, better bank distribution
```

## Related

- [LDS Hardware Page](../hardware/lds.md)
- context-hub: `opt-bank-conflict-avoidance`

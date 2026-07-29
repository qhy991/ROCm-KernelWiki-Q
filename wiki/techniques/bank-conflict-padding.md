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
sources: [doc-cdna4-isa, blog-amdgpu-kernel-opt, pr-rocm_libraries-9260, pr-rocm_libraries-9233]
reproducibility: snippet
---

# LDS Bank Conflict Padding

Change an LDS stride at the granularity of the conflicting access pattern so
wavefront lanes distribute across banks without breaking vector or
asynchronous transfers.

## Problem

With 32 banks and 64-thread wavefronts, strided row access can serialize LDS ports when multiple lanes hit the same bank.

## Basic Row-Padding Fix

```cpp
// BAD: power-of-two width often causes conflicts with 64 lanes
__shared__ float tile[64][64];

// GOOD: pad inner dimension by 1
__shared__ float tile[64][65];
float v = tile[row][col];  // same indexing, better bank distribution
```

Row padding is a starting pattern, not a universal rule. Verify the generated
address pattern and the transfer primitive.

## Preserve Asynchronous DMA Slabs

`pr-rocm_libraries-9260` covers a gfx950 D256 attention layout that moves
two-row K slabs with asynchronous DMA. Padding each row would make the source
transfer discontinuous. The implementation instead pads between slabs and
uses the same slab stride in the QK read mapping.
The change first merged to a feature branch; `pr-rocm_libraries-9233` is the
later `develop` landing that explicitly ships the padded route.

For the source's MI355X S8192 shape:

- bank-conflict events per 1000 decrease from 897 to 660;
- reported bank-conflict cycles decrease by roughly 4.5x;
- latency decreases from 1592 to 1195 microseconds;
- LDS remains around 65 KB, with two workgroups per CU;
- VGPR use is unchanged.

This yields a more general procedure:

1. Identify the exact lane-to-address pattern that conflicts.
2. Preserve the contiguous unit required by DMA or vector loads.
3. Insert padding between those units.
4. Update producer and consumer strides together.
5. Re-check LDS capacity, occupancy, conflict counters, and numerical output.

## Related

- [LDS Hardware Page](../hardware/lds.md)
- context-hub: `opt-bank-conflict-avoidance`

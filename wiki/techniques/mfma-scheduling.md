---
id: technique-mfma-scheduling
title: MFMA Instruction Scheduling
type: wiki-technique
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [mfma, optimization, compute]
confidence: source-reported
techniques: [mfma-scheduling]
hardware_features: [mfma, dual-cma]
kernel_types: [gemm, attention]
related: [hw-mfma-matrix-core, technique-ck-tile-programming]
sources: [blog-matrix-cores-cdna, doc-cdna4-isa]
reproducibility: snippet
---

# MFMA Scheduling

Strategies to keep matrix cores busy while hiding memory latency on CDNA GPUs.

## Core Principles

1. **Interleave MFMA with global/LDS loads** — MFMA has multi-cycle latency; issue memory ops between MFMA batches
2. **Match tile size to register budget** — larger tiles raise FLOPs/inst but reduce occupancy
3. **Use dual CMA on MI300X+** — two matrix pipes per CU when instruction mix allows

## Manual HIP + ASM Pattern

```cpp
// Prefetch tile A0/B0 while computing on C accumulators from previous k-tile
load_a_to_lds(k + 1);
load_b_to_lds(k + 1);
#pragma unroll
for (int i = 0; i < MFMA_PER_TILE; ++i) {
    asm volatile("v_mfma_f32_16x16x16f16 ..." :::);
}
__syncthreads();
```

## When to Use CK Instead

For production GEMM/attention, CK Tile pipelines already encode MFMA scheduling, double buffering, and epilogue fusion. Manual scheduling is best for small fused kernels or research prototypes.

## Related

- [MFMA Matrix Core](../hardware/mfma-matrix-core.md)
- [CK Tile Programming](ck-tile-programming.md)

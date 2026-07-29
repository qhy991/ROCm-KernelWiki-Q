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
sources: [blog-matrix-cores-cdna, doc-cdna4-isa, pr-rocm_libraries-9403]
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

## Interleave Scheduling With Launch Geometry

`pr-rocm_libraries-9403` is a concrete gfx950 attention example where
instruction order and launch geometry are coupled. The targeted D128 prefill
path interleaves online-softmax operations with MFMA and changes dispatch from
two to four waves.

The source reports:

- VGPR use decreasing from 296 to 250;
- AGPR use decreasing from 40 to 0;
- resident waves increasing from four to eight per CU;
- occupancy increasing from 11.8% to 23.5%;
- S8192 throughput increasing from 333.9 to 440.7 TFLOPS.

The reusable lesson is not "always use four waves." Moving independent scalar
or vector work into MFMA latency slots can alter the register live range, which
then changes the viable wave count. Re-evaluate both together for each shape
selector.

## Related

- [MFMA Matrix Core](../hardware/mfma-matrix-core.md)
- [CK Tile Programming](ck-tile-programming.md)

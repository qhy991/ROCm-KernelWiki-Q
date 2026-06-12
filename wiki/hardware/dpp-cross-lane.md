---
id: hw-dpp-cross-lane
title: DPP — Data-Parallel Primitives (Cross-Lane Operations)
type: wiki-hardware
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [dpp, cross-lane, compute, hardware]
confidence: verified
hardware_features: [dpp, bpermute]
related: [hw-wavefront, hw-mfma-matrix-core]
sources: [doc-cdna4-isa, blog-gcn-cross-lane]
cuda_equivalent: warp_shuffle
---

# DPP — Data-Parallel Primitives

Cross-lane data movement within a wavefront. AMD's equivalent to CUDA warp shuffle operations.

## Overview

DPP instructions allow threads (lanes) within a wavefront to exchange data without using LDS. They use the LDS routing hardware but **do not access LDS memory**.

### Key Instructions

| Instruction | Description | CUDA Equivalent |
|-------------|-------------|-----------------|
| `v_mov_dpp` | Move with permutation pattern | `__shfl_sync()` |
| `ds_bpermute_b32` | Arbitrary lane-to-lane via bypass | `__shfl_sync()` (any source) |
| `v_permlane16_b32` | Static 16-lane permutation | Limited equivalent |
| `v_permlanex16_b32` | Static cross-half permutation | Limited equivalent |

## Permutation Modes for v_mov_dpp

```
Wavefront layout (64 lanes):
Row 0: [L0  L1  L2  L3] [L4  L5  L6  L7] [L8  L9  L10 L11] [L12 L13 L14 L15]
Row 1: [L16 L17 L18 L19] [L20 L21 L22 L23] [L24 L25 L26 L27] [L28 L29 L30 L31]
Row 2: [L32 L33 L34 L35] [L36 L37 L38 L39] [L40 L41 L42 L43] [L44 L45 L46 L47]
Row 3: [L48 L49 L50 L51] [L52 L53 L54 L55] [L56 L57 L58 L59] [L60 L61 L62 L63]
```

### Mode Reference

| Mode | Pattern | Use Case |
|------|---------|----------|
| `quad_perm:[0,1,2,3]` | Identity within quad | — |
| `quad_perm:[1,2,3,0]` | Rotate within quad | Quad-level reduction |
| `row_shl:1` | Shift left by 1 within row | Prefix sum, scan |
| `row_shr:1` | Shift right by 1 within row | Reverse scan |
| `row_ror:8` | Rotate right by 8 within row | Cross-quad exchange |
| `wave_ror:16` | Rotate across entire wavefront | Cross-row exchange |
| `row_mirror` | Mirror within row | Butterfly patterns |

## Common Patterns

### Warp-Level Reduction

```c
// Sum all lanes in wavefront
float sum = value;
sum += __shfl_xor(sum, 32);  // Exchange with lane 32 away
sum += __shfl_xor(sum, 16);  // Exchange with lane 16 away
sum += __shfl_xor(sum, 8);   // Exchange with lane 8 away
sum += __shfl_xor(sum, 4);   // Exchange with lane 4 away
sum += __shfl_xor(sum, 2);   // Exchange with lane 2 away
sum += __shfl_xor(sum, 1);   // Exchange with lane 1 away
// Now every lane has the total sum
```

> ⚠️ **Important**: AMD wavefront has 64 lanes vs CUDA warp of 32.
> Reductions need one extra step on AMD (the first `xor 32` step).

### Prefix Sum (Scan)

```c
float prefix_sum(float val) {
    // Step 1: Shift and add
    float tmp = __shfl_up(val, 1);  // → v_mov_dpp row_shr:1
    if (threadIdx.x % 2 >= 1) val += tmp;

    tmp = __shfl_up(val, 2);
    if (threadIdx.x % 4 >= 2) val += tmp;

    tmp = __shfl_up(val, 4);
    if (threadIdx.x % 8 >= 4) val += tmp;

    tmp = __shfl_up(val, 8);
    if (threadIdx.x % 16 >= 8) val += tmp;

    tmp = __shfl_up(val, 16);
    if (threadIdx.x % 32 >= 16) val += tmp;

    tmp = __shfl_up(val, 32);
    if (threadIdx.x % 64 >= 32) val += tmp;

    return val;
}
```

### GEMM Tile Transpose

```c
// Transpose a 4x4 tile using DPP (no LDS needed)
// Each lane holds one element of the 4x4 tile
float val = tile[row * 4 + col];

// DPP quad perm for 4x4 transpose
asm volatile("v_mov_dpp %0, %1 quad_perm:[0,4,8,12] row_mask:0xf bank_mask:0xf"
    : "=v"(val) : "v"(val));
```

## Performance

| Property | Value |
|----------|-------|
| Latency | ~4 cycles (same as VALU) |
| Throughput | 1 per cycle per SIMD |
| LDS traffic | None (uses routing hardware only) |
| Bank conflicts | Not applicable |

## Key Difference from CUDA Shuffles

1. **Wavefront size**: 64 lanes (CUDA: 32) — need extra reduction steps
2. **DPP is richer**: More permutation patterns than CUDA's index-based shuffle
3. **bpermute**: Uses LDS bypass hardware — can address any lane, but slightly higher latency
4. **Masking**: `row_mask` and `bank_mask` provide fine-grained lane control

## References

- [AMD GCN Cross-Lane Operations](https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/)
- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [SCALE: CUDA Shuffles → AMD DPP](https://scale-lang.com/posts/2026-01-19-optimizing-cuda-shuffles)

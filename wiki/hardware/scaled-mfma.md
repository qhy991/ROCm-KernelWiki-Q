---
id: hw-scaled-mfma
title: Scaled MFMA (CDNA4 Block-Scaled Matrix Operations)
type: wiki-hardware
architectures: [cdna4]
tags: [scaled-mfma, mfma, block-scale, fp8, compute]
confidence: source-reported
hardware_features: [mfma, scaled-mfma, block-scale]
related: [hw-mfma-matrix-core]
sources: [doc-cdna4-isa, doc-cdna4-whitepaper, blog-matrix-cores-cdna, pr-composable_kernel-3540]
cuda_equivalent: nvfp4_block_scale
---

# Scaled MFMA (CDNA4 Block-Scaled Matrix Operations)

CDNA4 introduces block-scaled MFMA instructions (`v_mfma_scale_*`, e.g. `v_mfma_scale_f32_16x16x128_f8f6f4`) — matrix operations that apply per-block scaling factors during matrix multiply-accumulate, enabling efficient low-precision (FP8/BF8/FP6/FP4) GEMM with higher effective dynamic range.

> **Naming note:** these block-scaled instructions are **not** the `v_smfmac_*` family. `v_smfmac_*` ("Sparse MFMA with Compression") performs *structured-sparsity* matmul (one operand 2:4-sparse) and is unrelated to block scaling. The CDNA4 ISA reference ([`doc-cdna4-isa`](../../sources/docs/cdna4-isa.md)) names the scaled forms `v_mfma_scale_*`, matching the sibling [MFMA page](mfma-matrix-core.md).

## Overview

| Feature | CDNA3 | CDNA4 |
|---------|-------|-------|
| Standard MFMA | ✓ | ✓ |
| Scaled MFMA | ✗ | ✓ |
| Block scaling | ✗ | Per 16×16 block |
| FP8/BF8 support | Standard only | Standard + Scaled |
| F6/F4 support | ✗ | ✓ (via Scaled MFMA) |

## Key Instructions

The block-scaled forms use a unified `F8F6F4` operand encoding: the element format (FP8 E4M3, BF8 E5M2, FP6, FP4) is selected per-operand by a format field rather than by a separate opcode per dtype pair. There are two tile shapes:

| Instruction | Input Format | Accumulator | Scale Block | MFMA Tile |
|-------------|--------------|-------------|-------------|-----------|
| `v_mfma_scale_f32_16x16x128_f8f6f4` | FP8 / BF8 / FP6 / FP4 (per-operand format select) | F32 | per 32-element block | 16×16×128 |
| `v_mfma_scale_f32_32x32x64_f8f6f4` | FP8 / BF8 / FP6 / FP4 (per-operand format select) | F32 | per 32-element block | 32×32×64 |

> Standard (non-scaled) FP8/BF8/INT8/BF16 matmul continues to use the regular `v_mfma_*` opcodes documented on the [MFMA Matrix Core](mfma-matrix-core.md) page; block scaling applies only to the `f8f6f4` low-precision path above. Confirm exact operand/scale encodings against the CDNA4 ISA PDF before relying on a specific spelling.

## Block Scaling Concept

```
Standard MFMA:   D += A × B                    (single scale for all)
Scaled MFMA:     D += ScaleA ⊙ A × ScaleB ⊙ B  (per-block scale factors)

Where:
  ScaleA: [num_blocks_M, 1]     — one scale per 16 rows of A
  ScaleB: [1, num_blocks_N]     — one scale per 16 cols of B
  ⊙:      element-wise multiply applied before MFMA
```

This is similar to NVIDIA's Block-Scaled FP4 (NVFP4) on Blackwell, but operates at the hardware instruction level.

## Data Layout

```
Input A (FP8/BF8): [M, K] packed as 128 elements per MFMA
Input B (FP8/BF8): [K, N] packed as 128 elements per MFMA
Scale A: [M/16, K/128] — one scale factor per 16×128 block
Scale B: [K/128, N/16] — one scale factor per 128×16 block
Accumulator C: [M/16, N/16] → 16 VGPRs per wavefront
```

## HIP Programming

```c
// Using Scaled MFMA via inline assembly
__global__ void scaled_gemm_kernel(
    const __half* A, const __half* B, float* C,
    const float* scale_a, const float* scale_b,
    int M, int N, int K) {

    // Load scale factors for this block's tiles
    float sa = scale_a[blockIdx.y];  // per 16-row block
    float sb = scale_b[blockIdx.x];  // per 16-col block

    // Load A and B tiles, apply scaling
    // Issue v_mfma_scale_f32_16x16x128_f8f6f4
    // The hardware applies per-block scale factors internally
}
```

## Performance Benefits

> The figures below are **qualitative/illustrative**, not measured. No source page in this knowledge base contains specific TFLOPS numbers for these instructions; treat any absolute throughput claim as unverified until tied to a benchmark with a `source_id`.

| Metric | Standard FP8 MFMA | Block-Scaled MFMA | Direction |
|--------|------------------|-------------------|-----------|
| Effective dynamic range | Limited to the FP8/FP4 element format | Extended by per-block scale factors | Higher (depends on block size) |
| Accuracy at low precision | Degrades vs FP16 for outlier-heavy tensors | Closer to FP16 via per-block scaling | Better |
| Compute throughput | — | Same FLOPs as the equivalent non-scaled opcode | ≈ Equal |
| Memory traffic | 1× (packed elements) | 1× + small scale-factor overhead | Minimal overhead |

## Use Cases

1. **LLM Inference**: FP8/BF8 weight + activation GEMM with block scaling for MoE layers
2. **Training**: FP8 gradient accumulation with block-scale for mixed precision
3. **Quantized models**: FP4/F6 weight-only quantization with scaled MFMA for dequantize-GEMM fusion

## References

- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/)
- [CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/)

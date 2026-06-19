---
id: kernel-moe-grouped-gemm-cdna4
title: MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [moe, grouped-gemm, gemm, quantization, fp4, fp8, block-scale, cdna4]
confidence: source-reported
kernel_types: [moe, grouped-gemm, gemm]
languages: [hip-cpp, ck-dsl, triton-rocm]
techniques: [ck-tile-programming, mfma-scheduling, persistent-kernel]
hardware_features: [mfma, scaled-mfma, block-scale, lds]
related: [hw-scaled-mfma, hw-mfma-matrix-core, technique-persistent-kernel, technique-ck-tile-programming]
sources:
  - pr-hipblaslt-330
  - pr-hipblaslt-424
  - pr-hipblaslt-431
  - pr-hipblaslt-433
  - pr-hipblaslt-353
  - pr-hipblaslt-970
  - pr-composable_kernel-3603
  - pr-composable_kernel-3620
  - pr-composable_kernel-3540
  - pr-composable_kernel-3735
  - pr-aiter-3470
  - pr-transformerengine-578
  - pr-composable_kernel-3619
  - pr-vllm-46063
  - pr-vllm-46117
  - pr-vllm-46123
  - pr-sglang-28658
  - pr-sglang-28676
  - pr-sglang-28712
  - doc-cdna4-isa
reproducibility: concept
---

# MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)

Mixture-of-Experts (MoE) models route each input token to a subset of "expert" weight matrices, producing a batch of small GEMMs with varying M dimensions. On CDNA4, block-scaled MFMA instructions (`v_mfma_scale_*`) enable native FP4/FP8 quantized MoE GEMM with per-block scale factors, avoiding dequantization round-trips and achieving throughput close to dense FP8 GEMM.

## The MoE GEMM Problem

```
Standard dense GEMM:  Y = X @ W              one large matmul
MoE GEMM:            Y[i] = X[i] @ W[e[i]]   many small, variable-M GEMMs

Challenges:
  - Each expert gets a different number of tokens → load imbalance
  - M per expert is small (1–64 tokens) → underutilization
  - Total experts is large (64–256) → kernel launch overhead
  - Weights are quantized (FP4/FP8) → need block-scale for accuracy
```

## Architecture: Multi-Stage MoE GEMM

The CDNA4 MoE implementations across CK, hipBLASLt, and AITER share a common multi-stage pattern:

```
Stage 0: Token Routing
  Router assigns each token to top-K experts → expert_ids[token], token_counts[expert]

Stage 1: Grouped GEMM (per-expert)
  For each expert e:
    Y_e = X[expert_tokens[e]] @ W_e    ← the actual GEMM, often block-scaled FP4/FP8

Stage 2: Scatter-Reduce
  For each token, combine outputs from its K chosen experts
  (weighted by router gate scores)
```

The implementations differ in how they handle the grouped GEMM stage and the data format.

## Activation and Epilogue Fusion

Production MoE kernels often need more than grouped matmul. Transformer MLP experts commonly fuse bias, activation, scaling, and scatter/reduce behavior around the GEMM core.

| Fusion point | Design concern | Evidence |
|--------------|----------------|----------|
| SwiGLU / SwiGLU-step | Activation parameters such as clamp and alpha must match the model contract | `pr-composable_kernel-3735` |
| Split-K epilogue | `k_batch == 1` should store final output, not accumulate stale data | `pr-composable_kernel-3735` |
| AITER MXFP4 backend | MoE sort, quant, GEMM, epilogue, scatter, and reduce are integrated for gfx950 | `pr-aiter-3470` |

When indexing MoE work, separate the **GEMM substrate** from the **expert epilogue contract**. A low-precision GEMM can be correct in isolation but still fail a model if the activation, bias, routing, or split-K accumulation semantics differ.

## Implementation Approaches

### 1. Persistent-Kernel Grouped GEMM (CK Tile)

CK's block-scale GEMM pipeline ([`pr-composable_kernel-3540`](../../sources/prs/composable_kernel/PR-3540.md)) uses a persistent kernel pattern where a single kernel launch processes all expert GEMMs via a global work queue:

```c
// Simplified: CK Tile persistent MoE GEMM
// Each block claims an expert tile atomically
while (true) {
    int tile_id = atomicAdd(&work_queue.head, 1);
    if (tile_id >= total_tiles) break;
    int expert = expert_for_tile(tile_id);
    // Load block-scaled A (activations) and B (weights) tiles
    // Issue v_mfma_scale_f32_16x16x128_f8f6f4
    // Write C tile to output
}
grid.sync();  // GWS barrier for scatter-reduce stage
```

The ABQuant pipeline ([`pr-composable_kernel-3620`](../../sources/prs/composable_kernel/PR-3620.md)) optimizes the dual-quantized (both A and B block-scaled) path with sweep-tile improvements.

### 2. Two-Stage MoE GEMM (hipBLASLt)

hipBLASLt splits the MoE GEMM into two kernel stages for BF16×FP4 weight-only quantization:

- **Stage 1** ([`pr-hipblaslt-424`](../../sources/prs/hipblaslt/PR-424.md)): BF16 activations × FP4 weights — dequantize weights via block scales, compute partial GEMM
- **Stage 2** ([`pr-hipblaslt-431`](../../sources/prs/hipblaslt/PR-431.md)): A16W4 format — accumulate into BF16 output with expert routing

The Turbo MXFP8 path ([`pr-hipblaslt-330`](../../sources/prs/hipblaslt/PR-330.md)) targets gfx950 specifically, using native block-scaled MFMA for both MXFP8 activation and weight quantization.

### 3. Native MXFP4 MoE Backend (AITER)

The most CDNA4-specialized implementation is AITER's MXFP4 backend ([`pr-aiter-3470`](../../sources/prs/aiter/PR-3470.md)), which avoids FP8 intermediates entirely:

- **3-stage sorting**: `moe_3stage_sort.cuh` — efficient token→expert routing
- **Native FP4×FP4 MFMA**: `mfma_f4f4.hpp` — direct FP4 block-scaled matmul
- **Scatter-reduce**: `moe_scatter_reduce.cuh` — combine expert outputs
- **XCD remapping**: `xcd_remap.hpp` — optimize data layout across chiplets

### 4. Triton Grouped GEMM

For workloads that prefer Triton over HIP/CK:

- **Work-stealing grouped GEMM** ([`pr-hipblaslt-353`](../../sources/prs/hipblaslt/PR-353.md)): Dynamic load balancing across expert GEMMs via a `ws_mode` API
- **Grouped Triton GEMM for TTFT** ([`pr-hipblaslt-970`](../../sources/prs/hipblaslt/PR-970.md)): Optimizes Time-To-First-Token for MoE inference
- **DeepGEMM-compatible blockscale** ([`pr-hipblaslt-433`](../../sources/prs/hipblaslt/PR-433.md)): Unified grouped+batched GEMM kernel matching DeepGEMM's blockscale API

### 5. FP4 (A4W4) Block-Scale GEMM (CK Tile)

CK Tile's a4w4 path ([`pr-composable_kernel-3603`](../../sources/prs/composable_kernel/PR-3603.md)) supports FP4 packing for both A and B tensors:

- **A**: 1D block scale (per-32-element column)
- **B**: 2D block scale (per-block)
- **Regular pipeline**: Stores FP4 as FP8 in LDS (wider load, dequantize in VGPR)
- **WP (Preshuffle-B) pipeline**: Stores A as raw FP4 in LDS, dequantize on register load — saves LDS bandwidth when B is preshuffled

## Data Format Summary

| Format | Activation | Weight | Scale | Target Hardware | Key PR |
|--------|-----------|--------|-------|-----------------|--------|
| BF16×FP4 (stage1) | BF16 | FP4 (MXFP4) | per-block | CDNA2–4 | [`pr-hipblaslt-424`](../../sources/prs/hipblaslt/PR-424.md) |
| A16W4 (stage2) | BF16 | MXFP4 | per-block | CDNA2–4 | [`pr-hipblaslt-431`](../../sources/prs/hipblaslt/PR-431.md) |
| MXFP8 (Turbo) | MXFP8 | MXFP8 | per-block | CDNA4 (gfx950) | [`pr-hipblaslt-330`](../../sources/prs/hipblaslt/PR-330.md) |
| A4W4 (native FP4) | FP4 | FP4 | per-block | CDNA2–4 | [`pr-composable_kernel-3603`](../../sources/prs/composable_kernel/PR-3603.md) |
| MXFP4 (AITER) | FP4 | FP4 | per-block | CDNA4 (gfx950) | [`pr-aiter-3470`](../../sources/prs/aiter/PR-3470.md) |
| MXFP4 + biased SwiGLU | FP4 | FP4 | per-block | CDNA4 candidate path | [`pr-composable_kernel-3735`](../../sources/prs/composable_kernel/PR-3735.md) |
| MXFP8 grouped GEMM (TE) | MXFP8 | MXFP8 | backend-specific | gfx1250 integration evidence | [`pr-transformerengine-578`](../../sources/prs/transformerengine/PR-578.md) |
| MXFP8 grouped GEMM (TE CK) | MXFP8 | MXFP8 | backend-specific | gfx1250 integration evidence, not CDNA4 claim | [`pr-transformerengine-613`](../../sources/prs/transformerengine/PR-613.md) |
| RDNA4 grouped GEMM (CK) | BF16/INT8 or FP16 examples | fixed-NK multi-ABD | ordinary WMMA path | RDNA4 contrast case | [`pr-composable_kernel-3619`](../../sources/prs/composable_kernel/PR-3619.md) |

## Mapping to Hardware Features

| Feature | Role in MoE GEMM |
|---------|-----------------|
| [Scaled MFMA](../hardware/scaled-mfma.md) | Native block-scaled FP4/FP8 matmul — the core compute primitive |
| [MFMA](../hardware/mfma-matrix-core.md) | Standard FP8/BF16 MFMA for non-block-scaled paths |
| [LDS](../hardware/lds.md) | Tile staging for A/B blocks; bank-conflict padding critical at FP4 density |
| [GWS](../hardware/gws.md) | Grid-wide barrier for persistent-kernel MoE (scatter-reduce sync) |

## When to Use Which Approach

| Scenario | Recommended approach |
|----------|---------------------|
| MoE inference on MI350X, FP4 weights | AITER MXFP4 backend (native FP4×FP4) |
| MoE inference, FP8 weights, MI300X | hipBLASLt Turbo MXFP8 or CK ABQuant |
| MoE training (BF16 activations) | hipBLASLt 2-stage BF16×FP4 |
| Research / custom kernel | CK Tile ABQuant with preshuffle pipeline |
| Triton-based framework (vLLM, SGLang) | Triton grouped GEMM with work-stealing |
| Very large expert count (>128) | Persistent kernel pattern for launch overhead |

## Serving Framework Evidence

Recent open vLLM and SGLang PRs show that production MoE serving often tunes the dispatch boundary as much as the GEMM itself:

| Evidence | What it contributes |
|----------|---------------------|
| `pr-vllm-46063` | Open vLLM AITER small-M dispatch proposal for MiniMax-M3 MXFP8 dense and grouped-MoE decode on gfx950 |
| `pr-vllm-46117` | Open vLLM native MXFP8 decode tile/grouped-MoE tuning for MiniMax-M3 on CDNA4 |
| `pr-vllm-46123` | Open vLLM opt-in FlyDSL BF16 two-stage MoE route for MiniMax-M3 MXFP8 emulation on gfx942 |
| `pr-sglang-28658` | Open SGLang fusion of shared-expert sigmoid/cast into the MoE append kernel to reduce decode launches |
| `pr-sglang-28676` | Open SGLang correctness fix for MXFP8 routed-MoE layout caches across RL weight reloads |
| `pr-sglang-28712` | Open SGLang MiniMax-M3 foundation PR with gfx95 MXFP8 MoE configs and top-k/JIT routing kernels |

## References

- [CDNA4 ISA Reference](../../sources/docs/cdna4-isa.md) — `v_mfma_scale_*` instruction encodings
- [Scaled MFMA (CDNA4)](../hardware/scaled-mfma.md) — block-scaled MFMA instruction details
- [Persistent Kernel Pattern](../techniques/persistent-kernel.md) — work-queue GWS pattern used by CK MoE

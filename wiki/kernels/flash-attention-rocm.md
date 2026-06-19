---
id: kernel-flash-attention-rocm
title: Flash Attention on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [attention, flash-attention, mfma, lds, dpp]
confidence: source-reported
kernel_types: [attention, flash-attention]
languages: [hip-cpp, ck-dsl, triton-rocm, python]
techniques: [ck-tile-programming, mfma-scheduling, double-buffering, occupancy-tuning]
hardware_features: [mfma, lds, dpp, wavefront]
related:
  - hw-mfma-matrix-core
  - hw-lds
  - technique-mfma-scheduling
  - technique-occupancy-tuning
  - kernel-fp8-flash-attention-rocm
  - kernel-paged-prefill-attention-rocm
  - kernel-triton-flash-attention-rocm
sources:
  - blog-matrix-cores-cdna
  - doc-flash-attention-rocm
  - blog-flash-attention-rocm
  - pr-flash-attention-157
  - pr-flash-attention-179
  - pr-flash-attention-181
  - pr-flash-attention-182
  - pr-flash-attention-183
  - pr-flash-attention-178
  - pr-flash-attention-184
performance_claims:
  - gpu: MI300X
    dtype: fp16
    shape: "seqlen=8192, headdim=128"
    metric: TFLOPS
    value: 520
    utilization: "~79%"
    source_id: blog-flash-attention-rocm
  - gpu: MI250X
    dtype: fp16
    shape: "seqlen=4096, headdim=128"
    metric: TFLOPS
    value: 260
    utilization: "~68%"
    source_id: blog-flash-attention-rocm
reproducibility: snippet
artifact_dir: artifacts/kernels/flash-attention-rocm
---

# Flash Attention on ROCm

Optimized attention kernel implementations for AMD CDNA GPUs, covering multiple backends and approaches.

## Available Implementations

ROCm supports **7 different attention backends** (as of early 2026):

| Backend | Library | Architecture | Notes |
|---------|---------|-------------|-------|
| CK SDPA | Composable Kernel | CDNA2+ | AMD's primary implementation |
| FlashAttention-ROCm | ROCm/flash-attention | CDNA2+ | Direct port of FlashAttention |
| FlashAttention-Triton | Triton-ROCm | CDNA3+ | Triton-based implementation |
| IREE | SHARK/IREE | CDNA3+ | MLIR-compiled |
| PyTorch SDPA | PyTorch ROCm | CDNA2+ | Fallback path |
| xFormers | xFormers ROCm | CDNA2+ | Memory-efficient attention |
| Custom HIP | Manual | Any | Maximum control |

## Architecture: CK SDPA (Recommended)

The Composable Kernel (CK) implementation uses a tiled approach:

```
Tile Q: Mr × d   (kept in VGPR)
Tile K: d × Kr    (streamed from HBM through LDS)
Tile V: Kr × d    (streamed from HBM through LDS)
Output: Mr × d    (in VGPR, written to HBM)
```

### Key Tiling Parameters

| Parameter | MI250X (CDNA2) | MI300X (CDNA3) |
|-----------|----------------|----------------|
| Mr (Q tile rows) | 64 | 128 |
| Kr (K/V tile cols) | 32 | 64 |
| MFMA tile | 16×16×16 (f16) | 16×16×16 (f16) |
| LDS tile size | 16 KB | 32 KB |

### Kernel Structure

```c
// Simplified CK SDPA structure
__global__ void flash_attention_kernel(
    const __half* Q, const __half* K, const __half* V,
    __half* O, int seq_len, int head_dim) {

    // 1. Load Q tile into VGPR (stays for entire row)
    // 2. Loop over K/V tiles:
    for (int kv_tile = 0; kv_tile < seq_len / Kr; kv_tile++) {
        // 2a. Load K tile into LDS
        // 2b. Compute S_tile = Q_tile @ K_tile^T using MFMA
        // 2c. Apply softmax (row-wise, using DPP reduction)
        // 2d. Load V tile into LDS
        // 2e. Accumulate O_tile += softmax(S) @ V_tile using MFMA
    }
    // 3. Write O tile to HBM
}
```

### MFMA Usage Pattern

```c
// Q @ K^T → attention scores
// Q_tile: Mr × d  (in VGPR)
// K_tile: d × Kr  (in LDS → VGPR)
// S_tile: Mr × Kr  (in VGPR)

// Using 16×16×16 FP16 MFMA:
for (int m = 0; m < Mr; m += 16) {
    for (int n = 0; n < Kr; n += 16) {
        for (int k = 0; k < d; k += 16) {
            // Load 16×16 K tile from LDS to VGPR
            // Issue MFMA: S[m:m+16, n:n+16] += Q[m:m+16, k:k+16] * K[k:k+16, n:n+16]
        }
    }
}
```

### Softmax with DPP

```c
// Row-wise max and sum using DPP reduction
float max_val = s_tile[thread_lane];
max_val = fmax(max_val, __shfl_xor(max_val, 32));
max_val = fmax(max_val, __shfl_xor(max_val, 16));
max_val = fmax(max_val, __shfl_xor(max_val, 8));
max_val = fmax(max_val, __shfl_xor(max_val, 4));
max_val = fmax(max_val, __shfl_xor(max_val, 2));
max_val = fmax(max_val, __shfl_xor(max_val, 1));
// max_val now holds row maximum (same across all lanes in the row)
```

## Performance Numbers

### MI300X (CDNA3) — FP16 Attention

| Sequence Length | Head Dim | TFLOPS | Utilization |
|----------------|----------|--------|-------------|
| 1024 | 128 | ~340 | ~52% |
| 2048 | 128 | ~440 | ~67% |
| 4096 | 128 | ~490 | ~75% |
| 8192 | 128 | ~520 | ~79% |

### MI250X (CDNA2) — FP16 Attention

| Sequence Length | Head Dim | TFLOPS | Utilization |
|----------------|----------|--------|-------------|
| 1024 | 128 | ~180 | ~47% |
| 2048 | 128 | ~230 | ~60% |
| 4096 | 128 | ~260 | ~68% |

## Tuning Tips

1. **Tile size**: Larger Mr improves arithmetic intensity but increases register pressure
2. **Double buffering**: Load next K/V tile while computing current attention scores
3. **KV cache layout**: Use paged KV cache (vLLM) for variable-length sequences
4. **FP8 on CDNA3+**: Use `v_mfma_f32_*_fp8_*` for KV cache in FP8

## CK Tile Backward Workspace Evolution

The ROCm FlashAttention backward path has a second performance surface outside the MFMA loop: workspace ownership and host/device synchronization. Recent CK Tile PRs show a staged migration:

| Evidence | Lesson |
|----------|--------|
| `pr-flash-attention-181` | Small CK submodule/API changes can require host-wrapper argument updates such as `sink_ptr` and `d_sink_ptr`. |
| `pr-flash-attention-182` | The backward wrapper moved to CK's unified workspace API, replacing explicit `dq_acc` tensor plumbing with workspace-size, prepare, and launcher calls. |
| `pr-flash-attention-183` | Workspace preparation became stream-async to avoid a host-blocking `cu_seqlens.cpu()` style synchronization in group-mode backward. |

For profiling, this means a slow FMHA backward step may not be caused by MFMA occupancy. Check whether the wrapper allocates, copies, or prepares sequence metadata on the host path before tuning tile sizes.

## Edge Cases and Architecture Guards

`seq_q=0` backward is a representative correctness edge case. `pr-flash-attention-179` fixed a NaN path by updating CK and adding a dedicated test, with the underlying issue connected to early exit and register allocation behavior in CK-side fixes.

Architecture gating is another recurring theme. `pr-flash-attention-178` enabled RDNA build support with guards for unstable backward paths, while `pr-flash-attention-184` is an open PR that removes local gfx11/gfx12 backward guards after updating CK. For CDNA-focused retrieval, treat RDNA PRs as compatibility evidence rather than direct CDNA performance claims. For RDNA-specific retrieval, use `kernel-rdna-rocm`.

## Evidence Map

| Evidence | What it contributes |
|----------|---------------------|
| `blog-flash-attention-rocm` | Baseline ROCm FlashAttention performance and backend context |
| `pr-flash-attention-157` | FlashAttention v3 enablement with FP8 and paged-attention paths |
| `pr-flash-attention-179` | FMHA backward `seq_q=0` NaN edge-case fix |
| `pr-flash-attention-181` | CK submodule and `fmha_bwd_args` compatibility update |
| `pr-flash-attention-182` | Unified workspace API migration for FMHA backward |
| `pr-flash-attention-183` | Stream-async workspace preparation for FMHA backward |
| `pr-flash-attention-178` | RDNA gfx11/gfx12 build support with backward guards |
| `pr-flash-attention-184` | Open RDNA backward enablement after CK update |

## References

- [vLLM ROCm Attention Backends](https://vllm.ai/blog/2026-02-27-rocm-attention-backend)
- [ROCm Flash Attention](https://github.com/ROCm/flash-attention)
- [Composable Kernel Library](https://github.com/ROCm/composable_kernel)
- [Matrix Core Programming CDNA3/CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)

---
id: lang-triton-rocm
title: Triton on ROCm (Triton-ROCm Backend)
type: wiki-language
architectures: [cdna2, cdna3, cdna4]
tags: [triton-rocm, programming, programming-model, optimization]
confidence: source-reported
languages: [triton-rocm]
sources:
  - pr-hipblaslt-353
  - pr-hipblaslt-970
  - pr-hipblaslt-381
  - pr-hipblaslt-388
  - blog-amdgpu-kernel-opt
related:
  - technique-ck-tile-programming
  - hw-mfma-matrix-core
  - hw-scaled-mfma
  - lang-ck-dsl
---

# Triton on ROCm (Triton-ROCm Backend)

Triton is a Python-based DSL for writing GPU kernels at a higher abstraction level than HIP/SASS. On AMD GPUs, the Triton-ROCm backend compiles Triton kernels to CDNA ISA via MLIR, targeting MFMA/Scaled-MFMA instructions and generating LDS tiling automatically. It is the primary path for vLLM, SGLang, and other inference frameworks running on ROCm.

## Why Triton on ROCm

| Aspect | HIP C++ / CK | Triton-ROCm |
|--------|-------------|-------------|
| Abstraction level | Tile/pipeline level | Block/pointer level |
| MFMA dispatch | Manual (inline asm or CK templates) | Automatic (compiler selects `v_mfma_*`) |
| LDS management | Manual layout + padding | Automatic tiling |
| Block-scale / Scaled MFMA | CK Tile ABQuant pipeline | Automatic via `tl.dot` with scaled dtypes |
| Tuning | Manual or TensileLite auto-tune | `@triton.autotune` decorator |
| Debuggability | Full control, easier printf | Python-level, harder to inspect asm |
| Performance ceiling | Highest (hand-tuned) | Close to hand-tuned for standard patterns |

## Kernel Structure

```python
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,       # pointers to A, B, C matrices
    M, N, K,                    # dimensions
    stride_am, stride_ak,       # A strides
    stride_bk, stride_bn,       # B strides
    stride_cm, stride_cn,       # C strides
    BLOCK_M: tl.constexpr,      # tile sizes (compile-time)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID → tile coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K-loop (compiler generates MFMA pipelining)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am
                           + (offs_k[None, :] + k) * stride_ak)
        b = tl.load(b_ptr + (offs_k[:, None] + k) * stride_bk
                           + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)  # → v_mfma_f32_*_f16 on CDNA

    # Store result
    tl.store(c_ptr + offs_m[:, None] * stride_cm
                   + offs_n[None, :] * stride_cn, acc)
```

The compiler maps `tl.dot(a, b)` to the appropriate `v_mfma_*` or `v_mfma_scale_*` instruction based on the input dtypes and target architecture.

## Key Capabilities on ROCm

### GEMM (Dense & Grouped)

- **Dense GEMM**: Standard tiled GEMM with autotuning over BLOCK_M/N/K and num_warps
- **Grouped GEMM**: Batched GEMM with variable M per group — critical for MoE. hipBLASLt provides a work-stealing variant ([`pr-hipblaslt-353`](../../sources/prs/hipblaslt/PR-353.md)) with a `ws_mode` API for dynamic load balancing
- **FP4 GEMM**: MXFP4 + NVFP4 support on MI300X/MI350X ([`pr-hipblaslt-381`](../../sources/prs/hipblaslt/PR-381.md)) via scaled-MFMA with block-scale

### MoE (Mixture of Experts)

- **Grouped Triton GEMM for TTFT** ([`pr-hipblaslt-970`](../../sources/prs/hipblaslt/PR-970.md)): Optimized Time-To-First-Token for MoE models by reducing kernel launch overhead and improving expert-load balance
- **BF16 Grouped GEMM backward** ([`pr-hipblaslt-388`](../../sources/prs/hipblaslt/PR-388.md)): Gradient computation for MoE training

### Block-Scaled MFMA

On CDNA4 (gfx950), Triton can target `v_mfma_scale_*` instructions for block-scaled FP4/FP6/FP8 matmul. The compiler handles scale-factor layout and packing when the kernel uses FP4/FP8 dtypes with the scaled-MFMA path.

## Autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8),
        triton.Config({'BM': 128, 'BN': 64,  'BK': 32}, num_warps=4),
        triton.Config({'BM': 64,  'BN': 128, 'BK': 32}, num_warps=4),
        triton.Config({'BM': 64,  'BN': 64,  'BK': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K'],  # autotune per shape
)
@triton.jit
def tuned_gemm_kernel(...):
    ...
```

Autotuning sweeps over tile sizes and warp counts, benchmarking each config on the target GPU. On CDNA architectures, `num_warps` controls how many wavefronts cooperate on a tile (each wavefront is 64 threads on AMD, vs 32 per warp on NVIDIA — halve your mental model of `num_warps`).

## Differences from Triton on NVIDIA

| Feature | NVIDIA (CUDA) | AMD (ROCm) |
|---------|---------------|------------|
| Warp size | 32 | 64 (wavefront) |
| Shared memory | `tl.load` with eviction policy | LDS (same API, different bank structure) |
| Tensor cores | `tl.dot` → `mma.sync` | `tl.dot` → `v_mfma_*` |
| Block scale | NVFP4 via `tl.dot` with scaled types | `v_mfma_scale_*` via same API |
| Barrier | `tl.debug_barrier()` | Same API → `s_barrier` or `ds_gws_*` |
| Grid synchronization | Cooperative groups | GWS (transparent via runtime) |

## Performance Tips

1. **Tile size**: AMD MFMA instructions operate on 16×16 or 32×32 tiles. Match BLOCK_M/N to multiples of 16 for best utilization.
2. **num_warps**: A wavefront is 64 threads. `num_warps=4` means 256 threads (4 wavefronts) per block. On CDNA3/4, 4–8 wavefronts per block is typical for compute-bound kernels.
3. **LDS bank conflicts**: The compiler handles basic padding, but pathological access patterns may still conflict. Profile with `rocprof` and add manual padding if needed.
4. **Vectorized loads**: Use `tl.load` with 128-bit alignment for maximum global-memory throughput.

## References

- [AMDGPU Kernel Optimization Guide](../../sources/blogs/amdgpu-kernel-opt.md)
- [MFMA Matrix Core](../hardware/mfma-matrix-core.md) — what `tl.dot` compiles to
- [Scaled MFMA](../hardware/scaled-mfma.md) — block-scaled path on CDNA4
- [CK Tile Programming](../techniques/ck-tile-programming.md) — alternative HIP-based approach

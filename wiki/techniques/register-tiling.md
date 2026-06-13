---
id: technique-register-tiling
title: Register Tiling for MFMA Kernels
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [register-tiling, mfma, optimization, vgpr]
confidence: source-reported
techniques: [register-tiling]
hardware_features: [mfma]
kernel_types: [gemm, attention, moe]
related: [hw-mfma-matrix-core, technique-mfma-scheduling, technique-double-buffering]
sources: [doc-cdna4-whitepaper, blog-amdgpu-kernel-opt, doc-ck-structure]
reproducibility: snippet
---

# Register Tiling for MFMA Kernels

Keeping intermediate data in VGPR (vector general-purpose registers) to minimize LDS traffic and maximize MFMA throughput.

## Core Idea

Instead of storing intermediate results in LDS (shared memory), keep them in VGPR registers. MFMA instructions read directly from VGPR and write results to VGPR — no LDS round-trip needed for the inner compute loop.

```
Without register tiling:
  Global → LDS → VGPR → MFMA → LDS → Global
                    ↑             ↓
                    └─ LDS round-trip ─┘

With register tiling:
  Global → LDS → VGPR → MFMA → VGPR → Global
                 ↑                ↓
                 └─ stays in VGPR ─┘
```

## VGPR Budget Planning

Each CDNA CU has 65,536 VGPRs total, split across active wavefronts:

| Waves/CU | VGPRs/Wavefront | VGPRs for MFMA accum | Free for other data |
|----------|-----------------|---------------------|-------------------|
| 8 | 512 | 256 (32×32 accum) | 256 |
| 16 | 256 | 128 (16×16 accum) | 128 |
| 32 | 128 | 64 (4×4 accum) | 64 |

**Key tradeoff**: More VGPRs per wavefront → fewer wavefronts per CU → lower occupancy.

## Tiling Strategy

### Example: 16×16 FP16 GEMM with Register Tiling

```c
// Each wavefront holds a 16×16 F32 accumulator in 16 VGPRs
float16 accum = {0};  // 16 VGPRs

// Inner K-loop: data stays in VGPR
for (int k = 0; k < K; k += 16) {
    // Load A tile (16×16 fp16 = 8 VGPRs) from LDS to VGPR
    float8 a_tile = load_from_lds(a_ptr + k);

    // Load B tile (16×16 fp16 = 8 VGPRs) from LDS to VGPR
    float8 b_tile = load_from_lds(b_ptr + k);

    // MFMA: accum += a_tile × b_tile (all in VGPR)
    accum = mfma_f32_16x16x16f16(accum, a_tile, b_tile);

    // No LDS write needed! accum stays in VGPR
}

// Write accum to global memory (once, after K loop)
store_to_global(c_ptr, accum);
```

### VGPR Usage Breakdown

| Data | VGPRs | Purpose |
|------|-------|---------|
| Accumulator (16×16 F32) | 16 | Output matrix tile |
| A tile (16×16 FP16) | 8 | Input matrix A |
| B tile (16×16 FP16) | 8 | Input matrix B |
| Buffer A (next tile) | 8 | Double buffer |
| Buffer B (next tile) | 8 | Double buffer |
| Index/pointer vars | ~4 | Loop control |
| **Total** | **~52** | Per wavefront |

With 52 VGPRs/wavefront → max ~12 wavefronts/CU (65536/52/64 ≈ 12).

## Comparison: LDS vs Register Tiling

| Aspect | LDS Tiling | Register Tiling |
|--------|-----------|----------------|
| Latency | LDS read ~4-8 cycles | VGPR access ~0 cycles |
| Bandwidth | Limited (32 banks) | Unlimited (register file) |
| Bank conflicts | Possible | Not applicable |
| Capacity | 64 KB/CU | ~256 KB/CU (at 8 waves) |
| Shared | Yes (across wavefronts) | No (per-wavefront) |
| Complexity | Lower | Higher |

## When to Use

| Scenario | Recommended |
|----------|-------------|
| Standard GEMM (large M, N) | Register tiling (CK default) |
| Small batch GEMM | LDS tiling (more sharing) |
| Flash Attention Q×K^T | Register tiling (Q stays in VGPR) |
| Reduction patterns | LDS tiling (cross-wave needed) |

## CK Library Default

CK uses register tiling by default for its tiled GEMM implementations. The `ck_tile` framework provides:

```cpp
// CK handles register tile allocation automatically
using TileDistribution = ck::make_tile_distribution<
    ck::Pattern<...>,  // How data maps to VGPR
    ck::NumDim<2>,     // 2D tile
    ck::Size<16, 16>   // 16×16 per wavefront
>();
```

## References

- [CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/)
- [AMDGPU Kernel Optimization Guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md)
- [CK Library Structure](https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/Composable-Kernel-structure.html)

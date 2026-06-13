---
id: migration-cdna3-to-cdna4
title: CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide
type: wiki-migration
architectures: [cdna3, cdna4]
tags: [migration, cdna3, cdna4, mfma, scaled-mfma, lds-transpose, block-scale]
confidence: source-reported
sources:
  - doc-cdna4-isa
  - doc-cdna4-whitepaper
  - pr-composable_kernel-3603
  - pr-composable_kernel-3620
  - pr-composable_kernel-3540
  - pr-composable_kernel-3636
  - pr-rocm_libraries-5813
  - pr-aiter-3470
related:
  - hw-scaled-mfma
  - hw-lds-transpose
  - hw-mfma-matrix-core
  - hw-gws
  - migration-cuda-to-hip
from_architecture: cdna3
to_architecture: cdna4
difficulty: moderate
---

# CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide

CDNA4 (gfx950 / MI350X / MI355X) is a superset of CDNA3 at the instruction level — existing CDNA3 kernels run unmodified. The migration gains come from three new hardware features that require explicit opt-in: **Scaled MFMA** (block-scaled FP4/FP6/FP8), **LDS read-with-transpose**, and **dual-CMA** (2× matrix accelerator per CU). This guide covers what to change and what stays the same.

## What Stays the Same

| Feature | CDNA3 | CDNA4 | Migration action |
|---------|-------|-------|-----------------|
| Standard `v_mfma_*` | ✓ | ✓ | None — all CDNA3 MFMA opcodes work |
| FP8/BF8 MFMA | ✓ | ✓ | None — `v_mfma_f32_*_fp8_*` unchanged |
| LDS (64 KB/CU) | ✓ | ✓ | None — same capacity and bank structure |
| DPP / bpermute | ✓ | ✓ | None — same cross-lane ops |
| GWS | ✓ | ✓ | None — same `ds_gws_*` barrier API |
| Wavefront (64 threads) | ✓ | ✓ | None |
| `hipLaunchCooperativeKernel` | ✓ | ✓ | None |

## New Features Requiring Code Changes

### 1. Scaled MFMA (`v_mfma_scale_*`)

The biggest change. CDNA4 adds block-scaled matrix instructions that apply per-block scale factors during matmul, enabling efficient FP4/FP6/FP8 GEMM without a separate dequantize step.

**CDNA3 approach** (still works on CDNA4):
```c
// CDNA3: dequantize FP8 weights, then standard MFMA
// Step 1: Load FP8 weights to LDS
// Step 2: Dequantize to BF16 in VGPRs (scale * fp8_value)
// Step 3: v_mfma_f32_16x16x32_bf16 with BF16 operands
```

**CDNA4 approach** (opt-in for throughput and accuracy):
```c
// CDNA4: block-scaled MFMA — hardware applies scale factors
// Step 1: Load packed FP4/FP8 weights + per-block scale factors to LDS
// Step 2: v_mfma_scale_f32_16x16x128_f8f6f4
//         Hardware applies ScaleA ⊙ A × ScaleB ⊙ B internally
// No explicit dequantize step; wider K-dimension (128 vs 32 for BF16)
```

See [Scaled MFMA](../hardware/scaled-mfma.md) for the full instruction table and data layout.

**CK migration**: Use the `CK_USE_GFX950` macro ([`pr-composable_kernel-3636`](../../sources/prs/composable_kernel/PR-3636.md)) to enable block-scale paths. The ABQuant pipeline ([`pr-composable_kernel-3620`](../../sources/prs/composable_kernel/PR-3620.md), [`pr-composable_kernel-3603`](../../sources/prs/composable_kernel/PR-3603.md)) handles the data layout and instruction selection automatically.

### 2. LDS Read-with-Transpose

CDNA4 adds LDS load instructions that read and transpose in a single operation, eliminating the software transpose step in GEMM epilogues and attention kernels.

**CDNA3 approach**:
```asm
; Two-step: read LDS, then shuffle/transpose in VGPRs
ds_read_b128 v[0:3], v0 offset:0
; ... explicit VGPR transpose ...
```

**CDNA4 approach**:
```asm
; One-step: hardware transpose during LDS read
ds_read_b128_tr_b16 v[0:3], v0 offset:0
```

See [LDS Read-with-Transpose](../hardware/lds-transpose.md) for the instruction family. CK Tile enables this automatically for transpose layouts ([`pr-rocm_libraries-5813`](../../sources/prs/hipblaslt/PR-5813.md)).

### 3. Dual CMA (Dual Compute Matrix Accelerator)

CDNA3 has one CMA per CU; CDNA4 has two. This doubles peak MFMA throughput per CU without any code changes for standard MFMA. For scaled MFMA, dual-CMA further increases the FP4/FP8 FLOPS ceiling.

**Migration action**: Re-tune occupancy and workgroup sizes. With 2× MFMA units, the balance between memory-bound and compute-bound shifts — kernels that were near the roofline on CDNA3 may now have MFMA headroom, making memory layout and LDS bandwidth the new bottleneck.

## Migration Checklist

### Compile-time

- [ ] Set `-DCK_USE_GFX950=ON` (CK) or equivalent target flag for gfx950
- [ ] Re-run tuning/benchmarking — occupancy parameters from CDNA3 may be suboptimal
- [ ] If using block-scale: add scale-factor allocation and layout code (see [Scaled MFMA](../hardware/scaled-mfma.md) data layout)

### Kernel-level

- [ ] **GEMM**: Evaluate block-scaled MFMA path vs standard FP8 MFMA. Block-scale wins when weight quantization to FP4/FP6 is acceptable (inference MoE, quantized training)
- [ ] **Attention**: Consider LDS transpose for attention-score epilogue (saves one LDS round-trip). Block-scale FP8 attention is also available ([`pr-composable_kernel-3635`](../../sources/prs/composable_kernel/PR-3635.md))
- [ ] **MoE / Grouped GEMM**: Use persistent-kernel pattern with GWS for dynamic expert routing. See [MoE Grouped GEMM](../kernels/moe-grouped-gemm-cdna4.md) for the CDNA4-specific paths
- [ ] **Conv**: WMMA grouped-conv paths have CDNA4-specific tuning. Direct-load LDS transpose can fuse im2col layout conversion

### Infrastructure

- [ ] Update ROCm version to 6.x+ (required for gfx950 ISA support)
- [ ] If using TensileLite: update tuning libraries for gfx950 ([`pr-rocm_libraries-7767`](../../sources/prs/hipblaslt/PR-7767.md), [`pr-rocm_libraries-7782`](../../sources/prs/hipblaslt/PR-7782.md))
- [ ] Re-validate numerical accuracy — block-scaled FP4 has different rounding than FP8 dequantize

## Common Pitfalls

1. **Assuming `v_smfmac_*` is the scaled-MFMA instruction.** It is not — `v_smfmac_*` is the *structured-sparsity* MFMA. Block-scaled MFMA uses `v_mfma_scale_*` (see [Scaled MFMA](../hardware/scaled-mfma.md)).

2. **Forgetting scale-factor layout.** Scaled MFMA requires `ScaleA: [M/16, K/128]` and `ScaleB: [K/128, N/16]` — these are new tensors that don't exist in the standard GEMM interface. The kernel launcher must allocate and populate them.

3. **Using CDNA3 occupancy tuning on CDNA4.** Dual-CMA changes the VGPR/MFMA balance. A kernel that was compute-bound on MI300X may now be memory-bound on MI350X at the same tile sizes.

4. **Not opting in to LDS transpose.** The compiler does not automatically use `ds_read_*_tr_*` — you must either use CK Tile's transpose-aware pipeline or write inline asm. Standard LDS reads still use the non-transpose path.

## References

- [CDNA4 ISA Reference](../../sources/docs/cdna4-isa.md) — authoritative instruction encodings
- [CDNA4 Whitepaper](../../sources/docs/cdna4-whitepaper.md) — dual-CMA, memory hierarchy
- [Scaled MFMA](../hardware/scaled-mfma.md) — block-scaled MFMA instruction details
- [LDS Transpose](../hardware/lds-transpose.md) — read-with-transpose instructions
- [MFMA Matrix Core](../hardware/mfma-matrix-core.md) — standard MFMA across CDNA generations
- [CUDA → HIP Migration](cuda-to-hip.md) — if also migrating from CUDA

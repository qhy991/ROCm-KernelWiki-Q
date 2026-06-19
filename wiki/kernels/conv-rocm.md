---
id: kernel-conv-rocm
title: Convolution Kernels on ROCm (CK Grouped Conv)
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [conv, grouped-gemm, optimization, mfma]
confidence: source-reported
kernel_types: [conv, grouped-gemm]
languages: [hip-cpp, ck-dsl, assembly, python]
techniques: [ck-tile-programming, mfma-scheduling, double-buffering, occupancy-tuning]
hardware_features: [mfma, lds]
related:
  - hw-mfma-matrix-core
  - hw-lds
  - technique-ck-tile-programming
  - technique-occupancy-tuning
  - lang-amd-assembly
  - kernel-moe-grouped-gemm-cdna4
sources:
  - pr-composable_kernel-3547
  - pr-composable_kernel-3556
  - pr-composable_kernel-3562
  - pr-composable_kernel-3582
  - pr-composable_kernel-3595
  - pr-composable_kernel-3632
  - pr-composable_kernel-3648
  - pr-composable_kernel-3624
  - pr-miopen-3702
  - pr-miopen-3918
  - pr-miopen-3925
  - doc-ck-readme
  - doc-ck-structure
reproducibility: concept
---

# Convolution Kernels on ROCm (CK Grouped Conv)

Convolution on AMD CDNA GPUs is implemented primarily through the Composable Kernel (CK) library, which decomposes convolution into GEMM-like tile operations using MFMA instructions. CK supports both direct convolution and GEMM-based (im2col) approaches, with grouped convolution being the most actively optimized path across CDNA2–CDNA4.

## Convolution → GEMM Mapping

Convolutions are typically mapped to GEMM internally, allowing reuse of the highly-optimized MFMA tiling infrastructure:

```
Conv2D forward (N, C, H, W) × (K, C/G, R, S) → (N, K, OH, OW)

Maps to GEMM:
  A: im2col input  [N×OH×OW, C×R×S]    (or implicit via direct load)
  B: weight        [K, C×R×S]
  C: output        [N×OH×OW, K]

Grouped conv (G groups):
  G independent GEMMs, each (N×OH×OW, C/G×R×S) × (K/G, C/G×R×S) → (N×OH×OW, K/G)
```

CK provides both **im2col** (explicit reshape) and **direct-load** (implicit on-the-fly im2col in LDS/VGPR) paths. The direct-load path avoids the memory overhead and bandwidth cost of materializing the expanded input.

## CK Conv Architecture

```
┌──────────────────────────────────────────────────────┐
│ Host: DeviceGroupedConvFwdXdl                        │
│   Grid: (N·OH·OW / TILE_M) × (K/G / TILE_N) × G    │
└───────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│ Kernel: GridwiseGemm (conv-configured)               │
│                                                       │
│   for each K-tile:                                    │
│     1. Load input tile → LDS (direct load or im2col) │
│     2. Load weight tile → LDS                         │
│     3. v_mfma_f32_*_f16  (CDNA2/3)                   │
│        v_mfma_f32_*_bf16 (CDNA2/3)                   │
│        v_mfma_f32_*_fp8  (CDNA3/4)                   │
│     4. Shuffle C-tile → global memory                 │
│                                                       │
│   Pipeline: double-buffer LDS (load next while MFMA) │
└──────────────────────────────────────────────────────┘
```

## Forward Convolution Variants

### XDL (MFMA) Forward

The standard path uses XDL (cross-lane data movement) with MFMA. Two sub-variants exist:

## MIOpen Solver and Heuristic Layer

CK pages explain many conv kernels as GEMM-like tiles, but MIOpen adds another layer: solver selection. The chosen solver may be a CK/XDL implicit-GEMM path, a direct solver, or a handwritten assembly Winograd path.

| Solver path | What it contributes | Evidence |
|-------------|---------------------|----------|
| Winograd Rage assembly | Handwritten gfx942 assembly kernels selected by filter size | `pr-miopen-3702` |
| 3D conv KTN heuristics | TunaNet input/config encoders score gfx942 CK 3D grouped conv candidates | `pr-miopen-3918` |
| 3D conv AI heuristics | TunaNet model and metadata extraction for gfx942 3D convolution candidate selection | `pr-miopen-3925` |

This means a convolution performance issue can live in three places: the kernel implementation, the solver heuristic, or the model metadata used to select a candidate. Do not assume a slower conv shape requires a new kernel before checking MIOpen's selection path.

- **Im2col**: Explicit input expansion to GEMM layout. Simpler but uses more memory.
- **Direct load** ([`pr-composable_kernel-3632`](../../sources/prs/composable_kernel/PR-3632.md)): Implicit on-the-fly im2col during LDS load. Eliminates the expansion buffer; the kernel computes the im2col index mapping in hardware. Available with vector load sizes 1 and 2.

### WMMA Forward

CK also provides WMMA (Warp Matrix Multiply-Accumulate) paths for grouped convolution, optimized for large tensors:

- **Large tensor flavors** ([`pr-composable_kernel-3582`](../../sources/prs/composable_kernel/PR-3582.md)): Additional kernel variants tuned for large spatial dimensions (high H×W)
- **Fused epilogue** ([`pr-composable_kernel-3595`](../../sources/prs/composable_kernel/PR-3595.md)): Bias + BatchNorm + Clamp fused into the conv epilogue, avoiding separate kernel launches for common inference patterns

```c
// Fused conv + bias + batchnorm + clamp (WMMA path)
// Single kernel instead of 4 separate launches
conv_fwd_output = clamp(bn(conv(input, weight) + bias))
```

## Backward Convolution

### Backward Data

Gradient w.r.t. input — essentially a transposed convolution:

```c
// Backward data = deconvolution = conv with flipped weights
// CK maps to GEMM with col2im epilogue
DeviceGroupedConvBwdDataXdl
```

A WMMA correctness fix ([`pr-composable_kernel-3562`](../../sources/prs/composable_kernel/PR-3562.md)) adds a check for insufficient iterations when increasing waves per block — a tuning edge case that silently produced wrong results on certain tile configurations.

### Backward Weight

Gradient w.r.t. weights — the most memory-intensive backward pass:

```c
// For each group g:
//   dW[g] = input[g]^T @ doutput[g]  (GEMM mapping)
```

The direct-load backward weight path ([`pr-composable_kernel-3648`](../../sources/prs/composable_kernel/PR-3648.md)) avoids materializing the expanded activation tensor:

- Refactored `gridwise_gemm_xdl_cshuffle_conv_v3` (+273/−144 lines) for conv-aware tiling
- Added XDLOPS pipeline variants optimized for the conv weight gradient access pattern
- New BF16/F16 direct-load instances for grouped conv bwd weight

## CK Builder (Next-Gen Interface)

CK Builder is a new high-level API for configuring and dispatching convolution operations:

- **Conv traits refactored** ([`pr-composable_kernel-3547`](../../sources/prs/composable_kernel/PR-3547.md)): Convolution descriptors converted to structs with factory functions, replacing the previous trait-based metaprogramming
- **Test infrastructure** ([`pr-composable_kernel-3556`](../../sources/prs/composable_kernel/PR-3556.md)): Builder-based test framework for grouped conv forward, supporting ndhwgc/nhwgc tensor formats
- **CI enablement** ([`pr-composable_kernel-3624`](../../sources/prs/composable_kernel/PR-3624.md)): Grouped conv forward tile tests enabled in daily CI

## Data Formats and Tensor Layouts

CK supports several tensor layout conventions for convolutions:

| Format | Layout | Use case |
|--------|--------|----------|
| NHWC | `[N, H, W, C]` | Standard inference layout |
| NCHW | `[N, C, H, W]` | PyTorch default |
| NDHWC | `[N, D, H, W, C]` | 3D conv (video, medical imaging) |
| GNHWC | `[G, N, H, W, C]` | Grouped conv with explicit group dimension |
| NHWGC | `[N, H, W, G, C]` | Channel-last grouped layout |

The CK Tile tests cover BF16, FP16, and FP32 for both 2D and 3D grouped conv.

## Performance Considerations

1. **Direct load vs im2col**: Direct load saves memory bandwidth and allocation. Prefer it for large spatial dims where the im2col buffer would exceed LDS.

2. **Tile size**: MFMA tile sizes (16×16×16 for BF16, 16×16×32 for FP8) determine the inner-loop K-step. Larger tiles amortize LDS load overhead but increase register pressure.

3. **Group count**: Grouped conv with many small groups benefits from persistent-kernel patterns (reuse wavefronts across groups). See the [Persistent Kernel Pattern](../techniques/persistent-kernel.md).

4. **Fused epilogue**: For inference, always prefer the fused bias+bnorm+clamp path over separate kernels — saves 2–3 kernel launches per layer.

5. **Backward weight tuning**: The most sensitive backward pass. Direct load with XDLOPS pipeline is typically 1.5–2× faster than im2col for weight gradients.

## References

- [CK Library README](../../sources/docs/ck-readme.md) — library overview
- [CK Structure](../../sources/docs/ck-ck-structure.md) — codebase layout
- [MFMA Matrix Core](../hardware/mfma-matrix-core.md) — MFMA instruction details
- [LDS](../hardware/lds.md) — LDS layout and bank conflicts
- [MoE Grouped GEMM](moe-grouped-gemm-cdna4.md) — related grouped-GEMM kernel type

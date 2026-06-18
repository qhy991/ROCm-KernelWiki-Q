---
id: lang-ck-dsl
title: Composable Kernel (CK) DSL
type: wiki-language
architectures: [cdna2, cdna3, cdna4]
tags: [ck-dsl, composable_kernel, programming, library]
confidence: source-reported
sources: [doc-ck-structure, doc-ck-readme]
related: [technique-ck-tile-programming, hw-mfma-matrix-core, kernel-flash-attention-rocm]
languages: [ck-dsl, hip-cpp]
---

# Composable Kernel (CK) DSL

AMD's high-performance kernel programming library for CDNA GPUs — the ROCm equivalent of NVIDIA's CUTLASS.

## Overview

| Property | Value |
|----------|-------|
| GitHub | https://github.com/ROCm/composable_kernel |
| Language | C++ (heavily templated) |
| License | MIT-like (BSD-3) |
| Min ROCm | 5.3+ |
| Supported Arch | CDNA2 (gfx90a), CDNA3 (gfx940/gfx942), CDNA4 (gfx950) |

## Four-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Client API                                         │
│   Direct calls from hipBLASLt, MIOpen, vLLM, etc.          │
│   Pre-instantiated with common tile sizes and data types    │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Instantiated Kernel & Invoker                      │
│   Concrete parameters: M/N/K tile sizes, MFMA variant      │
│   Ready to launch on GPU                                    │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Templated Kernel & Invoker                         │
│   Generic over tile sizes, layouts, data types              │
│   Compose operators at compile time                         │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Tile Operators                                      │
│   Primitive operations: load, store, MFMA, reduce, copy    │
│   Each maps to specific hardware instructions              │
└─────────────────────────────────────────────────────────────┘
```

## Supported Operations

| Operation | Description | Status |
|-----------|-------------|--------|
| GEMM | Dense matrix multiply | ✓ Stable |
| Batched GEMM | Batched dense matmul | ✓ Stable |
| Grouped GEMM | Variable-size grouped matmul (MoE) | ✓ Stable |
| GEMM+GEMM | Fused dual GEMM (gate+up) | ✓ Stable |
| GEMM+Permute | GEMM with output permutation | ✓ Stable |
| Flash Attention | Scaled dot-product attention | ✓ Stable |
| Grouped GEMM+GEMM | Fused grouped dual GEMM | ✓ Stable |
| Contraction | General tensor contraction | Experimental |
| Convs | Forward/backward convolutions | ✓ Stable |
| Reduction | Various reduction patterns | ✓ Stable |
| Softmax | Online softmax | ✓ Stable |
| LayerNorm | Layer normalization | ✓ Stable |

## Quick Start: GEMM

```cpp
#include "ck/library/tensor_operation_instance/gpu/gemm_bilinear.hpp"

// CK provides pre-built instances
using GemmInstance = ck::tensor_operation::device::DeviceGemmBilinear<
    ck::tensor_layout::RowMajor,     // A layout
    ck::tensor_layout::ColumnMajor,  // B layout
    ck::tensor_layout::RowMajor,     // C layout
    ck::half_t,                      // A type
    ck::half_t,                      // B type
    ck::half_t,                      // C type
    ck::half_t,                      // Accumulator type
    ck::tensor_operation::element_wise::PassThrough,  // A element op
    ck::tensor_operation::element_wise::PassThrough,  // B element op
    ck::tensor_operation::element_wise::PassThrough   // C element op
>;

// Query available instances (different tile sizes, pipelines)
auto instances = GemmInstance::GetInstances();

// Select and run
auto gemm = instances[0];
auto argument = gemm->MakeArgument(
    a_device, b_device, c_device,
    M, N, K,
    StrideA, StrideB, StrideC,
    a_element_op, b_element_op, c_element_op);

auto invoker = gemm->MakeInvoker();
invoker.Run(argument, stream);
```

## Key Design Patterns

### Pipeline (Double/Triple Buffering)

```cpp
// Configure pipeline depth for overlapping HBM→LDS→VGPR
static constexpr index_t kPipeline = 2;  // 2-stage double buffer
// Or: kPipeline = 3 for triple buffer (more LDS, better overlap)
```

### Epilogue Fusion

```cpp
// Fuse GEMM with element-wise operations
struct ScaleAndBias {
    float scale;
    const float* bias;

    __device__ float operator()(float val, index_t row, index_t col) const {
        return val * scale + bias[col];
    }
};
// Applied during GEMM output write — no extra kernel launch
```

### Tile Specialization

```cpp
// CK auto-selects optimal MFMA tile size based on template params
// But you can force specific tiles:
using TileSize = ck::GemmTile<128, 128, 32>;  // M=128, N=128, K=32 per block
```

## Comparison with CUTLASS

| Feature | CUTLASS | CK |
|---------|---------|-----|
| Language | C++ | C++ |
| Abstraction | MMA → Warp → Threadblock | Tile → Kernel → Client |
| GEMM support | ✓ | ✓ |
| Attention | CUTLASS 3.x | CK SDPA |
| Grouped GEMM | ✓ (CUTLASS 3.x) | ✓ |
| MoE | ✓ | ✓ |
| Pipeline | Sm80+ cp.async | Flat loads + LDS |
| Accumulator | Register / TMEM | VGPR |
| Epilogue fusion | ✓ | ✓ |
| Community | Very large | Growing |

## References

- [CK Library (GitHub)](https://github.com/ROCm/composable_kernel)
- [CK Structure Documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/Composable-Kernel-structure.html)
- [CK Installation Guide](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/install/Composable-Kernel-install.html)
- [ck_tile README](https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/README.md)

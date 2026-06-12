---
id: technique-ck-tile-programming
title: CK Tile Programming Model
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [ck-tile, composable-kernel, programming-model, tiling]
confidence: source-reported
techniques: [ck-tile-programming, register-tiling, double-buffering]
hardware_features: [mfma, lds]
kernel_types: [gemm, attention, moe, grouped-gemm]
related: [hw-mfma-matrix-core, kernel-flash-attention-rocm]
sources: [doc-ck-structure, blog-ck-tutorial]
reproducibility: snippet
---

# CK Tile Programming Model

The Composable Kernel (CK) tile-based programming model for writing performance-critical GPU kernels on AMD CDNA architectures.

## Overview

CK provides a 4-layer abstraction:

```
Layer 4: Client API           ← hipBLASLt, MIOpen integration
Layer 3: Instantiated Kernel  ← Concrete tile sizes, data types
Layer 2: Templated Kernel     ← Generic over tile sizes, layouts
Layer 1: Tile Operators       ← Primitive tile-level operations
```

### Key Abstractions

| Concept | Description | CUDA Equivalent |
|---------|-------------|-----------------|
| **Tile** | A fixed-size block of data in VGPR/LDS | Register tile / shared memory tile |
| **TileWindow** | View into a larger tensor | Thread map + shared memory view |
| **TileDistribution** | How tile data maps to lanes | Thread mapping (tiled, blocked) |
| **ElementOp** | Per-element operations | Epilogue functor |
| **BlockOperator** | Tile-level compute | GEMM/Attention building blocks |

## Basic Structure: GEMM Example

```c
#include "ck/ck.hpp"
#include "ck/tile_program.hpp"

// 1. Define tile types
using ADataType = ck::half_t;
using BDataType = ck::half_t;
using CDataType = float;

// 2. Define tile dimensions
static constexpr index_t kMPerBlock = 128;
static constexpr index_t kNPerBlock = 128;
static constexpr index_t kKPerBlock = 32;

// 3. CK handles:
//    - LDS bank conflict avoidance
//    - MFMA scheduling
//    - Double buffering
//    - Register pressure management
//    - Global memory vectorization
```

## ck_tile Core API

```c
#include "ck/tile/core/tile.hpp"

// Create a distributed tile (data split across wavefront lanes)
auto tile = ck::make_distributed_tile<float, 16, 16>(...);

// Load tile from global memory to LDS
ck::tile_load(tile_window, lds_tile);

// Load from LDS to VGPR
ck::tile_read(vgpr_tile, lds_tile);

// MFMA operation
auto result = ck::tile_mfma(a_tile, b_tile, c_tile);

// Store result
ck::tile_store(result_tile, output_window);
```

## Programming Steps

### Step 1: Define Kernel Traits

```c
template <typename ADataType, typename BDataType, typename CDataType,
          index_t MPerBlock, index_t NPerBlock, index_t KPerBlock>
struct MyGemmKernel {
    // A matrix: row-major
    using ALayout = ck::tensor_layout::RowMajor;
    // B matrix: column-major
    using BLayout = ck::tensor_layout::ColumnMajor;
    // C matrix: row-major
    using CLayout = ck::tensor_layout::RowMajor;

    // Pipeline: number of double-buffer stages
    static constexpr index_t kPipeline = 2;
};
```

### Step 2: Configure Tile Distribution

```c
// Wavefront-level distribution: how data maps to 64 lanes
// Common patterns:
// - WarpGemm: MFMA-friendly layout for 16x16 tiles
// - BlockMapped: One tile per block subset
using ADistribution = ck::WarpGemmADistribution<MPerBlock, KPerBlock>;
using BDistribution = ck::WarpGemmBDistribution<KPerBlock, NPerBlock>;
using CDistribution = ck::WarpGemmCDistribution<MPerBlock, NPerBlock>;
```

### Step 3: Write Kernel Body

```c
__global__ void gemm_kernel(
    const __half* a, const __half* b, float* c,
    index_t M, index_t N, index_t K) {

    // CK generates the tiled loop structure:
    // 1. Load A tile from HBM → LDS (async)
    // 2. Load B tile from HBM → LDS (async)
    // 3. Wait for loads
    // 4. Read A, B from LDS → VGPR
    // 5. MFMA: C += A * B
    // 6. Double buffer: start loading next tiles
    // 7. After K loop: write C from VGPR → HBM
}
```

## Performance Expectations

| Kernel Type | CK Achieved (MI300X) | Theoretical Peak | Utilization |
|-------------|---------------------|------------------|-------------|
| FP16 GEMM (large) | ~580 TFLOPS | 654.7 TFLOPS | ~89% |
| BF16 GEMM (large) | ~575 TFLOPS | 654.7 TFLOPS | ~88% |
| FP8 GEMM (large) | ~1100 TFLOPS | 1309 TFLOPS | ~84% |
| Flash Attention (8K) | ~520 TFLOPS | 654.7 TFLOPS | ~79% |

## When to Use CK vs Raw HIP

| Scenario | Recommended |
|----------|------------|
| Standard GEMM (dense) | CK (handles all tiling) |
| Flash Attention | CK SDPA |
| Grouped GEMM (MoE) | CK grouped_gemm |
| Custom fused kernel | CK tile primitives |
| Novel algorithm | Raw HIP + rocWMMA |
| Maximum control | Inline ASM |

## References

- [CK Library (GitHub)](https://github.com/ROCm/composable_kernel)
- [CK Structure Documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/Composable-Kernel-structure.html)
- [CK Hello World Tutorial](https://rocm.docs.amd.com/projects/composable_kernel/en/docs-5.7.1/tutorial_hello_world.html)

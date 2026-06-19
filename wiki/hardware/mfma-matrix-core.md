---
id: hw-mfma-matrix-core
title: MFMA Matrix Core (CDNA1–CDNA4)
type: wiki-hardware
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [mfma, compute, hardware]
confidence: verified
hardware_features: [mfma, dual-cma]
related: [hw-lds, hw-dpp-cross-lane, hw-wavefront]
sources: [doc-cdna4-isa, doc-cdna4-whitepaper, blog-matrix-cores-cdna, pr-rocwmma-500, pr-rocwmma-510]
cuda_equivalent: tensor_core
---

# MFMA Matrix Core

AMD's Matrix Fused Multiply-Add (MFMA) unit — the hardware block responsible for matrix operations on CDNA GPUs.

## Overview

MFMA is AMD's equivalent to NVIDIA's Tensor Core. Each CU contains one or more Compute Matrix Accelerator (CMA) units that execute `v_mfma_*` instructions.

| Architecture | CMAs per CU | Peak FP16 TFLOPS (per GPU) |
|-------------|-------------|---------------------------|
| CDNA1 (MI100) | 1 | 184.6 |
| CDNA2 (MI250X) | 1 | 383.0 |
| CDNA3 (MI300X) | 2 | 654.7 |
| CDNA4 (MI350X) | 2 | ~900+ |

## Supported Data Types

### MFMA Input → Accumulator Combinations

| Input Type | Accumulator | Instruction Pattern | Min Arch |
|-----------|-------------|-------------------|----------|
| F32 → F32 | F32 | `v_mfma_f32_*_f32` | CDNA1 |
| F16 → F32 | F32 | `v_mfma_f32_*_f16` | CDNA1 |
| BF16 → F32 | F32 | `v_mfma_f32_*_bf16` | CDNA1 |
| I8 → I32 | I32 | `v_mfma_i32_*_i8` | CDNA1 |
| FP8 (FP8/BF8) → F32 | F32 | `v_mfma_f32_*_fp8_*` | CDNA3 |
| F6/F4 → F32 | F32 | `v_mfma_f32_*_f6_*` | CDNA3 |
| **Scaled F8/F6/F4** → F32 | F32 | `v_mfma_f32_*_scale_*` | **CDNA4** |

## Tile Sizes

### Common MFMA Tile Dimensions

| Tile (M×N×K) | Input Dtype | VGPRs (Accum) | FLOPs/inst | Use Case |
|--------------|-------------|---------------|------------|----------|
| 16×16×1 | F32 | 16 | 512 | Rank-1 update |
| 16×16×4 | F32 | 16 | 2048 | FP32 GEMM building block |
| 32×32×2 | F32 | 32 | 4096 | Large FP32 tiles |
| 4×4×1 | F32 | 1 | 8 | Minimal |
| 16×16×16 | F16 | 16 | 8192 | **FP16 GEMM workhorse** |
| 32×32×8 | F16 | 32 | 16384 | Large FP16 tiles |
| 16×16×32 | BF16 | 16 | 16384 | BF16 GEMM |
| 32×32×16 | BF16 | 32 | 32768 | Large BF16 tiles |
| 16×16×32 | FP8 | 16 | 16384 | FP8 GEMM (CDNA3+) |
| 16×16×64 | FP8 | 16 | 32768 | High-throughput FP8 (CDNA3+) |

## Programming Interfaces

### 1. Direct Inline ASM

```c
// v_mfma_f32_16x16x16f16 v[0:15], v[16:23], v[24:31], v[0:15]
asm volatile(
    "v_mfma_f32_16x16x16f16 %0, %1, %2, %0"
    : "+v"(accum[0])
    : "v"(mat_a[0]), "v"(mat_b[0])
);
```

### 2. rocWMMA (CUDA WMMA Equivalent)

```c
#include <rocwmma/rocwmma.h>
using namespace rocwmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

mma_sync(c_frag, a_frag, b_frag, c_frag);
```

### 3. Composable Kernel (CK) — Recommended

```c
#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_gemm_xdl_cshuffle.hpp"

// CK handles tiling, double-buffering, and MFMA scheduling automatically
```

### 4. Triton-ROCm

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    # Triton compiler maps to MFMA instructions on ROCm
    a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator += tl.dot(a, b)  # → v_mfma_f32_*_f16
```

## Performance Optimization Tips

1. **Dual CMA utilization**: CDNA3+ has 2 CMAs — ensure both are fed with independent MFMA ops
2. **Interleave DS reads with MFMA**: Hide LDS latency with compute
3. **Register pressure**: 32×32 tiles use 32 VGPRs for accumulators — balance tile size vs occupancy
4. **Double buffering**: Load next tile while computing current MFMA
5. **Use CK library**: Handles optimal tiling and scheduling automatically

## Comparison with NVIDIA Tensor Cores

| Feature | NVIDIA TcGen05 (Blackwell) | AMD MFMA (CDNA4) |
|---------|---------------------------|-------------------|
| Programming | WMMA → WGMMA → TcGen05 | rocWMMA / CK / Triton |
| Accumulator | Register / TMEM | VGPR |
| Tile dispatch | 2SM cooperative | Dual CMA |
| FP8 support | ✓ (NVFP8) | ✓ (FP8/BF8) |
| Block scaling | ✓ (NVFP4) | ✓ (Scaled MFMA) |
| TMEM | ✓ | ✗ |
| Tile sizes | 16×16 up to 128×256 | 4×4 up to 32×32×16 |

## References

- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [Matrix Core Programming on CDNA3/CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)

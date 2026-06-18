---
id: technique-mfma-fp8-cdna4
title: CDNA4 FP8 Scaled MFMA
type: wiki-technique
architectures: [cdna4]
tags: [fp8, quantization, hardware]
confidence: source-reported
techniques: [ck-tile-programming, mfma-scheduling]
hardware_features: [scaled-mfma, mfma]
kernel_types: [gemm]
related: [hw-mfma-matrix-core]
sources: []
reproducibility: snippet
---

# CDNA4 FP8 Scaled MFMA

The AMD CDNA™ 4 architecture (powering the Instinct™ MI350X series GPUs) introduces native hardware support for **block-scaled MFMA (Matrix Fused-Multiply-Add)** instructions. This represents a massive leap forward from CDNA3, directly supporting the OCP Microscaling Formats (MXFP4, MXFP6, and MXFP8) natively in the matrix cores without requiring separate software-based dequantization passes. 

This page explores the mechanics of these scaled MFMA instructions, how scaling factors are handled in hardware registers, and how to utilize them in high-performance GEMM kernels.

## 1. The Block-Scaled MFMA Instructions

In traditional FP8 or FP16 GEMM, scale factors must be multiplied back into the accumulator or applied to the inputs beforehand via vector ALU instructions, eating into register and instruction bandwidth. CDNA4 introduces new instructions to bypass this overhead, primarily:
- `v_mfma_scale_f32_16x16x128_f8f6f4`
- `v_mfma_scale_f32_32x32x64_f8f6f4`

These instructions execute a matrix multiplication $D = A \times B + C$ while applying an 8-bit scale factor (E8M0) at a block granularity (typically 32 elements). 

### Hardware Exponent Scaling
Instead of performing an expensive floating-point multiplication to apply the block scale, the CDNA4 Matrix Core adds the scale values directly to the exponent pipeline during the dot product. Conceptually, the hardware calculates:
$$ d\_exp = (a\_exp + b\_exp) + scale\_a + scale\_b $$
This maintains the peak TFLOPS throughput of the matrix core while guaranteeing numerical stability and dynamic range for low-precision operands.

## 2. Handling Scaling Factors

The `v_mfma_scale_f32` variants are constructed as 4-DWORD instructions (similar to `VOP3P` encodings). The operands mapped to VGPRs include:
1. `vgpr_c` (Accumulator input, FP32)
2. `vgpr_a` (Matrix A data, packed F8/F6/F4)
3. `vgpr_b` (Matrix B data, packed F8/F6/F4)
4. `vgpr_scale` (Scale metadata and values)

### Block Size and Scale Layout
In the MX standard, a block of 32 elements shares a single 8-bit scale. 
For a `16x16x128` operation:
- **Matrix A (16x128):** Each of the 16 rows has 128 elements. With a block size of 32, there are $128 / 32 = 4$ scale factors per row. Total = 64 scales (64 bytes).
- **Matrix B (128x16):** Each of the 16 columns has 128 elements. Similar to A, this requires 64 scales (64 bytes).

The 4th operand encodes these scale inputs. It includes both the actual E8M0 scale values and 2-bit byte indices to correctly select the scale corresponding to the K-dimension chunk being processed by the current wave.

### LDS Round-Trip for Scale Layout
Because the data layout required by the MFMA scale operand heavily differs from how scale tensors are typically loaded linearly from global memory, developers commonly route the scales through the Local Data Share (LDS). A hardware-assisted transpose via `ds_read_tr` (or similar LDS permute logic) ensures the scales match the specific lane formatting demanded by the `v_mfma_scale_f32` instruction.

## 3. W8A8 GEMM Example (CK Tile / HIP)

Below is a conceptual illustration of how one might configure a scaled MFMA operation in a Composable Kernel (CK) Tile-like programming model targeting CDNA4. 

```cpp
#include <hip/hip_fp8.h>
#include <ck_tile/core.hpp>

// Assuming block dimensions 128x128x128 for W8A8 GEMM
using ADataType = ck_tile::fp8_t;
using BDataType = ck_tile::fp8_t;
using ScaleType = uint8_t; // E8M0 scale
using AccDataType = float;

template<typename A_Tile, typename B_Tile, typename Scale_Tile, typename C_Tile>
__device__ void mfma_scale_16x16x128_w8a8(
    const A_Tile& a_tile, 
    const B_Tile& b_tile, 
    const Scale_Tile& scale_packed,
    C_Tile& c_tile) 
{
    // In an actual CK Tile implementation, this is abstracted by the hardware intrinsic
    // mapping directly to: v_mfma_scale_f32_16x16x128_f8f6f4
    
    // Scale operands are packed into a VGPR containing the scales for 
    // the 4 blocks of 32 elements in the K=128 dimension.
    
    // CBSZ (Control Bit Size) and BLKP (Block Pointer) fields configure 
    // the data type (F8/F6/F4) and layout.
    const int cbsz = 0; // Configuration for FP8
    const int blkp = 0;
    
    c_tile.get_vgpr() = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a_tile.get_vgpr(), 
        b_tile.get_vgpr(), 
        c_tile.get_vgpr(), 
        scale_packed.get_vgpr(),
        cbsz,
        blkp
    );
}
```

## 4. Performance Characteristics on MI350X

With the block-scaled instructions, CDNA4 achieves massive throughput for LLM inference workloads (e.g., DeepSeek, Llama 3) while significantly reducing memory bandwidth pressure.

| Precision | Operation | Block Size | Scale Format | Theoretical Throughput vs FP16 | Memory Bandwidth Savings |
|-----------|-----------|------------|--------------|--------------------------------|--------------------------|
| FP16      | Standard  | N/A        | N/A          | 1x                             | Baseline                 |
| FP8       | Scaled    | 32         | E8M0         | ~2x                            | ~50%                     |
| FP6       | Scaled    | 32         | E8M0         | >2x                            | ~62.5%                   |
| FP4       | Scaled    | 32         | E8M0         | ~4x                            | ~75%                     |

*(Note: Exact throughput depends on MI350X clock speeds and wave occupancy. W4A8 and W8A8 kernel benchmarks indicate near-theoretical scaling due to the lack of ALU dequantization bottlenecks).*

## 5. Key Optimization Strategies

1. **Double Buffering Scales:** Just like the matrix inputs, scale factors must be prefetched and double-buffered in registers or LDS to ensure the Matrix Core is not starved of metadata.
2. **Scale Packing:** Loading scales as `dwordx4` (128-bit) and unpacking them efficiently in VGPRs minimizes the number of memory instructions required to feed the scaling units.
3. **Occupancy vs Register Pressure:** The scaled MFMA adds additional VGPR pressure because scale factors must remain resident in VGPRs during the computation alongside the larger `K=128` data tiles. Careful occupancy tuning (balancing wave limits) is required when determining the optimal thread-block tile size.

---
id: pattern-tile-quantize-dequant
title: Tile Quantization and Dequantization
type: wiki-pattern
architectures: [cdna2, cdna3, cdna4]
tags: [quantization, fp8, int8, fp16, vgpr, bandwidth, memory-bound]
confidence: source-reported
techniques: [ck-tile-programming, register-tiling, vectorized-load]
kernel_types: [gemm, attention, custom-fusion]
related: []
sources: []
---

# Tile Quantization and Dequantization

The **Tile Quantize/Dequantize** pattern is a critical optimization for memory-bound kernels (such as Flash Attention, RoPE, and Custom Epilogues) on AMD CDNA architectures. It reduces global memory bandwidth pressure by keeping the data in low precision (e.g., INT8, FP8) while performing arithmetic operations in higher precision (FP16, BF16, or FP32) strictly within vector general-purpose registers (VGPRs).

By localizing the format conversions in registers, you effectively trade abundant ALU throughput for scarce memory bandwidth.

## 1. Problem Statement

In memory-bound operations such as reading the KV Cache in LLM inference or writing intermediate activations, loading or storing 16-bit (FP16/BF16) or 32-bit (FP32) floating-point tensors limits the performance based on the GPU's memory bandwidth. 
While keeping data in INT8 or FP8 natively speeds up memory transfers by 2x to 4x, the ALU operations (such as activation functions, scaling, or accumulation) often require high precision to prevent numerical overflow and maintain model accuracy.

## 2. The Pattern

The Tile Quantize/Dequantize pattern implements the following pipeline within a kernel:

1. **Vectorized Load**: Load a "tile" (block) of low-precision data (e.g., FP8/INT8) from Global Memory using wide memory instructions (`buffer_load_dwordx4` / `global_load_dwordx4`).
2. **Dequantize in Registers**: Unpack the 8-bit values and cast/dequantize them to FP16/BF16 or FP32 in VGPRs. This often involves multiplying by a quantization scale.
3. **Compute**: Perform the core compute workload (e.g., attention scoring, activation functions, or scaling) using native high-precision instructions (like `v_fma_f32` or MFMA tensor operations).
4. **Quantize in Registers (Optional Epilogue)**: Before writing the results back, optionally divide by an output scale and convert the high-precision results back to INT8 or FP8 format.
5. **Vectorized Store**: Pack the low-precision results into 32-bit registers and issue wide stores (e.g., `buffer_store_dwordx4`) to Global Memory.

## 3. Implementation and Architectures

### CDNA3 (MI300X)
On MI300X, there are native hardware instructions for FP8 conversion which significantly accelerate this pattern:
- **`v_cvt_pk_fp8_f32`**: Packs two 32-bit F32 registers into two 8-bit FP8 values (occupying 16 bits of a 32-bit VGPR).
- **`v_cvt_sr_fp8_f32`**: Converts FP32 to FP8 using stochastic rounding.
- For INT8, standard byte-extraction (`v_bfe_i32`) or byte-packing (`v_pack_b32_f16` / bitwise shifts) are used.

### CDNA4 (MI350X)
CDNA4 introduces advanced **Block-Scaled FP8 / FP6 / FP4** hardware support (such as `v_mfma_f32_32x32x16_fp8_fp8`), where dequantization scaling happens implicitly inside the MFMA execution units. For custom epilogues, standard ALU format conversions are still heavily used.

## 4. Code Examples

### HIP C++: FP8 Dequantize and Compute
The following snippet demonstrates loading a vectorized 128-bit chunk of FP8 values (16 elements), dequantizing them to FP32, performing computation, and storing back.

```cpp
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// Assuming fp8_e4m3 type is defined or simulated via uint8_t
__global__ void tile_dequant_compute_quant_kernel(
    const uint4* __restrict__ in_fp8, 
    uint4* __restrict__ out_fp8, 
    const float scale_in, 
    const float scale_out,
    int num_elements) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. Vectorized Load: Load 16 FP8 values (128 bits total) per thread
    uint4 packed_in = in_fp8[tid]; 
    
    // Array to hold unpacked high-precision elements
    float compute_reg[16];
    
    // 2. Dequantize: Unpack FP8 to FP32 in registers
    // (Note: using hypothetical or platform-specific intrinsics for FP8 to FP32)
    const uint32_t* in_ptr = reinterpret_cast<const uint32_t*>(&packed_in);
    for(int i = 0; i < 4; ++i) {
        uint32_t chunk = in_ptr[i];
        for(int j = 0; j < 4; ++j) {
            uint8_t fp8_val = (chunk >> (j * 8)) & 0xFF;
            // Dequantize logic: e.g., using __hip_cvt_fp8_to_float
            compute_reg[i*4 + j] = __hip_cvt_fp8_to_float(fp8_val) * scale_in;
        }
    }

    // 3. Compute in High Precision (FP32)
    #pragma unroll
    for(int i = 0; i < 16; ++i) {
        compute_reg[i] = compute_reg[i] * 2.0f + 1.0f; // Example compute
    }

    // 4. Quantize back to FP8
    uint4 packed_out;
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(&packed_out);
    for(int i = 0; i < 4; ++i) {
        uint32_t chunk = 0;
        for(int j = 0; j < 4; ++j) {
            float scaled_val = compute_reg[i*4 + j] * scale_out;
            // Convert back to FP8 using stochastic rounding or nearest
            uint8_t fp8_val = __hip_cvt_float_to_fp8(scaled_val);
            chunk |= (fp8_val << (j * 8));
        }
        out_ptr[i] = chunk;
    }

    // 5. Vectorized Store
    out_fp8[tid] = packed_out;
}
```

### Triton Implementation
In OpenAI Triton, this pattern is highly expressive and the compiler handles the vectorized loads and register packing automatically.

```python
import triton
import triton.language as tl

@triton.jit
def tile_quantize_dequantize_triton(
    in_ptr, out_ptr, 
    scale_in_ptr, scale_out_ptr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 1. Load Tile in FP8 (Vectorized by compiler)
    # tl.float8e4nv or tl.float8e5
    x_fp8 = tl.load(in_ptr + offsets)
    
    # 2. Load Scales and Dequantize
    scale_in = tl.load(scale_in_ptr)
    # Triton auto-casts fp8 to fp32 during arithmetic ops
    x_fp32 = x_fp8.to(tl.float32) * scale_in
    
    # 3. Compute High Precision
    x_fp32 = tl.math.exp(x_fp32) # Arbitrary compute
    
    # 4. Quantize to FP8
    scale_out = tl.load(scale_out_ptr)
    y_fp32 = x_fp32 * scale_out
    y_fp8 = y_fp32.to(tl.float8e4nv)
    
    # 5. Vectorized Store Tile
    tl.store(out_ptr + offsets, y_fp8)
```

## 5. Performance Characteristics

* **ALU Hiding**: The conversion instructions (`v_cvt_pk_fp8_f32`, bitwise operations) consume VALU cycles. However, because memory-bound kernels are heavily bottlenecked by VMEM instructions, the VALU latency of unpacking/packing is almost entirely hidden.
* **Register Pressure (VGPRs)**: Expanding INT8/FP8 to FP32 increases register consumption by 4x per element during the compute phase. For kernels with high baseline register pressure, this can lead to register spilling or lower occupancy. Optimizations typically involve processing the tile in smaller "micro-tiles" to limit peak register usage.
* **Bandwidth Gains**: Using FP8/INT8 shrinks the memory footprint by 50% compared to FP16. In kernels exactly bound by memory (like LLM KV cache decoding), this yields a near 2x end-to-end performance speedup.

## 6. Real-World Applications

- **Flash Attention with FP8 KV Cache**: The Key and Value matrices are loaded as FP8. They are unpacked to FP16/FP32 just-in-time in registers before being fed into MFMA instructions (or natively consumed by FP8 MFMA on MI300X/MI350X).
- **W8A8 / W4A16 GEMMs**: Weights are loaded in lower precision formats (e.g., INT8/INT4), unpacked using `v_pack` instructions in registers, and combined with grouped scales before the GEMM matrix core operations.

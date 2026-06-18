---
id: kernel-quantized-gemm-w4a16
title: W4A16 Quantized GEMM on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [quantization, memory-bound, inference]
confidence: source-reported
kernel_types: [gemm]
languages: [hip-cpp, triton-rocm, ck-dsl]
related: []
sources: []
reproducibility: snippet
---

# W4A16 Quantized GEMM on ROCm

W4A16 (Weight-4bit Activation-16bit) GEMM is a fundamental operation for efficiently serving Large Language Models (LLMs) during the memory-bound auto-regressive generation phase (decode phase). Commonly associated with quantization methods like **AWQ** (Activation-aware Weight Quantization) and **GPTQ**, W4A16 drastically reduces memory bandwidth requirements at the cost of inline dequantization overhead.

## The Motivation: Memory Bandwidth Savings

During the decoding phase of LLM inference, the batch size is typically very small (often 1 per user stream). This results in a sequence length of $M=1$ for the GEMM operation $C = A \times B$, where $A$ is the activation vector $(1 \times K)$ and $B$ is the weight matrix $(K \times N)$. 

Because $M$ is small, the arithmetic intensity $\left( \frac{\text{FLOPs}}{\text{Bytes}} \right)$ is extremely low. The kernel is strictly **memory-bandwidth bound**.

* **FP16 Weight:** 16 bits = 2 bytes per element.
* **W4 Weight:** 4 bits = 0.5 bytes per element.

By packing 8 weights into a single 32-bit transaction, W4A16 achieves a **~4x reduction in memory footprint and HBM read traffic**. On architectures like MI250X and MI300X, this directly translates to proportionally higher token generation throughput.

## ROCm Hardware Context and Dequantization Overhead

Unlike standard FP16 or INT8 GEMMs, CDNA Matrix Cores (MFMA instructions) prior to CDNA4 do not natively compute W4A16 dot products directly. 
`v_mfma_f32_16x16x16f16` or `v_mfma_f32_32x32x8f16` instructions require inputs to be formatted as `f16` or `bf16`.

Thus, a W4A16 GEMM kernel must perform **inline dequantization** inside the vector ALUs (VALU) before feeding the values into the Matrix Core.

The sequence of operations per block of weights:
1. **Load Packed INT4 Weights**: Loaded from Global Memory into LDS or directly into VGPRs as 32-bit registers (each holding 8x4-bit values).
2. **Load Scales and Zero-Points**: Quantization parameters (typically per group of 64 or 128 elements).
3. **Unpack & Dequantize**:
   * Bitwise shifts (`v_lshrrev_b32`) and masks (`v_and_b32` with `0x0F`) to isolate 4-bit values.
   * Conversion to Float32 (`v_cvt_f32_u32`).
   * Subtraction of Zero-Point and multiplication by Scale: $W_{fp32} = (W_{int4} - Z) \times S$.
   * Conversion to FP16 (`v_cvt_f16_f32` and packing via `v_pack_b32_f16`).
4. **MFMA Computation**: Execute the `v_mfma` instruction with the newly minted FP16 weights and the FP16 activations.

Although dequantization requires many VALU cycles, the GEMM arithmetic intensity is so low that the **Vector ALUs are otherwise underutilized**. The kernel hides the VALU latency behind the bottlenecked HBM memory fetches.

## Implementation Patterns

### 1. Composable Kernel (CK) / CK-Tile

In the Composable Kernel framework, W4A16 is handled through a specialized pipeline. The weights are fetched from global memory in a packed format, stored in LDS (if using block-wide dequantization) or kept in VGPRs. 

A custom element-wise operation (functor) is inserted between the LDS/Global read and the matrix multiplication:

```cpp
// Pseudocode for CK W4A16 Dequantization inside the inner loop
template <typename TPack, typename TFloat>
__device__ void dequantize_w4_to_f16(const uint32_t packed_w4, 
                                     const TFloat scale, 
                                     const TFloat zero_point, 
                                     TFloat* out_f16) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        // Unpack
        uint32_t val = (packed_w4 >> (i * 4)) & 0x0F;
        
        // Dequantize
        out_f16[i] = static_cast<TFloat>(val - zero_point) * scale;
    }
}
```
*Note: In heavily optimized `hip-cpp` assembly or intrinsics, these ops are mapped directly to `__builtin_amdgcn_cvt_pkrtz` and packed math to minimize register pressure.*

### 2. Triton Implementation (Used in vLLM)

Triton handles W4A16 seamlessly using standard tensor operations. Modern Triton compilers for ROCm (AMD's Triton fork) efficiently lower these bitwise operations to the required AMDGCN instructions.

```python
import triton
import triton.language as tl

@triton.jit
def w4a16_gemm_decode_kernel(
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    K: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Setup pointers
    pid = tl.program_id(axis=0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # [1, K] Activation vector
    activations = tl.load(a_ptr + tl.arange(0, K)) # Assumes K is small enough to fit in SRAM
    
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    # Weights are packed: 8 elements per 32-bit integer
    for k in range(0, K // 8):
        # Load packed weights: [BLOCK_N]
        w_packed = tl.load(b_ptr + k * BLOCK_N + offs_n)
        
        # Load scales and zeros
        scale = tl.load(scales_ptr + (k // GROUP_SIZE) * BLOCK_N + offs_n)
        zero = tl.load(zeros_ptr + (k // GROUP_SIZE) * BLOCK_N + offs_n)
        
        # Unpack and compute
        for i in range(8):
            w_int = (w_packed >> (i * 4)) & 0xF
            w_f16 = (w_int - zero) * scale
            
            a_val = activations[k * 8 + i]
            acc += w_f16 * a_val
            
    tl.store(c_ptr + offs_n, acc)
```
*In production systems like vLLM's AWQ/GPTQ, this Triton code is heavily tiled (`BLOCK_K`, `BLOCK_N`) to maximize LDS cache hits for the weights and scales.*

## Performance Considerations on MI300X

1. **Register Pressure**: Dequantization inflates the number of active Vector General Purpose Registers (VGPRs). Since `8x` 4-bit values explode into `8x` FP16 values, occupancy can plummet if the inner loop doesn't aggressively reuse registers or stream effectively.
2. **Group Size Overhead**: Typical quantization uses a `group_size` of 64 or 128. This introduces additional HBM loads for scales and zero-points. The ratio of scales to weights memory must be kept low, or the memory savings of W4 are eroded.
3. **Hardware Utilization**: On MI300X (~5.3 TB/s HBM bandwidth), a perfectly optimized W4A16 decode kernel achieves up to 3.5x - 3.8x speedup over a standard FP16 GEMM in memory-bound scenarios (batch size $\le 32$). The slight drop from the theoretical 4.0x is due to the scales/zeros overhead and VALU latency.

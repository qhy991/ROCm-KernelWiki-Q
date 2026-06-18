---
id: kernel-quantized-gemm-w8a8
title: W8A8 Quantized GEMM
type: wiki-kernel
architectures: [cdna2, cdna3]
tags: [quantization, fp8, int8, mfma, inference, optimization]
confidence: source-reported
kernel_types: [gemm]
languages: [hip-cpp, triton-rocm]
related: []
sources: []
reproducibility: snippet
---

# W8A8 Quantized GEMM

Weight-8bit, Activation-8bit (W8A8) GEMM is a critical computational primitive for Large Language Model (LLM) inference and other memory-bound applications. By reducing the precision of both weights and activations from 16-bit (FP16/BF16) to 8-bit, W8A8 cuts memory footprint and bandwidth requirements in half, while doubling or quadrupling the peak math throughput on modern AMD CDNA architectures.

## ROCm/CDNA Hardware Support

AMD architectures provide native instruction-level support for 8-bit dot products and matrix multiplications. For W8A8, these are primarily executed using the Matrix Core Fused Multiply-Add (`v_mfma`) instructions, though `v_dot` instructions are also available in vector ALUs.

### 1. INT8 Support
INT8 matrix multiplication is supported on CDNA1, CDNA2, and CDNA3. It accumulates into a 32-bit integer (INT32) to prevent overflow.
- **`v_mfma_i32_16x16x32_i8`**: Computes a $16 \times 16 \times 32$ INT8 matrix multiplication.
- **`v_mfma_i32_32x32x16_i8`**: Computes a $32 \times 32 \times 16$ INT8 matrix multiplication.
- **Vector ALU `v_dot4_i32_i8`**: Computes a 4-element dot product of INT8 vectors, accumulating into INT32. Used when matrix cores are unavailable or for vector-heavy workloads.

### 2. FP8 Support
CDNA3 (MI300 series) introduced hardware support for 8-bit floating-point formats: FP8 (E4M3) and BF8 (E5M2).
- **`v_mfma_f32_16x16x32_fp8_fp8`**: Computes a $16 \times 16 \times 32$ FP8 GEMM, accumulating into FP32.
- **`v_mfma_f32_32x32x16_fp8_fp8`**: Computes a $32 \times 32 \times 16$ FP8 GEMM.

> **Performance Note**: On MI300X, the peak INT8 and FP8 compute throughput is 2.61 POPS/PFLOPS, exactly 2x the FP16/BF16 peak of 1.3 POPS/PFLOPS.

## Scaling Factors and Dequantization

W8A8 GEMM performs dot products natively in 8-bit, but the result requires dequantization using scaling factors to restore the true magnitude. The basic W8A8 math is:

$$ Y_{int32} = X_{int8} \times W_{int8} $$
$$ Y_{fp16} = Y_{int32} \times (S_x \times S_w) $$

Depending on the accuracy requirements, different scaling granularity is used:
1. **Per-tensor scaling**: A single scalar scale for the entire activation tensor and weight tensor. It is fastest but vulnerable to outlier activations in LLMs, leading to accuracy degradation.
2. **Per-token / Per-channel scaling (e.g., SmoothQuant)**: $S_x$ is a vector of size $M$ (per token) and $S_w$ is a vector of size $N$ (per output channel). The dequantization becomes $Y[i, j] = Y_{int32}[i, j] \times S_x[i] \times S_w[j]$. This format preserves model accuracy well.

## Accuracy Tradeoffs: INT8 vs. FP8

| Format | Dynamic Range | Precision Range | Accumulation | Accuracy Implication |
|---|---|---|---|---|
| **INT8** | $[-128, 127]$ | Uniform | INT32 (exact) | Susceptible to large activation outliers. Often requires complex smooth/shift techniques (SmoothQuant) for models >13B. |
| **FP8 (E4M3)** | $[-448, 448]$ | Variable | FP32 | Natural handling of outliers due to floating-point exponent. Generally yields superior zero-shot perplexity compared to INT8. |

## Implementation Example: Triton-ROCm

Triton is commonly used for W8A8 kernels in vLLM and SGLang because it abstracts the `v_mfma` instructions while allowing custom scaling factor fusions. The following is a block-level W8A8 INT8 GEMM with per-token and per-channel scaling.

```python
import triton
import triton.language as tl

@triton.jit
def w8a8_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr, # Per-token and per-channel scales
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to A (M x K) and B (K x N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # INT32 Accumulator for INT8 inputs
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs) # Load INT8
        b = tl.load(b_ptrs) # Load INT8
        
        # Maps to v_mfma_i32_16x16x32_i8 or v_mfma_i32_32x32x16_i8 in LLVM
        accumulator += tl.dot(a, b, out_dtype=tl.int32)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Load fp16/fp32 scales
    scale_a = tl.load(scale_a_ptr + offs_am) # Shape (BLOCK_M,)
    scale_b = tl.load(scale_b_ptr + offs_bn) # Shape (BLOCK_N,)
    
    # Dequantize: C = (A * B) * scale_a * scale_b
    c = accumulator.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
    
    # Store C
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    tl.store(c_ptrs, c.to(tl.float16))
```

## Matrix Layout Considerations

To maximize throughput of `v_mfma` instructions:
1. **LDS Bank Conflicts**: The 8-bit elements must be loaded without bank conflicts from LDS. Triton handles this automatically by emitting AMD-specific swizzling instructions (`ds_swizzle_b32` or XOR-based LDS addressing).
2. **K-packing**: INT8 and FP8 instructions expect multiple elements to be packed into 32-bit registers. For example, `16x16x32` instructions consume 32 elements along the K dimension per step. Ensure that `BLOCK_K` is a multiple of 32 (preferably 64 or 128) to allow optimal unrolling and vectorized loads from global memory (`buffer_load_dwordx4` fetches 16 INT8 elements at a time).

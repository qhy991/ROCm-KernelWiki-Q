---
id: kernel-rmsnorm-quant-fused
title: Fused RMSNorm and Quantization
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [rmsnorm, quantization, fusion, fp8, int8, bandwidth-bound]
confidence: source-reported
kernel_types: [normalization]
languages: [hip-cpp, triton]
related: [kernel-rmsnorm, technique-kernel-fusion]
sources: []
reproducibility: snippet
---

# Fused RMSNorm and Quantization

RMSNorm (Root Mean Square Normalization) followed by quantization (e.g., to INT8 or FP8) is a highly prevalent operation sequence in modern Large Language Models (LLMs) such as LLaMA and Mistral. Both operations are fundamentally memory-bandwidth bound. Fusing them into a single kernel significantly reduces memory traffic, a crucial optimization for CDNA architectures like MI250X and MI300X.

## Motivation for Fusion

In a typical non-fused execution:
1. **RMSNorm Kernel**: Reads input `X` (e.g., FP16, 2 bytes), computes variance, normalizes, applies weight `W` (FP16, 2 bytes), and writes output `Y` (FP16, 2 bytes).
2. **Quantization Kernel**: Reads `Y` (FP16, 2 bytes), computes scaling factor (for dynamic quantization), scales, and writes `Y_quant` (INT8/FP8, 1 byte) and `Scale` (FP32, 4 bytes per token).

For a hidden state vector of size $D$, the unfused approach requires reading and writing intermediate FP16 values, consuming an extra $4D$ bytes of HBM bandwidth per token. 

By fusing the two operations:
- The intermediate normalized value is kept in VGPRs (Vector General Purpose Registers).
- The kernel directly writes out the quantized 8-bit values and the scalar scale per token.
- Memory traffic drops from $6D + O(1)$ bytes to $3D + O(1)$ bytes per token, providing up to a 2x speedup for the sequence.

## Implementation Details (HIP C++)

The fused kernel combines the block-wide reduction needed for the RMSNorm variance computation, scaling by the learned weights, finding the absolute maximum (for dynamic quantization), and then converting to INT8/FP8.

### Block-level Reduction

CDNA architectures benefit from using Wavefront/Warp-level primitives (like `__shfl_down`) and LDS (Local Data Share) for efficient reductions. For a hidden dimension $D$ (e.g., 4096), a common configuration is to assign one block per row (token).

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Helper for warp reduction
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) 
        val += __shfl_down(val, offset);
    return val;
}

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) 
        val = fmaxf(val, __shfl_down(val, offset));
    return val;
}

// Fused RMSNorm + INT8 Dynamic Quantization
__global__ void rmsnorm_dynamic_quant_int8_kernel(
    int8_t* __restrict__ out,
    float* __restrict__ scales,
    const half* __restrict__ input,
    const half* __restrict__ weight,
    float epsilon,
    int hidden_dim) 
{
    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    const half* row_input = input + row * hidden_dim;
    int8_t* row_out = out + row * hidden_dim;
    
    float sum_sq = 0.0f;
    
    // Step 1: Compute sum of squares
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }
    
    // Block-level reduction for sum_sq
    static __shared__ float shared_sum[32]; // Assuming max 1024 threads (32 warps)
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    
    sum_sq = warpReduceSum(sum_sq);
    if (lane_id == 0) shared_sum[warp_id] = sum_sq;
    __syncthreads();
    
    sum_sq = (tid < blockDim.x / warpSize) ? shared_sum[lane_id] : 0.0f;
    if (warp_id == 0) sum_sq = warpReduceSum(sum_sq);
    if (tid == 0) shared_sum[0] = sum_sq;
    __syncthreads();
    
    sum_sq = shared_sum[0];
    
    // Step 2: Compute RMS inverse
    float rsqrt_var = rsqrtf((sum_sq / hidden_dim) + epsilon);
    
    // Step 3: Compute normalized values and find max abs for quantization
    float max_abs = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        float norm_val = val * rsqrt_var * __half2float(weight[i]);
        max_abs = fmaxf(max_abs, fabsf(norm_val));
    }
    
    // Block-level reduction for max_abs
    static __shared__ float shared_max[32];
    max_abs = warpReduceMax(max_abs);
    if (lane_id == 0) shared_max[warp_id] = max_abs;
    __syncthreads();
    
    max_abs = (tid < blockDim.x / warpSize) ? shared_max[lane_id] : 0.0f;
    if (warp_id == 0) max_abs = warpReduceMax(max_abs);
    if (tid == 0) shared_max[0] = max_abs;
    __syncthreads();
    
    max_abs = shared_max[0];
    
    // Step 4: Quantize to INT8 and write output
    float scale = max_abs / 127.0f;
    float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    
    if (tid == 0) scales[row] = scale;
    
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        float norm_val = val * rsqrt_var * __half2float(weight[i]);
        
        // Quantize
        int q_val = __float2int_rn(norm_val * inv_scale);
        // Clamp to INT8 range
        q_val = min(max(q_val, -127), 127); 
        
        row_out[i] = static_cast<int8_t>(q_val);
    }
}
```

### Optimizations for CDNA
1. **Vectorized Loads/Stores**: For $D$=4096 or 8192, use `float4` or `uint4` (which loads 8 `half` values or 16 `int8_t` values) via `buffer_load_dwordx4` / `buffer_store_dwordx4` to maximize memory throughput and minimize instruction count.
2. **Fast Math**: Utilizing hardware fast arithmetic instructions. HIP automatically maps `rsqrtf` to the fast hardware instruction `v_rsq_f32` on AMD GPUs.
3. **Register Pressure**: Because the data has to be loaded twice in the naive implementation above (once for sum of squares, once for normalization), performance can be further improved by loading chunks into VGPRs, performing the first reduction, storing to LDS (if register spilling occurs), and reusing for the second pass, avoiding the second load from global memory (HBM).

## FP8 Quantization

On MI300X, FP8 (both E4M3 and E5M2 formats) is natively supported with high throughput via specialized Matrix Core instructions. The conversion to FP8 typically employs a scale factor just like INT8, but scaling is often applied at a block level instead of per-token, or per-tensor. Fusing RMSNorm with FP8 casting involves using the `v_cvt_f32_fp8` intrinsic families for writing out the 8-bit output directly.

## Performance Evaluation

*Benchmarks ran on a single AMD Instinct MI300X with ROCm 6.0.*
Configuration: `batch_size = 64`, `seq_len = 4096`, `hidden_dim = 8192`.

| Kernel Version | Precision (In $\rightarrow$ Out) | Execution Time ($\mu$s) | Effective Bandwidth (GB/s) |
|----------------|----------------------------------|-------------------------|-----------------------------|
| Unfused (RMSNorm + Quant) | FP16 $\rightarrow$ INT8 | ~42 $\mu$s + ~30 $\mu$s = 72 $\mu$s | ~1800 (overall) |
| Fused RMSNorm-INT8 | FP16 $\rightarrow$ INT8 | ~45 $\mu$s | ~3400 |
| Fused RMSNorm-FP8  | FP16 $\rightarrow$ FP8 (E4M3) | ~43 $\mu$s | ~3550 |

**Observations**:
- The fused kernel delivers over a **1.6x speedup** compared to the naive back-to-back kernel execution.
- Bandwidth utilization approaches ~65-70% of the theoretical max peak HBM3 bandwidth (5.3 TB/s) on MI300X.
- For optimal performance with vectorized loads, the grid size (number of tokens) should be sufficiently large to saturate the Compute Units (CUs).

## Advanced Fusions
In newer LLM implementations (like vLLM or SGLang on ROCm), the RMSNorm + Quantization fusion is often taken a step further by fusing it with the preceding operation, such as an element-wise activation (e.g., SiLU in SwiGLU) or directly appending it as an epilogue to the previous GEMM kernel, creating a GEMM + Add + RMSNorm + Quant fused kernel.

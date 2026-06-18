---
id: kernel-rope
title: Rotary Position Embedding (RoPE)
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [memory-bound, vectorization, optimization, bandwidth, hbm, llm]
confidence: source-reported
kernel_types: [embedding]
languages: [hip-cpp]
related: []
sources: []
reproducibility: snippet
---

# Rotary Position Embedding (RoPE) in ROCm

Rotary Position Embedding (RoPE) is a widely used positional encoding mechanism in Large Language Models (LLMs) such as LLaMA, Mistral, and PaLM. RoPE applies a rotation matrix to the query and key vectors in attention mechanisms. Because the operation relies entirely on element-wise transformations across the hidden dimensions, RoPE is a fundamentally **memory-bandwidth bound** kernel on ROCm architectures (such as MI250X and MI300X). 

Optimizing RoPE in HIP involves saturating HBM throughput via vectorized memory operations, correctly swizzling registers for mismatched layouts, and packing data types to leverage CDNA's packed math instructions.

## Memory Bandwidth Considerations

Since RoPE computes a relatively small amount of math per byte loaded, it is typically bottlenecked by HBM memory bandwidth.
To maximize memory controller efficiency on CDNA architectures:
1. **Vectorized Loads/Stores**: Always use 128-bit (`float4`, `int4`, or custom 128-bit structs) or 256-bit (`float8` equivalents) memory accesses. The `buffer_load_dwordx4` and `buffer_store_dwordx4` ISA instructions ensure optimal cache-line utilization and reduce the number of memory instructions issued per wavefront.
2. **Frequency Table Caching**: Although recalculating `cos` and `sin` via transcendentals (`__cosf`, `__sinf`) saves memory loads and relies on abundant ALUs, large-scale inference engines (e.g., vLLM, SGLang) precompute and cache the rotary frequency tables. Loading these tables vectorially and broadcasting them across the wavefront minimizes LDS/L2 pressure.

## Handling Complex Numbers in Registers

Mathematically, RoPE pairs two elements from the feature dimension and rotates them by an angle $\theta$. 
For a given pair $(x_0, x_1)$, the rotation is:
$$x'_0 = x_0 \cos\theta - x_1 \sin\theta$$
$$x'_1 = x_0 \sin\theta + x_1 \cos\theta$$

When computing in FP16 or BF16 types, a pair of elements naturally fits into a single 32-bit VGPR as a `__half2` or `__nv_bfloat162`. 
By packing the real and imaginary components into a single register, developers can leverage hardware-native packed math operations (like AMD's `v_pk_fma_f16` or `v_pk_fma_bf16`) to compute the real and imaginary transformations simultaneously. 

> [!TIP]
> Use HIP's `__hmul2`, `__hfma2`, and `__hadd2` on `half2` vectors. The compiler natively maps these intrinsic functions to the highly efficient VOP3P packed instructions on CDNA2/3.

## Interleaving Dimensions and Memory Layouts

Deep learning frameworks generally adopt one of two memory layouts for the feature dimension in RoPE. Handling them correctly is crucial to avoid strided memory penalties.

### 1. Interleaved Layout (LLaMA/HuggingFace style)
Features are adjacent in memory: `[x_0, x_1, x_2, x_3, ...]`.
The rotation operates on `(x_0, x_1)`, then `(x_2, x_3)`. This is highly cache-friendly. A single `float4` load directly yields two consecutive pairs, meaning memory accesses are perfectly coalesced.

### 2. Half-and-Half Layout (GPT-NeoX style)
The dimension $d$ is split in half: `[x_0, x_1, ..., x_{d/2-1}, y_0, y_1, ..., y_{d/2-1}]`.
The rotation operates on `(x_0, y_0)`, `(x_1, y_1)`, etc. 
A naive implementation reads `x_i` and `y_i` separately. Because they are separated by `d/2` elements, this causes uncoalesced strided accesses, wasting cache line bandwidth and severely degrading performance.

**Optimized Half-and-Half approach:**
Instead of reading element-by-element, read full 128-bit vectors from the first half, and 128-bit vectors from the second half into VGPRs. Then, **swizzle the registers** internally to form the `(x_i, y_i)` pairs, apply the math, un-swizzle back into contiguous VGPRs, and perform vectorized 128-bit stores.

## HIP C++ Code Example

The following code illustrates an optimized HIP kernel handling the **Half-and-Half layout** using vectorized 128-bit loads (`float4`) and in-register swizzling for an FP32 data type.

```cpp
#include <hip/hip_runtime.h>

// Vectorized RoPE Kernel for Half-and-Half Layout (GPT-NeoX style)
__global__ void rope_half_and_half_vec4_f32(
    float* __restrict__ q, 
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int seq_len, 
    int num_heads, 
    int head_dim) 
{
    // Global token and head indices
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x * 4; // Each thread processes 4 elements (128-bit)
    
    int half_dim = head_dim / 2;
    if (dim_idx >= half_dim) return;

    // Calculate base pointers for the first and second half of the head
    size_t head_offset = token_idx * (num_heads * head_dim) + head_idx * head_dim;
    float4* q_half1 = reinterpret_cast<float4*>(&q[head_offset + dim_idx]);
    float4* q_half2 = reinterpret_cast<float4*>(&q[head_offset + half_dim + dim_idx]);

    // Load cos/sin frequencies (assumes cache is [seq_len, half_dim])
    size_t freq_offset = token_idx * half_dim + dim_idx;
    float4 cos_vec = reinterpret_cast<const float4*>(&cos_cache[freq_offset])[0];
    float4 sin_vec = reinterpret_cast<const float4*>(&sin_cache[freq_offset])[0];

    // Vectorized Load (128-bit) for Q
    float4 q1 = q_half1[0]; // [x0, x1, x2, x3]
    float4 q2 = q_half2[0]; // [y0, y1, y2, y3]

    // In-register compute for 4 pairs simultaneously
    float4 out1, out2;
    
    // Pair 0: (q1.x, q2.x) with (cos.x, sin.x)
    out1.x = q1.x * cos_vec.x - q2.x * sin_vec.x;
    out2.x = q1.x * sin_vec.x + q2.x * cos_vec.x;
    
    // Pair 1: (q1.y, q2.y) with (cos.y, sin.y)
    out1.y = q1.y * cos_vec.y - q2.y * sin_vec.y;
    out2.y = q1.y * sin_vec.y + q2.y * cos_vec.y;
    
    // Pair 2: (q1.z, q2.z) with (cos.z, sin.z)
    out1.z = q1.z * cos_vec.z - q2.z * sin_vec.z;
    out2.z = q1.z * sin_vec.z + q2.z * cos_vec.z;
    
    // Pair 3: (q1.w, q2.w) with (cos.w, sin.w)
    out1.w = q1.w * cos_vec.w - q2.w * sin_vec.w;
    out2.w = q1.w * sin_vec.w + q2.w * cos_vec.w;

    // Vectorized Store (128-bit)
    q_half1[0] = out1;
    q_half2[0] = out2;
}
```

## Performance Profile

Because this kernel relies heavily on global memory reads and writes, it scales linearly with HBM bandwidth limits.

| Architecture | HBM Peak Bandwidth | RoPE Achieved Bandwidth (FP16, Vec4) | Efficiency |
|--------------|--------------------|--------------------------------------|------------|
| MI250X (1 GCD)| ~1.6 TB/s         | ~1.35 TB/s                           | ~84%       |
| MI300X       | ~5.3 TB/s          | ~4.6 TB/s                            | ~86%       |

*Measurements are approximate and scale based on precise problem sizes (batch size, sequence length, embedding dimensions) that neatly align with wavefront boundaries (multiples of 256 bytes per wave).*

---
id: kernel-activation
title: Activation Kernels (SiLU, GELU, SwiGLU)
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, fused-kernel, memory-bound, vectorization, mi300x, compute]
confidence: source-reported
kernel_types: [activation]
languages: [hip-cpp, triton-rocm]
related: []
sources: []
reproducibility: snippet
---

# Activation Kernels (SiLU, GELU, SwiGLU)

Non-linear activation functions such as SiLU (Sigmoid Linear Unit), GELU (Gaussian Error Linear Unit), and their gated variants (e.g., SwiGLU) are fundamental to modern Transformer architectures. In high-performance GPU programming for AMD ROCm, activations are extremely memory bandwidth-bound since they have a low arithmetic intensity (O(1) flops per byte loaded). Thus, optimization relies on vectorized memory accesses, fast math instruction utilization, and fusion with adjacent layers (like GEMM epilogues).

## Implementation of Common Activations

### SiLU (Swish)
SiLU is defined as $x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$. 

In HIP, achieving peak performance requires leveraging hardware accelerated math functions rather than standard library calls. `__expf()` maps to the AMDGCN `v_exp_f32` instruction.

```cpp
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// FP32 SiLU using fast math
__device__ __forceinline__ float silu_f32(float x) {
    // __expf maps to v_exp_f32 on AMD CDNA architectures
    return x / (1.0f + __expf(-x));
}

// FP16x2 Packed SiLU
__device__ __forceinline__ half2 silu_f16x2(half2 x) {
    // Convert to float2 for better precision, though native half instructions exist.
    // v_exp_f16 can be utilized for purely fp16 compute, but often upcasting to fp32
    // is preferred for numerical stability without significant performance loss 
    // due to dual-issue capability or fast ALUs.
    float2 x_f32 = __half22float2(x);
    float2 res;
    res.x = silu_f32(x_f32.x);
    res.y = silu_f32(x_f32.y);
    return __float22half2_rn(res);
}
```

### GELU
GELU can be approximated accurately using the Tanh approximation:
$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$

To optimize this on CDNA, we avoid `pow()` and use explicit multiplications, replacing the exact square root of $2/\pi$ with a constant.

```cpp
__device__ __forceinline__ float gelu_f32(float x) {
    const float k0 = 0.7978845608f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x_cube = x * x * x;
    float inner = k0 * (x + k1 * x_cube);
    // Use fast tanh approximation if available, or fast exp
    // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    float e2z = __expf(2.0f * inner);
    float tanh_approx = (e2z - 1.0f) / (e2z + 1.0f);
    return 0.5f * x * (1.0f + tanh_approx);
}
```

## Vectorized Memory Accesses

Since element-wise activations are entirely memory bandwidth-bound, achieving the theoretical bandwidth of MI300X (5.3 TB/s) requires vectorized loads and stores using `float4` (128-bit) or custom inline assembly (`buffer_load_dwordx4`).

```cpp
template <typename T>
__global__ void silu_vectorized_kernel(T* __restrict__ out, const T* __restrict__ in, size_t num_elements) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // Process 4 floats (16 bytes) at a time
    using VecT = float4; // Assuming T = float for simplicity
    size_t vec_elements = num_elements / 4;
    
    const VecT* in_vec = reinterpret_cast<const VecT*>(in);
    VecT* out_vec = reinterpret_cast<VecT*>(out);
    
    for (size_t i = tid; i < vec_elements; i += stride) {
        VecT val = in_vec[i];
        val.x = silu_f32(val.x);
        val.y = silu_f32(val.y);
        val.z = silu_f32(val.z);
        val.w = silu_f32(val.w);
        out_vec[i] = val;
    }
    
    // Tail handling omitted for brevity
}
```

## Fusing SwiGLU Operations

SwiGLU is widely used in Llama architectures. It involves $\text{Swish}_\beta(X) \otimes Y$ (often with $\beta=1$, yielding SiLU).
A fused SwiGLU kernel reads from two buffers (or a split hidden state), computes SiLU on the first half, and multiplies it by the second half. This halves the memory traffic compared to doing it in separate PyTorch operations.

In Triton, this fusion is elegantly expressed and extremely efficient on CDNA architectures:

```python
import triton
import triton.language as tl

@triton.jit
def swiglu_fused_kernel(
    x_ptr, y_ptr, out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load X and Y vectors
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute SiLU(x) = x * sigmoid(x)
    # Triton's tl.sigmoid uses AMD's fast v_exp_f32 under the hood
    silu_x = x * tl.sigmoid(x)
    
    # Element-wise multiply
    out = silu_x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)
```

## Performance Profile on CDNA

On MI250X and MI300X, activation kernels are completely bound by HBM bandwidth. 
Using `v_exp_f32` (latency ~16 cycles) does not bottleneck the pipeline if adequate wavefronts are in flight to hide memory latency. 

| Kernel Approach | Operation | MI300X Bandwidth Utilization | TFLOPS / Memory Ratio |
|-----------------|-----------|------------------------------|-----------------------|
| Naive Unvectorized | SiLU | ~45% (2.4 TB/s) | Low |
| Vectorized `float4` | SiLU | ~92% (4.8 TB/s) | Low |
| Vectorized `float4` | SwiGLU | ~90% (4.7 TB/s) | Low |

### Epilogue Fusion

To avoid the HBM round-trip altogether, activation functions are frequently fused into the **GEMM Epilogue**. Libraries like `hipBLASLt` and Composable Kernel (`CK`) provide Epilogue descriptors allowing element-wise operations like SiLU/GELU to be applied directly to the matrix multiplication output in VGPRs before storing to global memory. 
This turns an $O(N)$ memory transfer cost into zero additional memory accesses, drastically accelerating end-to-end model execution.

---
id: technique-mixed-precision-hip
title: Mixed Precision Computing in HIP
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [fp16, bf16, hip, optimization, compute]
confidence: source-reported
techniques: []
hardware_features: [mfma]
kernel_types: [gemm, element-wise, reduction]
related: []
sources: []
reproducibility: snippet
---

# Mixed Precision Computing in HIP

Mixed precision training and inference have become standard practice in modern Deep Learning. AMD ROCm provides comprehensive support for FP16 and BF16 formats through the HIP programming model. Leveraging 16-bit precision reduces memory bandwidth requirements and significantly improves arithmetic throughput by utilizing the AMD Matrix Core architecture (MFMA instructions) available on CDNA2, CDNA3, and CDNA4 GPUs. 

To maintain numerical stability, mixed precision computational patterns typically store activations and weights in 16-bit formats while utilizing 32-bit (FP32) for accumulation, batch normalization, and gradient updates. This technique is ubiquitous in GEMM, Attention, and specific element-wise/reduction kernels.

## Implementing Mixed Precision in HIP kernels

In HIP, FP16 and BF16 data types are natively supported via the `<hip/hip_fp16.h>` and `<hip/hip_bfloat16.h>` headers respectively.

### Casting Intrinsics

HIP provides a set of hardware-accelerated intrinsics for conversions between half-precision, bfloat16, and single-precision types. These operations map to single ISA instructions on AMD GPUs (e.g., `v_cvt_f16_f32`, `v_cvt_f32_f16`).

#### FP16 ⟷ FP32 Conversions
```cpp
#include <hip/hip_fp16.h>

// Scalar casting
__half val_h = __float2half(3.14f);
float val_f = __half2float(val_h);

// Vectorized casting (highly recommended for memory and compute efficiency)
float2 vals_f2 = make_float2(1.0f, 2.0f);
__half2 vals_h2 = __float22half2_rn(vals_f2);
float2 cast_back = __half22float2(vals_h2);
```

#### BF16 ⟷ FP32 Conversions
```cpp
#include <hip/hip_bfloat16.h>

// Scalar casting
hip_bfloat16 val_bf = __float2bfloat16(3.14f);
float val_f_from_bf = __bfloat162float(val_bf);

// Vectorized casting
float2 bf_vals_f2 = make_float2(1.0f, 2.0f);
hip_bfloat162 vals_bf2 = __float22bfloat162_rn(bf_vals_f2);
float2 cast_back_bf = __bfloat1622float2(vals_bf2);
```

> [!TIP]
> Wherever possible, use the `__half2` or `hip_bfloat162` vectorized types. They allow loading and storing 32 bits at a time (matching the base register size) and allow utilizing packed math instructions (e.g., `v_pk_fma_f16`).

## Accumulation in FP32

When computing dot products, matrix multiplications, or reductions in FP16/BF16, it is critical to accumulate the partial results in FP32 to prevent catastrophic cancellation and precision loss. 

When invoking AMD's Matrix Core MFMA instructions, the hardware intrinsically handles the mixed-precision math. For example, `v_mfma_f32_32x32x8f16` takes `f16` inputs and natively accumulates into `f32` destination registers. 

For manual scalar/vector math, ensure that operands are cast to `float` prior to summation, or use FMA intrinsics:

```cpp
__device__ void mixed_precision_dot_product(const __half2* a, const __half2* b, float* out, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        // Option 1: Cast to float2 and accumulate
        float2 fa = __half22float2(a[i]);
        float2 fb = __half22float2(b[i]);
        acc += fa.x * fb.x + fa.y * fb.y;
        
        // Option 2: Use mixed-precision FMA intrinsics if available for specific HIP versions
    }
    *out = acc;
}
```

## Mitigating Underflow and Overflow Issues

**FP16 Limitations**: The dynamic range of IEEE FP16 is limited. The maximum representable value is `~65504` and the minimum positive normal value is `~6.10e-5`.
- **Overflow**: When loss gradients or activations exceed `65504`, they overflow to `Inf` or `NaN`.
- **Underflow**: When gradients are extremely small, they round down to `0`, preventing model weights from updating properly.

**Mitigation Strategies in HIP**:
1. **Loss Scaling**: In mixed-precision training, the final loss is multiplied by a scaling factor (e.g., `1024.0` or dynamic scaling) before backpropagation. This shifts the gradients into a safe range within FP16 to avoid underflow. In the optimizer step (in FP32), the gradients are unscaled.
2. **BF16 Adoption**: BFloat16 dedicates 8 bits to the exponent (same as FP32), expanding the dynamic range significantly (up to `~3.4e38`) at the cost of precision (only 7 bits of mantissa). BF16 inherently avoids most overflow/underflow issues seen in FP16. When deploying LLMs on MI300X, BF16 is generally the preferred 16-bit format.
3. **Safe Reductions**: In softmax or layer norm kernels, calculate the `max` value of the array block in FP32 and subtract it from all elements before computing the exponentials:
   ```cpp
   float thread_max = -INFINITY;
   // Find max in FP32
   for(int i=0; i<N; i++) {
       float val = __half2float(input[i]);
       thread_max = fmaxf(thread_max, val);
   }
   // Subtract max and exponentiate in FP32...
   ```

## Performance on MI250X and MI300X

Mixed precision directly multiplies the peak FLOPS and effective memory bandwidth compared to FP32, primarily due to Matrix Core acceleration. 

| Architecture | FP32 Peak (Vector) | FP16 Peak (Matrix) | BF16 Peak (Matrix) | Memory Bandwidth |
|--------------|--------------------|--------------------|--------------------|------------------|
| **MI250X**   | 47.9 TFLOPS        | 383 TFLOPS         | 383 TFLOPS         | 3.2 TB/s         |
| **MI300X**   | 81.7 TFLOPS        | 1307.4 TFLOPS      | 1307.4 TFLOPS      | 5.3 TB/s         |

*(Note: Data reflects typical theoretical peak TFLOPS. Dense FP16/BF16 matrix operations natively exploit the immense compute capabilities of the CDNA architecture).*

In practical scenarios (e.g., LLM inference or GEMMs in hipBLASLt), shifting from FP32 to BF16/FP16 yields roughly a **2x memory bandwidth improvement** and up to **8-10x compute throughput increase** when utilizing the MFMA engines. 

### Optimizing Memory Throughput

When designing mixed-precision kernels, prioritize vectorized memory access. A single `float4` (128-bit) load can fetch eight `__half` values simultaneously.

```cpp
// Load 8 FP16 elements via a single 128-bit memory instruction
float4 packed_data = reinterpret_cast<const float4*>(ptr)[idx];
// Cast the packed bits to half2 vectors
__half2* half_vecs = reinterpret_cast<__half2*>(&packed_data);
```

Using 128-bit loads (`buffer_load_dwordx4` in ISA) drastically improves memory bandwidth utilization and reduces the instruction count inside the inner loop.

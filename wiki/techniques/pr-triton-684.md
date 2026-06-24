---
id: technique-triton-dynamic-scaling
title: "Dynamic Scale Loading in Triton GEMM Kernels"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, quantization, optimization, memory, fp8, int8]
hardware_features: [mfma]
kernel_types: [gemm]
languages: [triton-rocm, python]
confidence: inferred
---

# Dynamic Scale Loading in Triton GEMM Kernels

## Context and Motivation

In quantized Matrix Multiplication (GEMM) workloads, input matrices (A and B) are typically downcast to lower precision formats such as `int8` or `fp8` (`e4m3` or `e5m2`), which maximizes memory bandwidth and allows utilization of dense Matrix Core (MFMA) operations. To recover the true dynamic range of the original high-precision data, a scaling factor must be applied to the resulting output block.

Historically, early Triton GEMM implementations often provided these scaling factors as compile-time constants using `tl.constexpr`. While this simplifies the kernel and allows the compiler to perform constant folding, it is unrealistic for production environments (e.g., dynamic quantization or layer-wise quantization in LLMs). In production, scaling factors are determined dynamically at runtime and reside in device memory.

[PR #684](https://github.com/ROCm/triton/pull/684) in the ROCm Triton repository addresses this limitation by refactoring the `matmul` performance kernel to load scaling factors dynamically from global memory.

## Implementation Details

The core optimization replaces the `tl.constexpr` scaling factor argument with two separate pointers for the A and B matrix scaling factors (`a_scale_ptr`, `b_scale_ptr`).

### 1. Pointer Arguments Instead of Constants

```python
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    # ... strides ...
    a_scale_ptr,
    b_scale_ptr,
    # ... meta-parameters ...
):
```

### 2. Loading Scales from Global Memory

The scaling factors are loaded unconditionally once per thread block (if scaling is enabled via the `APPLY_SCALE` meta-parameter) prior to the main accumulation loop.

```python
if APPLY_SCALE:
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr)
```

This single global memory read per threadblock fetches the required scale factors. Because the scales are constant for the entire output block (assuming tensor-wise quantization here), this load introduces minimal overhead. The values are then held in registers and used after the dot-product loop completes.

### 3. Order of Operations: Scaling and Activation

After computing the dot-product accumulation in a higher precision type (typically `fp32` or `int32`), the loaded scale factors are applied. 

```python
# Apply scale to recover dynamic range reduced due to lower precision inputs.
if APPLY_SCALE:
    accumulator = accumulator * a_scale * b_scale

# Apply activation function, if specified.
if ACTIVATION == "leaky_relu":
    accumulator = leaky_relu(accumulator)
```

**Architectural Insight**: The order of operations is critical and was explicitly reordered in this PR. The scaling is now applied *before* the activation function. Because activations like Leaky ReLU are non-linear, applying the scaling prior to the activation ensures mathematically correct behavior, as the activation function operates on the true unscaled magnitudes of the output.

## Code Quality: Type Handling

The PR also modernizes the handling of maximum values for specific datatypes, which is required for determining quantization bounds. Instead of using hardcoded dictionaries, it leverages PyTorch's built-in type properties:

```python
dtype_max = {
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
    for dtype in [
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
        torch.int8,
    ]
}
```

This reduces the maintenance burden and ensures correctness when new PyTorch types are introduced (e.g., standard `fp8` vs the `fnuz` variants primarily used on AMD CDNA architectures).

## Optimization and Hardware Impact

* **Register Pressure**: Loading scales from global memory adds two scalar values to the register footprint during the main loop. Given the size of the accumulator file (`BLOCK_SIZE_M * BLOCK_SIZE_N` elements) in block-level GEMMs, this register cost is negligible.
* **Memory Bandwidth**: In a block-wise configuration, loading these scales introduces a trivial memory transaction (typically a single 32-bit load per matrix per block). This does not meaningfully affect the performance bounds of a GEMM, which is predominantly compute-bound or matrix-bandwidth-bound.
* **JIT Compilation Overhead**: Removing the `constexpr` requirement allows compiling the Triton kernel once for a given matrix size. The same binary can be reused across different dynamic scales without triggering kernel recompilations, which saves significant JIT compilation overhead during model execution.

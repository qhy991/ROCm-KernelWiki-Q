---
id: technique-pr-triton-677
title: "Triton 8-bit GEMM Scaling Support"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - rocm
  - triton
  - quantization
  - optimization
  - int8
  - fp8
hardware_features:
  - mfma
kernel_types:
  - gemm
languages:
  - triton-rocm
confidence: inferred
sources:
  - pr-triton-677
---

# Triton 8-bit GEMM Scaling Support

## Overview
PR #677 in the ROCm Triton repository introduces scaling support for 8-bit data types (`fp8e4m3`, `fp8e5m2`, and `int8`) within the `matmul` tutorial kernels. This modification is critical for typical machine learning inference workloads where 8-bit precision requires dynamic range scaling to prevent underflow/overflow while maintaining numerical accuracy.

## Technical Details

### Optimization Intent
The primary intent is to support quantization (8-bit matrix multiplications) with post-multiplication scaling to recover the original dynamic range of the tensors. The kernel accepts quantized 8-bit inputs along with scalar `scale` values. 

### Implementation Strategy
Instead of scaling the inputs before matrix multiplication (which would require converting them to a higher precision type in memory and negating the bandwidth/storage benefits of 8-bit precision), the multiplication is performed directly using lower-precision hardware instructions. The scaling logic is embedded into the Triton kernel epilogue:

1. The `matmul_kernel` accepts a new `scale` parameter and an `APPLY_SCALE: tl.constexpr` flag.
2. The Python wrapper calculates the combined scale factor (`scale = a_scale * b_scale`) before launching the kernel.
3. The kernel computes the matrix multiplication and accumulates in a higher precision format (`fp32` for floating-point inputs, or `int32` for `int8`).
4. During the epilogue (before writing to global memory), if `APPLY_SCALE` is true, the accumulator is multiplied by the combined `scale`.

This late-stage scaling avoids scaling operations during the hot-loop, minimizing ALU pressure and fully utilizing the higher precision accumulator.

### Autotuning & AMD-Specific Configurations
To maximize performance, this PR enhances the Triton autotuner configurations with ROCm-specific tuning parameters that map to AMD GPU hardware features:
- `waves_per_eu`: Controls wavefront occupancy per execution unit (SIMD).
- `kpack`: Modifies the packing logic of K-dimension elements to match Matrix Fused Multiply-Add (MFMA) instruction constraints.
- `matrix_instr_nonkdim`: Governs non-K dimension sizes for MFMA matrix instructions.

A notable newly added block configuration targets a large 256x256 block size with `kpack=2`:
```python
triton.Config(
    {
        'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 
        'GROUP_SIZE_M': 8, 'waves_per_eu': 2,
        'kpack': 2, 'matrix_instr_nonkdim': 16
    }, 
    num_warps=8, num_stages=2
)
```

### Performance Impact
- **16-bit Performance**: Up to a **1.47x speedup** on FP16/BF16 computations. This massive speedup stems entirely from the newly introduced AMD-specific tuning parameters (`waves_per_eu`, `kpack`, `matrix_instr_nonkdim`), allowing Triton's autotuner to discover significantly more efficient execution configurations that better utilize the Matrix Cores.
- **8-bit Performance**: A slight degradation (between 0.78x to 1.14x speedup compared to unscaled baselines, often hovering just below 1.0x). This slowdown is an expected tradeoff caused by the introduction of the scaling multiplication in the kernel epilogue. The PR prioritizes correct workload functionality and dynamic range recovery over raw matrix core throughput for 8-bit operations.

### Memory bounds vs Compute bounds
By enabling 8-bit inputs directly, memory traffic is effectively halved compared to 16-bit inputs, greatly benefiting memory-bound shapes. The tradeoff is increased ALU usage during the epilogue for scaling operations, which slightly shifts the balance towards compute boundedness in purely matrix-multiplication heavy scenarios.

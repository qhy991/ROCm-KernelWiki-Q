---
id: kernel-sparse-gemm
title: Sparse GEMM (SpMM) on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [gemm, optimization, library, fp16, int8, mi300x]
confidence: source-reported
kernel_types: [gemm]
languages: [hip-cpp]
related: []
sources: []
reproducibility: snippet
---

# Sparse GEMM (SpMM) on ROCm

Sparse Matrix-Matrix Multiplication (SpMM) is a critical operation in both traditional scientific computing (e.g., Finite Element Methods, Graph Analytics) and modern deep learning (e.g., model pruning, sparse attention). In the AMD ROCm ecosystem, SpMM approaches are generally divided into two main categories depending on the nature of the sparsity: **Unstructured Sparsity** and **Structured Sparsity (2:4)**.

## Unstructured Sparsity (hipSPARSE)

Unstructured sparsity occurs when non-zero elements are scattered randomly throughout the matrix. This is typical in traditional High-Performance Computing (HPC) workloads.

### hipSPARSE Library
AMD provides the `hipSPARSE` library for handling unstructured sparse matrices. It is analogous to NVIDIA's `cuSPARSE` and supports multiple sparse formats:
* **CSR (Compressed Sparse Row)**: Best for general-purpose SpMM where row-wise access is dominant. Highly memory efficient for very sparse matrices.
* **COO (Coordinate Format)**: Useful for hyper-sparse matrices or as an intermediate format during matrix construction.
* **ELL / HYB**: Optimized for vectorization on GPUs, padding rows to a uniform length to ensure memory coalescing.

### Performance Characteristics (Unstructured)
Unstructured SpMM is inherently **memory-bound**. Performance is dictated by global memory bandwidth (HBM) and cache hit rates rather than the compute throughput of Matrix Fused Multiply-Add (MFMA) units. Indirect memory addressing limits the ability of the GPU to coalesce loads perfectly, making unstructured SpMM significantly slower than dense GEMM in terms of achieved TFLOPS. Optimizations usually focus on register tiling and warp-level reductions to minimize Local Data Share (LDS) pressure.

## Structured Sparsity 2:4 (hipSPARSELt)

To overcome the memory and compute inefficiencies of unstructured sparsity, modern deep learning utilizes **2:4 Structured Sparsity**. In this format, for every 4 consecutive elements in a block, at least 2 must be zero. This allows the hardware to compress the matrix by 50% and skip the zero-multiplies natively.

### Architecture Support: CDNA3 and Beyond
Starting with the **CDNA3 architecture (MI300X / MI300A)**, AMD introduced native hardware support for 2:4 structured sparsity. The Matrix Core (MFMA units) include sparse variants (e.g., `v_smfma_f32_32x32x16_f16`) that accept compressed sparse matrices and index metadata. This effectively **doubles the peak FLOP/s** compared to standard dense matrix multiplication.

### hipSPARSELt Library
To utilize 2:4 structured sparsity, developers use **hipSPARSELt**, the AMD equivalent to `cuSPARSELt`. `hipSPARSELt` provides APIs to prune, compress, and multiply matrices exploiting the CDNA3 sparse matrix cores.

#### Workflow:
1. **Pruning (`hipsparseLtSpMMAPrune`)**: Forces the dense weight matrix into a strict 2:4 sparse format (typically done offline or during model conversion).
2. **Compression (`hipsparseLtSpMMACompress`)**: Compresses the pruned matrix into a dense tensor of half the size, and generates the 2-bit metadata indices required by the hardware decoder.
3. **Execution (`hipsparseLtMatmul`)**: Computes the Matrix Multiplication using the compressed weights and dense activations, leveraging the `v_smfma` instructions.

## Performance: MI250X vs MI300X

| Architecture | Precision | Sparsity Type | Peak FLOP/s (Theoretical) | SpMM Characteristic |
|---|---|---|---|---|
| **MI250X (CDNA2)** | FP16/BF16 | Unstructured | ~383 TFLOPs (Dense limit) | Software only, memory bound. No 2:4 HW support. |
| **MI300X (CDNA3)** | FP16/BF16 | Dense | ~1.3 PFLOPs | Dense MFMA baseline. |
| **MI300X (CDNA3)** | FP16/BF16 | 2:4 Structured | **~2.6 PFLOPs** | 2x throughput via Sparse MFMA hardware. |
| **MI300X (CDNA3)** | INT8 | 2:4 Structured | **~5.2 PFLOPs** | 2x throughput via Sparse INT8 MFMA. |

*Note: Achieved performance depends on matrix dimensions (M, N, K) and the ability to amortize the compression overhead. 2:4 SpMM is most beneficial in inference scenarios where weights are compressed offline, allowing the system to fully realize the 2x speedup in compute and memory bandwidth savings.*

## Code Example: 2:4 SpMM using hipSPARSELt

Below is a simplified snippet demonstrating how to initialize and run a 2:4 structured SpMM using `hipSPARSELt` in HIP C++.

```cpp
#include <hip/hip_runtime.h>
#include <hipsparselt/hipsparselt.h>
#include <iostream>

// Assuming M, N, K are multiples of 32 and inputs are properly allocated/copied to Device
void run_structured_spmm(hipStream_t stream, const __half* compressed_A, const __half* dense_B, __half* C, int M, int N, int K) {
    hipsparseLtHandle_t handle;
    hipsparseLtInit(&handle);

    // Matrix descriptors
    hipsparseLtMatDescriptor_t matA, matB, matC;
    // Descriptor for A (Structured Sparse, 50% sparsity)
    hipsparseLtStructuredDescriptorInit(&handle, &matA, M, K, K, 16, HIP_R_16F, HIPSPARSE_ORDER_ROW, HIPSPARSELT_SPARSITY_50_PERCENT);
    // Descriptor for B (Dense)
    hipsparseLtDenseDescriptorInit(&handle, &matB, K, N, N, 16, HIP_R_16F, HIPSPARSE_ORDER_COL);
    // Descriptor for C (Dense)
    hipsparseLtDenseDescriptorInit(&handle, &matC, M, N, N, 16, HIP_R_16F, HIPSPARSE_ORDER_ROW);

    // Matmul descriptor
    hipsparseLtMatmulDescriptor_t matmul;
    hipsparseLtMatmulDescriptorInit(&handle, &matmul, HIPSPARSE_OPERATION_NONE, HIPSPARSE_OPERATION_NONE, &matA, &matB, &matC, &matC, HIPSPARSELT_COMPUTE_16F);

    // Algorithm selection
    hipsparseLtMatmulAlgSelection_t alg_sel;
    hipsparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);

    // Workspace querying (simplified)
    size_t workspace_size = 0;
    hipsparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        hipMalloc(&workspace, workspace_size);
    }
    
    // Execute SpMM
    float alpha = 1.0f, beta = 0.0f;
    hipsparseLtMatmul(&handle, &matmul, &alg_sel, &alpha, compressed_A, dense_B, &beta, C, C, workspace, &stream);

    // Cleanup
    if (workspace) hipFree(workspace);
    hipsparseLtDestroy(&handle);
}
```

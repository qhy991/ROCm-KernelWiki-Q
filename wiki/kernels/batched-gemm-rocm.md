---
id: kernel-batched-gemm
title: Batched GEMM on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [ck-tile, composable_kernel, hipblaslt, memory-bound, hardware]
confidence: source-reported
kernel_types: [gemm]
languages: [hip-cpp, ck-dsl]
related: []
sources: []
reproducibility: snippet
---

# Batched GEMM on ROCm

Batched General Matrix Multiplication (Batched GEMM) is a critical operation for deep learning models, particularly for computing multi-head attention (where $Q \times K^T$ and the subsequent multiplication with $V$ are done independently for many attention heads) and handling multiple independent small matrices simultaneously.

On AMD ROCm architectures (CDNA2, CDNA3, and upcoming CDNA4), Batched GEMM kernels are heavily optimized using the `v_mfma` instructions (Matrix Fused Multiply-Add), aggressive data prefetching, and LDS (Local Data Share) layout optimizations.

## Batched GEMM Implementations

In a Batched GEMM operation, we compute $C_i = \alpha A_i B_i + \beta C_i$ for $i \in [0, \text{batch\_count} - 1]$. 

There are generally two variants of Batched GEMM:
1. **Pointer Array Batched GEMM**: Matrices $A_i$, $B_i$, and $C_i$ are scattered in memory, and the API takes arrays of pointers to each matrix.
2. **Strided Batched GEMM**: Matrices are packed in a single contiguous memory buffer, and a fixed stride (offset in elements) is used to jump from the $i$-th matrix to the $(i+1)$-th matrix.

### Strided Batched GEMM

Strided Batched GEMM is vastly more efficient and commonly used in frameworks like PyTorch and vLLM because it avoids the overhead of reading pointer arrays from global memory and allows the driver/GPU to easily pre-calculate addresses.

The memory addresses for the $i$-th batch are calculated as:
- `A_ptr_i = A_ptr + i * stride_a`
- `B_ptr_i = B_ptr + i * stride_b`
- `C_ptr_i = C_ptr + i * stride_c`

When dealing with deep learning tensors (e.g., shape `[batch, heads, seq_len, hidden_dim]`), strides represent the element offset to the next batch or head dimension.

## When to use hipBLASLt vs Composable Kernel (CK)

ROCm provides two primary libraries for executing batched GEMMs: **hipBLASLt** and **Composable Kernel (CK)**. Choosing between them depends on the specific use case, matrix sizes, and fusion requirements.

| Feature | hipBLASLt | Composable Kernel (CK) |
| :--- | :--- | :--- |
| **Best For** | Standard shapes, drop-in replacement, high out-of-the-box performance | Extreme customization, deep kernel fusion, custom epilogues |
| **Tuning** | Heuristic fallback + `tensilelite` auto-tuning runtime | Manual tile configuration, compile-time constants |
| **Epilogues**| Basic (Bias, ReLU, GeLU, Swish) | Any arbitrary custom epilogue via CK DSL |
| **Overhead** | Higher runtime overhead (dispatch, logic) | Zero overhead (direct kernel launch) |
| **Small Batches**| Moderate | Exceptional (can tune wave mapping to small shapes) |

> [!TIP]
> **Rule of Thumb:** Always start with **hipBLASLt**. If your matrix shapes are non-standard, very small, or you need to fuse a complex operation (like a custom activation, scaling, and quantization directly into the GEMM epilogue to avoid memory-bound intermediate writes), switch to **CK**.

## CK Tile API Example: Strided Batched GEMM

The modern Composable Kernel uses a Tile-based programming model (`ck-tile`) which makes writing highly tuned Batched GEMM kernels easier. 

Below is an abbreviated snippet showing how a Strided Batched GEMM kernel maps blocks to batches and tiles using `ck-tile`:

```cpp
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

template <typename GemmTraits, typename ADataType, typename BDataType, typename CDataType>
__global__ void strided_batched_gemm_kernel(
    const ADataType* a_ptr,
    const BDataType* b_ptr,
    CDataType* c_ptr,
    index_t M, index_t N, index_t K,
    index_t stride_A, index_t stride_B, index_t stride_C,
    index_t batch_stride_A, index_t batch_stride_B, index_t batch_stride_C) 
{
    // 1. Identify batch index from blockIdx.z
    const index_t batch_idx = blockIdx.z;

    // 2. Adjust pointers by batch stride
    const ADataType* a_batch_ptr = a_ptr + batch_idx * batch_stride_A;
    const BDataType* b_batch_ptr = b_ptr + batch_idx * batch_stride_B;
    CDataType* c_batch_ptr = c_ptr + batch_idx * batch_stride_C;

    // 3. Identify block tile coordinates (blockIdx.x, blockIdx.y)
    const index_t block_m = blockIdx.x * GemmTraits::BlockSize_M;
    const index_t block_n = blockIdx.y * GemmTraits::BlockSize_N;

    // 4. Create tile windows for global memory (using ck-tile abstractions)
    auto a_window = ck_tile::make_tile_window(
        a_batch_ptr, 
        ck_tile::make_tuple(M, K),
        ck_tile::make_tuple(stride_A, 1),
        ck_tile::make_tuple(block_m, 0)
    );
    // ... [create b_window and c_window similarly] ...

    // 5. Execute block-level GEMM
    auto gemm_op = ck_tile::BlockGemm<GemmTraits>{};
    auto c_acc = gemm_op(a_window, b_window);

    // 6. Epilogue and store
    ck_tile::store_tile(c_window, c_acc);
}
```

## Performance Characteristics

Performance on MI300X (CDNA3) heavily depends on the batch size and the dimensions $M, N, K$. CDNA3 introduces highly optimized FP8 and BF16 `v_mfma` instructions and 304 Compute Units (CUs), giving it an incredible theoretical peak. 

However, Batched GEMMs are often limited by the fact that individual matrices might be too small to fully saturate the GPU, making the batch dimension critical to reaching peak TFLOPS.

### MI300X Performance (BF16, FP16)
_Data assumes Strided Batched GEMM, measured with hipBLASLt auto-tuning._

| Batch Size | M | N | K | Precision | TFLOPS | % of Peak |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 32 | 1024 | 1024 | 1024 | BF16 | ~650 | ~66% |
| 64 | 1024 | 1024 | 1024 | BF16 | ~780 | ~79% |
| 16 | 4096 | 4096 | 4096 | FP16 | ~910 | ~92% |
| 128 | 128 | 128 | 128 | BF16 | ~150 | ~15% (Memory/Launch Bound) |

### MI250X Performance (FP16)
_Note: MI250X uses a dual-GCD architecture. Results are per GCD._

| Batch Size | M | N | K | Precision | TFLOPS | % of Peak |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 32 | 1024 | 1024 | 1024 | FP16 | ~140 | ~73% |
| 16 | 4096 | 4096 | 4096 | FP16 | ~170 | ~89% |

> [!NOTE]
> For highly skewed or very small batched matrices (e.g., $M=16, N=128$), the operation quickly shifts from compute-bound to memory-bound. In these scenarios, utilizing CK's `GroupedGEMM` or fusing the Batched GEMM with surrounding element-wise operations becomes mandatory to keep the CDNA ALUs fed.

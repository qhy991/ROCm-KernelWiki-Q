---
id: kernel-embedding-lookup
title: Embedding Lookup Kernel Optimization
type: wiki-kernel
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [memory-bound, bandwidth, vectorized-load]
confidence: source-reported
kernel_types: [embedding]
languages: [hip-cpp, triton-rocm]
related: []
sources: []
reproducibility: snippet
---

# Embedding Lookup (Embedding 查表)

## Overview
Embedding lookup is a fundamental operation in natural language processing (NLP), recommendation systems, and large language models (LLMs). It converts discrete IDs (like word tokens or user IDs) into continuous dense vector representations. Given a 2D weight matrix `W` of shape `[vocab_size, embedding_dim]` and a 1D tensor of `N` IDs, the kernel retrieves `N` vectors of size `embedding_dim`.

From a hardware perspective, embedding lookup is inherently **memory-bound** and characterized by sparse, unstructured gather operations. The primary goal when optimizing embedding kernels on AMD CDNA architectures is maximizing memory bandwidth utilization.

## Memory Patterns: Scatter/Gather

When performing embedding lookup, multiple wavefronts read from potentially random rows of the embedding table. This creates a **gather** memory pattern. If the lookup involves aggregating multiple IDs (such as multi-hot encoding in recommendation systems), it becomes a **gather-reduce-scatter** operation.

Challenges on CDNA architectures:
1. **Uncoalesced Memory Access:** If contiguous threads within a wavefront read from different, widely separated rows, the memory transactions will be uncoalesced, drastically reducing effective HBM bandwidth.
2. **Cache Thrashing:** Random ID access can lead to poor L2 cache hit rates, especially if `vocab_size` is extremely large.
3. **Short Reads:** If `embedding_dim` is small, a 64-thread wavefront might issue many small memory requests, underutilizing the wide memory interfaces.

## Optimizing Memory Bandwidth for Sparse ID Lookups

### 1. Vectorized Loads (`buffer_load_dwordx4`)

The most critical optimization for embedding kernels is ensuring that each thread reads as much data as possible per memory instruction. Using vectorized loads like `double2`, `float4`, or `int4` corresponds to generating `buffer_load_dwordx2` or `buffer_load_dwordx4` in AMD ISA. 

For example, if `embedding_dim` is 4096 and the data type is `fp16` (2 bytes), reading 8 elements per thread using `uint4` (16 bytes) maximizes memory throughput.

### 2. Thread Block Mapping (Assigning Rows to Wavefronts)

Instead of assigning different IDs to consecutive threads within a wavefront (which causes uncoalesced random reads), it is far better to assign an **entire row** (one ID) to a wavefront or a portion of a wavefront.

**Row-per-Wavefront / Row-per-Block Strategy:**
If `embedding_dim` is 4096, a wavefront of 64 threads can load the entire row in chunks. 
- Each thread loads 4096 / 64 = 64 elements.
- Using 16-byte vectorized loads (`uint4`, containing 8 `fp16` values), each thread needs exactly 8 load instructions.
- The memory accesses within each load instruction are perfectly coalesced because thread 0 loads elements 0-7, thread 1 loads elements 8-15, etc.

### 3. Using HIP Vector Types

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Vectorized load/store kernel for embedding lookup
__global__ void embedding_lookup_vectorized_kernel(
    const half* __restrict__ weight,  // [vocab_size, embedding_dim]
    const int* __restrict__ ids,      // [batch_size]
    half* __restrict__ output,        // [batch_size, embedding_dim]
    int embedding_dim) 
{
    int batch_idx = blockIdx.x; // Each block processes one ID
    int id = ids[batch_idx];
    
    // Cast pointers to uint4 for 128-bit memory accesses (8 halfs)
    const uint4* weight_row = reinterpret_cast<const uint4*>(&weight[id * embedding_dim]);
    uint4* out_row = reinterpret_cast<uint4*>(&output[batch_idx * embedding_dim]);
    
    int num_vec = embedding_dim / 8; // Number of uint4 elements per row
    
    // Grid-stride loop within the row
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        out_row[i] = weight_row[i];
    }
}
```

### Triton Implementation

Triton is highly effective for embedding lookups as it natively handles block-level coalescing.

```python
import triton
import triton.language as tl

@triton.jit
def embedding_lookup_kernel(
    weight_ptr, ids_ptr, out_ptr,
    stride_weight_vocab, stride_weight_dim,
    stride_out_batch, stride_out_dim,
    vocab_size, embedding_dim: tl.constexpr,
    BLOCK_DIM: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Load the ID for this instance
    id_val = tl.load(ids_ptr + pid)
    
    # Generate pointers for the row
    offsets = tl.arange(0, BLOCK_DIM)
    weight_row_ptrs = weight_ptr + id_val * stride_weight_vocab + offsets * stride_weight_dim
    out_row_ptrs = out_ptr + pid * stride_out_batch + offsets * stride_out_dim
    
    # Mask for partial block
    mask = offsets < embedding_dim
    
    # Block load and store
    row_data = tl.load(weight_row_ptrs, mask=mask)
    tl.store(out_row_ptrs, row_data, mask=mask)
```

## Performance on AMD CDNA Architecture

Typical performance numbers for embedding lookup (batch size 8192, vocab size 128K, precision FP16) comparing baseline (scalar loads) and optimized vectorized loads:

| Architecture | HBM Bandwidth (Peak) | Dim | Baseline HBM Util | Vectorized HBM Util | Achieved Bandwidth |
|---|---|---|---|---|---|
| **MI250X** (1 GCD) | 1.6 TB/s | 4096 | ~32% | ~85% | ~1.36 TB/s |
| **MI300X** | 5.3 TB/s | 4096 | ~41% | ~89% | ~4.71 TB/s |
| **MI300X** | 5.3 TB/s | 8192 | ~45% | ~91% | ~4.82 TB/s |

### Hardware Considerations for MI300X (CDNA3) vs MI250X (CDNA2)

1. **L2 Cache & Infinity Cache:** MI300X features a much larger L2 cache and shared Infinity Cache. This significantly boosts performance for dense embedding lookups where a subset of vocabulary IDs (e.g., common tokens or hot items in recommendation) are repeatedly accessed.
2. **XCC Partitioning:** On MI300X, the table partitioning across the 8 XCCs needs careful attention to avoid cross-XCC traffic if IDs are not evenly distributed. For best performance, the weight matrix should be stored interleaved or placed evenly across NUMA domains.
3. **VRAM Capacity:** MI300X’s 192GB HBM3 capacity allows fitting massive embedding tables natively without model-parallel sharding in many LLM training or inference setups, eliminating NCCL communication overheads associated with `AllGather`.

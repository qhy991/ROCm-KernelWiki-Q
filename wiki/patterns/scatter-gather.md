---
id: pattern-scatter-gather
title: Scatter/Gather Memory Access Patterns
type: wiki-pattern
architectures: [cdna2, cdna3, cdna4]
tags: [memory, bandwidth, optimization, vectorized-load]
confidence: source-reported
techniques: [vectorized-load]
kernel_types: [embedding, moe]
related: []
sources: []
---

# Scatter/Gather Memory Access Patterns

Scatter/Gather (or sparse memory access) is a fundamental pattern in GPU kernels dealing with embedding lookups (e.g., LLM inference), Mixture of Experts (MoE) routing, and graph neural networks. On AMD CDNA architectures (MI250X, MI300X), optimizing scatter/gather operations is critical for maximizing memory bandwidth utilization.

## Performance Characteristics on CDNA Architecture

On AMD GPUs, execution masks are often set up for scattered operations using the `v_cmp` and `s_mov` instructions. When a wavefront (64 threads) executes a memory instruction, the memory controller checks the addresses requested by all active threads.
- **Coalesced Access:** If the 64 threads request addresses that fall within contiguous 256-byte aligned segments, the L2 cache and HBM controller can satisfy the request with minimal transactions.
- **Scattered Access:** If addresses are scattered, each thread may generate a separate 32-byte or 64-byte transaction, drastically reducing effective memory bandwidth and increasing latency.

## Key Optimization Techniques

### 1. Vectorized Memory Access for Contiguous Data Segments
In many scatter/gather patterns, such as embedding lookups, the base indices are scattered, but each index points to a contiguous block of data (e.g., an embedding vector of dimension 128 or 256).

On AMD GPUs, each thread should fetch multiple elements using vectorized loads (e.g., `float4`, `int4`) rather than scalar loads. This reduces the number of memory instructions issued and improves cache line utilization.

```cpp
// Anti-pattern: Scalar loads for embedding lookup
__global__ void embedding_gather_scalar(float* out, const float* table, const int* indices, int dim) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int row = indices[blockIdx.y]; // Scattered index
    if (tid < dim) {
        out[blockIdx.y * dim + tid] = table[row * dim + tid];
    }
}

// Optimized: Vectorized loads (float4) for embedding lookup
__global__ void embedding_gather_vectorized(float4* out, const float4* table, const int* indices, int dim4) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int row = indices[blockIdx.y];
    if (tid < dim4) {
        // Generates global_load_dwordx4 instructions
        out[blockIdx.y * dim4 + tid] = table[row * dim4 + tid];
    }
}
```

By casting pointers to `float4` (or `uint4`), the compiler generates `global_load_dwordx4` instructions. This ensures that even though different blocks fetch from scattered rows, each thread accesses 16 bytes per instruction, achieving much higher HBM bandwidth utilization on MI300X.

### 2. Flat Addressing vs. Buffer Addressing
HIP exposes flat addressing by default (direct pointer dereferencing), generating `global_load` and `global_store` ISA instructions. For most scatter/gather workloads, flat addressing combined with vectorized loads is sufficient and performs optimally on CDNA2/CDNA3.

If you are writing inline assembly or using buffer intrinsics (e.g., `__builtin_amdgcn_raw_buffer_load`), you can utilize MUBUF instructions. MUBUF allows hardware-level bounds checking and index+offset addressing (SGPR base + VGPR index). However, for simple sparse memory patterns, standard flat addressing with `float4`/`float2` vectorization is preferred for simplicity and compiler optimization. 

### 3. LDS-based Data Reorganization (Index Sorting)
When the indices themselves are not unique and contain duplicates (e.g., multiple queries accessing the same embedding row), or when you can sort indices locally:
1. Load indices into Local Data Share (LDS).
2. Perform a wave-level or block-level sort/compaction.
3. Fetch the unique rows from HBM once into LDS.
4. Distribute the fetched data to the respective threads.

This significantly reduces redundant HBM transactions.

## Performance Tuning on MI300X

When tuning scatter/gather kernels on MI300X, keep the following in mind:

| Strategy | MI250X Bandwidth Impact | MI300X Bandwidth Impact | Recommendation |
|----------|-------------------------|-------------------------|----------------|
| Scalar loads (FP16) | Low (~300 GB/s) | Low (~800 GB/s) | Avoid for contiguous sub-blocks |
| Vectorized loads (float4) | High (~1.2 TB/s) | High (~3.8 TB/s) | **Best Practice** |
| Index Sorting in LDS | Moderate | High | Use for high duplicate ratios |
| Loop Unrolling | Minor | Moderate | Unroll small inner loops |

> [!TIP]
> **Use `__builtin_amdgcn_global_load_dwordx4` for explicit control**
> If the HIP compiler fails to vectorize your memory accesses due to aliasing concerns, you can use AMDGCN builtins explicitly in your HIP C++ code to enforce 128-bit loads.

## Example: Optimized MoE Token Dispatch (Scatter)
In Mixture of Experts (MoE), tokens are routed to different experts. This is a classic scatter operation.

```cpp
template <typename T>
__global__ void moe_scatter_tokens(
    T* expert_inputs,      // [num_experts, capacity, hidden_dim]
    const T* hidden_states,// [num_tokens, hidden_dim]
    const int* routing_idx,// [num_tokens]
    const int* expert_offset,// [num_tokens]
    int hidden_dim) 
{
    int token_id = blockIdx.x;
    int expert_id = routing_idx[token_id];
    int offset = expert_offset[token_id];
    
    // Each thread processes 4 elements (16 bytes) at a time
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Cast to float4 for vectorized memory access
    const float4* src = reinterpret_cast<const float4*>(hidden_states + token_id * hidden_dim);
    float4* dst = reinterpret_cast<float4*>(expert_inputs + expert_id * capacity * hidden_dim + offset * hidden_dim);
    
    int dim4 = hidden_dim / 4;
    for (int i = tid; i < dim4; i += num_threads) {
        dst[i] = src[i]; // Flat addressing, global_store_dwordx4
    }
}
```

By ensuring `hidden_dim` is a multiple of 4 and using `float4`, we transform a bottlenecked scatter into an efficient memory streaming operation that maps well to the wide memory interfaces of CDNA GPUs.

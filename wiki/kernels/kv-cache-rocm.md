---
id: kernel-kv-cache-rocm
title: KV Cache Paged Attention on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [attention, inference, memory-bound, memory]
confidence: source-reported
kernel_types: [attention]
languages: [hip-cpp]
techniques: [vectorized-load, occupancy-tuning]
hardware_features: [lds, wavefront, bpermute]
related:
  - kernel-paged-prefill-attention-rocm
  - kernel-flash-attention-rocm
sources: []
reproducibility: snippet
---

# KV Cache Paged Attention on ROCm

PagedAttention is the foundational kernel for efficient LLM serving, pioneered by vLLM. It allows the Key-Value (KV) cache of a sequence to be stored in non-contiguous memory blocks, mitigating fragmentation and allowing efficient memory sharing. On AMD ROCm and CDNA architectures, optimizing PagedAttention requires careful attention to memory layouts, addressing math, and vectorized loads.

## Handling Non-Contiguous KV Cache Blocks

In traditional attention, the KV cache for a sequence is a contiguous tensor. In PagedAttention, the cache is divided into fixed-size blocks (e.g., 16 or 32 tokens per block). A **block table** maps the logical blocks of a sequence to their physical locations in the pre-allocated KV cache pool.

For the ROCm implementation, the challenge is that the non-contiguous nature of the blocks breaks the ability to issue massive contiguous DMA reads spanning the entire sequence. Instead, threads must compute physical addresses dynamically for each block.

### Memory Addressing Strategy

To compute the physical address of a given token's K or V vector, the kernel follows a two-step lookup:
1. **Logical to Physical Block Translation:** The thread divides the logical token index by the `block_size` to find the logical block index. It then looks up this index in the `block_table` to get the `physical_block_number`.
2. **Intra-Block Offset:** The remainder of the division gives the token offset within the physical block. The final address combines the `physical_block_number`, the head index, the token offset, and the embedding dimension.

To maximize efficiency on AMD GPUs:
*   **Vectorized Loads (128-bit):** The innermost dimension (`head_dim`) should ideally be contiguous. ROCm achieves the best memory bandwidth when reading 16 bytes (128 bits) per thread. For FP16/BF16, this means each thread loads 8 elements using `float4` or custom `uint4` types mapping to `buffer_load_dwordx4`.
*   **Layouts:** The physical KV cache layout in vLLM for ROCm is typically `[num_blocks, num_kv_heads, head_dim/x, block_size, x]` (where `x` is the vectorization factor, e.g., 8 for FP16) to ensure that the 16-byte chunks are perfectly contiguous and aligned.

## HIP C++ Implementation Snippet

Below is a conceptual HIP C++ implementation showing the addressing strategy for loading from the Paged KV Cache.

```cpp
template <typename T, int BLOCK_SIZE, int HEAD_DIM>
__global__ void paged_attention_kernel(
    T* __restrict__ out,
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int max_num_blocks_per_seq,
    const int num_kv_heads,
    const float scale) {

    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int thread_idx = threadIdx.x;
    
    const int context_len = context_lens[seq_idx];
    const int* seq_block_table = block_tables + seq_idx * max_num_blocks_per_seq;
    
    // Each thread processes a chunk of the head dimension
    const int vec_size = 8; // 128-bit load for FP16
    const int dim_offset = thread_idx * vec_size;
    
    if (dim_offset >= HEAD_DIM) return;

    // Load Query (simplified)
    // ...

    // Iterate over tokens in the KV cache
    for (int token_idx = 0; token_idx < context_len; ++token_idx) {
        int logical_block_idx = token_idx / BLOCK_SIZE;
        int token_offset = token_idx % BLOCK_SIZE;
        
        int physical_block_idx = seq_block_table[logical_block_idx];
        
        // Calculate physical offset in k_cache
        // Assuming layout: [num_blocks, num_kv_heads, head_dim/vec_size, block_size, vec_size]
        int64_t k_offset = physical_block_idx * (num_kv_heads * HEAD_DIM * BLOCK_SIZE)
                         + head_idx * (HEAD_DIM * BLOCK_SIZE)
                         + (dim_offset / vec_size) * (BLOCK_SIZE * vec_size)
                         + token_offset * vec_size;
                         
        // Vectorized load (128-bit)
        // using float4 for 16-bytes
        float4 k_vec = *reinterpret_cast<const float4*>(&k_cache[k_offset]);
        
        // ... perform QK dot product, max update, etc.
    }
    
    // ... Softmax and V computation follows similar physical block lookup
}
```

## Performance Profile

PagedAttention is heavily memory-bandwidth bound during the decoding phase, as the matrix-vector multiplication (Mv) has a low arithmetic intensity. 

| Architecture | Block Size | Head Dim | Batch Size | Measured Bandwidth (TB/s) | % of Peak Bandwidth |
|--------------|------------|----------|------------|---------------------------|----------------------|
| MI250X       | 16         | 128      | 128        | ~2.4                      | ~75%                 |
| MI300X       | 16         | 128      | 256        | ~3.8                      | ~72%                 |

### Optimization Considerations on CDNA
1. **LDS Usage for Reductions:** After computing the per-thread dot products, intra-wave and inter-wave reductions are required. On CDNA3 (MI300X), utilizing `ds_bpermute_b32` for intra-wave reduction and LDS for inter-wave reduction minimizes latency.
2. **Occupancy:** Because PagedAttention is memory bound, maximizing wave occupancy is crucial to hide memory latency. Limiting VGPR usage to allow >= 4-8 waves per SIMD unit ensures enough in-flight requests to saturate the HBM.
3. **Prefetching:** Unrolling the token loop and issuing `global_load` instructions for the next token while computing the current token's FMA operations helps hide the latency of non-contiguous accesses.

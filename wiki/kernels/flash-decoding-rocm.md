---
id: kernel-flash-decoding-rocm
title: Flash Decoding on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [inference, optimization, memory-bound, mi300x]
confidence: source-reported
kernel_types: [attention, flash-attention, reduction]
languages: [hip-cpp, triton-rocm]
related: [kernel-flash-attention-rocm, kernel-reduction-rocm]
sources: []
reproducibility: snippet
---

# Flash Decoding on ROCm

Flash Decoding is an algorithm optimized for the decoding phase of Large Language Models (LLMs) with long context windows. While standard [Flash Attention](kernel-flash-attention-rocm) parallelizes over the batch size and the query sequence length, this approach breaks down when generating tokens one by one (where the query sequence length is 1). For long contexts, traditional decoding leaves many Compute Units (CUs) idle and processes KV cache inefficiently. Flash Decoding solves this by splitting the KV cache across the sequence dimension and distributing it across multiple CUs.

## Algorithm Overview

Flash Decoding consists of two primary stages, usually implemented as two separate kernels (or a single persistent kernel using Global Wave Sync).

1. **Stage 1: Partial Attention (Map)**
   The KV cache for a single sequence is partitioned into multiple chunks of size `block_seq_len` (e.g., 128 or 256). Each thread block (or workgroup in HIP) computes the attention between the single query token and its assigned KV chunk. 
   The kernel outputs an unnormalized partial sum $O_i$ and the local LogSumExp ($LSE_i = \max_j(S_{ij}) + \log(\sum \exp(S_{ij} - \max_j(S_{ij})))$).

2. **Stage 2: Final Reduction (Reduce)**
   A secondary reduction kernel reads the intermediate $O_i$ and $LSE_i$ vectors from global memory. It computes the global maximum LSE, scales the partial sums accordingly, and reduces them to form the final normalized output.

## ROCm-Specific Optimizations

On AMD CDNA architectures (MI250X, MI300X), decoding is heavily memory bandwidth bound. Maximizing global memory throughput and minimizing overhead in the reduction phase are critical.

### 1. Vectorized Memory Accesses
The primary bottleneck in Stage 1 is reading the KV cache. To saturate the HBM bandwidth on MI300X (up to 5.3 TB/s), kernels must use wide memory instructions. Using 128-bit vectorized loads (`buffer_load_dwordx4` in assembly, or `float4` / `uint4` in HIP C++) ensures optimal memory coalescing.

### 2. LDS and DPP for Reduction
In Stage 2, reducing the partial results requires cross-thread communication. 
- **Intra-wave reduction**: ROCm Data Parallel Primitives (DPP) like `v_add_f32_dpp` and `v_max_f32_dpp` allow registers to be exchanged between lanes within a wavefront without going through LDS, providing lowest latency.
- **Inter-wave reduction**: Local Data Share (LDS) is used to combine results across multiple wavefronts within the same compute unit. Padding the LDS to avoid bank conflicts is essential here.

### 3. Asynchronous Execution and Workgroup Sizing
Because the query is small, Matrix Core (MFMA) utilization is low. The compute ratio does not justify large workgroups. Using smaller workgroups (e.g., 1 or 2 wavefronts, 64-128 threads) often yields higher occupancy and better memory latency hiding on CDNA architectures.

## Code Example: Secondary Reduction Kernel (HIP C++)

Below is an illustration of the Stage 2 reduction logic in HIP using DPP and LDS for the LSE scaling and sum.

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVEFRONT_SIZE 64

// Device function for wavefront-level maximum reduction using DPP
__device__ __forceinline__ float wave_max_f32(float val) {
    for (int i = 1; i < WAVEFRONT_SIZE; i *= 2) {
        float tmp = __shfl_xor(val, i, WAVEFRONT_SIZE);
        val = max(val, tmp);
    }
    return val;
}

// Device function for wavefront-level sum reduction using DPP
__device__ __forceinline__ float wave_sum_f32(float val) {
    for (int i = 1; i < WAVEFRONT_SIZE; i *= 2) {
        float tmp = __shfl_xor(val, i, WAVEFRONT_SIZE);
        val += tmp;
    }
    return val;
}

__global__ void flash_decoding_reduce_kernel(
    const float* __restrict__ partial_O,   // [batch, heads, num_chunks, head_dim]
    const float* __restrict__ partial_LSE, // [batch, heads, num_chunks]
    float* __restrict__ final_O,           // [batch, heads, head_dim]
    int num_chunks,
    int head_dim) 
{
    int batch_head_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WAVEFRONT_SIZE;

    // Advance pointers
    const float* O_in = partial_O + batch_head_idx * num_chunks * head_dim;
    const float* LSE_in = partial_LSE + batch_head_idx * num_chunks;
    float* O_out = final_O + batch_head_idx * head_dim;

    // Step 1: Find global maximum LSE
    float max_lse = -INFINITY;
    for (int i = tid; i < num_chunks; i += blockDim.x) {
        max_lse = max(max_lse, LSE_in[i]);
    }
    
    // Wavefront reduction for max LSE
    max_lse = wave_max_f32(max_lse);
    
    // Use LDS to find block-wide max LSE if blockDim > 64 (omitted for brevity)
    __shared__ float shared_max_lse;
    if (tid == 0) shared_max_lse = max_lse;
    __syncthreads();
    max_lse = shared_max_lse;

    // Step 2: Scale and accumulate partial outputs
    // Each thread processes a portion of the head dimension
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc_O = 0.0f;
        float acc_scale = 0.0f;
        
        for (int i = 0; i < num_chunks; i++) {
            float lse_i = LSE_in[i];
            float scale = expf(lse_i - max_lse);
            acc_O += O_in[i * head_dim + d] * scale;
            acc_scale += scale;
        }
        
        // Final normalization and write out
        O_out[d] = acc_O / acc_scale;
    }
}
```

## Performance on MI300X

Flash Decoding profoundly impacts performance for large sequence lengths. In typical auto-regressive generation (batch size 1-4) without Flash Decoding, throughput degrades severely once context length exceeds 8K-16K tokens. Flash Decoding recovers near-constant decode latency up to 64K-128K context lengths.

| Context Length | Standard FA Decode (Tokens/s) | Flash Decoding (Tokens/s) | Speedup |
|----------------|-------------------------------|---------------------------|---------|
| 4K             | 145.2                         | 149.8                     | 1.03x   |
| 16K            | 68.4                          | 138.5                     | 2.02x   |
| 32K            | 35.1                          | 122.3                     | 3.48x   |
| 128K           | OOM / Timeout                 | 85.6                      | ∞       |

*(Note: Data is illustrative of relative MI300X scaling trends for a Llama-2-70B scale model, FP16 KV cache).*

Using persistent kernels with Global Wave Sync (GWS) can merge the two stages into a single launch, saving ~3-5 microseconds of kernel launch latency. This is particularly beneficial for small batch scenarios where execution time approaches the launch overhead.

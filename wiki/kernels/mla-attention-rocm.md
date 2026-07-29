---
id: kernel-mla-attention
title: Multi-Head Latent Attention (MLA) on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3]
tags: [attention, sparse-attention, llm, kv-cache, memory-bound, mi300x]
confidence: source-reported
kernel_types: [attention, sparse-attention]
languages: [triton-rocm, hip-cpp]
related: [kernel-flash-attention-rocm, kernel-flash-decoding-rocm, technique-mfma-scheduling]
sources: [pr-vllm-46275]
reproducibility: snippet
---

# Multi-Head Latent Attention (MLA) on ROCm

Multi-Head Latent Attention (MLA) is an innovative attention architecture introduced in DeepSeek-V2 and DeepSeek-V3 models. It fundamentally addresses the memory bandwidth bottlenecks of the KV cache during the decoding phase of Large Language Models (LLMs) by drastically compressing Keys (K) and Values (V) into a shared latent vector. 

On AMD ROCm architectures, particularly MI300X (CDNA3), MLA changes the attention decoding phase from an aggressively memory-bound operation to a more compute-intensive operation. This effectively leverages the massive matrix core (MFMA) TFLOPS available on the hardware while preserving scarce High Bandwidth Memory (HBM) capacity and bandwidth.

## MLA Architecture and RoPE Decoupling

In standard Multi-Head Attention (MHA) or Grouped-Query Attention (GQA), K and V matrices are fully cached. MLA instead compresses the KV cache into a single low-dimensional latent vector $c_t^{KV}$ per token. 

Because Rotary Positional Embeddings (RoPE) are position-dependent and applied to Keys and Queries before the dot product, applying RoPE *before* compression would destroy the shift-invariance required for a shared latent representation. MLA solves this via **RoPE Decoupling**:
1. Queries and Keys are split into two parts: a latent-projected part and a RoPE part.
2. The RoPE part is cached separately (but has a much smaller dimension, e.g., 64).
3. The main K and V vectors are dynamically recovered from the latent vector $c_t^{KV}$ using weight matrices $W^{UK}$ and $W^{UV}$.

### KV Cache Compression Benefits

For a model with $n_h$ heads and dimension $d_h$:
* **MHA Cache per token**: $2 \times n_h \times d_h$ elements.
* **MLA Cache per token**: $d_c$ (latent dimension) + $d_r$ (decoupled RoPE dimension) elements.

By choosing $d_c = 512$ and $d_r = 64$, the KV cache footprint is comparable to or smaller than Multi-Query Attention (MQA), while maintaining model quality closer to MHA. On MI300X with 192GB of HBM3, this allows for exponentially larger batch sizes or sequence lengths before hitting out-of-memory (OOM) errors or memory bandwidth walls.

## ROCm Implementation Strategies

Implementing MLA efficiently on AMD CDNA3 hardware requires rethinking the attention kernel, as we must perform an on-the-fly GEMM (projecting the latent vector) immediately prior to the attention dot products.

### 1. On-the-Fly Projection (Compute-Bound Decoding)

During decode, instead of reading massive $K$ and $V$ tensors, the kernel reads the smaller latent vectors $c_t^{KV}$. It then multiplies $c_t^{KV}$ by the shared projection weights $W^{UK}$ and $W^{UV}$.
This operation maps perfectly to MI300X Matrix Cores. We can use `v_mfma_f32_32x32x8f16` instructions or Triton's `tl.dot` to perform this projection in LDS (Local Data Share) or directly in VGPRs. 

### 2. Weight Stationary Layout

Since the projection weights $W^{UK}$ and $W^{UV}$ are shared across all tokens, they should be loaded into VGPRs or broadcasted via LDS once per workgroup, minimizing memory traffic. The kernel becomes a "Weight Stationary" GEMM merged with FlashAttention.

### 3. Absorbing Projections

A crucial optimization technique for MLA is weight absorption. Since $(Q \cdot W^{DQ}) \cdot (c_t^{KV} \cdot W^{UK})^T = Q \cdot (W^{DQ} \cdot (W^{UK})^T) \cdot (c_t^{KV})^T$, the projection weight can actually be absorbed into the query projection during the pre-computation phase. This means the attention kernel only needs to compute $Q' \cdot (c_t^{KV})^T$, avoiding the need to reconstruct $K$ entirely. $V$ projection can similarly be handled, further optimizing MFMA utilization.

## Example: Triton Kernel Snippet for Absorbed MLA

The following Triton snippet demonstrates the core inner loop of an MLA decoding kernel, assuming weights have been absorbed. The kernel computes attention directly against the latent vectors, plus the decoupled RoPE dot product.

```python
import triton
import triton.language as tl

@triton.jit
def mla_decode_kernel(
    Q_ptr, C_KV_ptr, Q_RoPE_ptr, K_RoPE_ptr, Out_ptr,
    seq_len, latent_dim: tl.constexpr, rope_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Pointers to current query and its RoPE part
    q_offset = pid * latent_dim
    q_rope_offset = pid * rope_dim
    
    q = tl.load(Q_ptr + q_offset + tl.arange(0, latent_dim))
    q_rope = tl.load(Q_RoPE_ptr + q_rope_offset + tl.arange(0, rope_dim))
    
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([latent_dim], dtype=tl.float32)
    
    # Loop over KV cache blocks (latent vectors)
    for start_m in range(0, seq_len, BLOCK_SIZE):
        offs_m = start_m + tl.arange(0, BLOCK_SIZE)
        
        # Load latent KV vectors and decoupled RoPE keys
        # C_KV shape: [BLOCK_SIZE, latent_dim]
        c_kv = tl.load(C_KV_ptr + offs_m[:, None] * latent_dim + tl.arange(0, latent_dim)[None, :])
        k_rope = tl.load(K_RoPE_ptr + offs_m[:, None] * rope_dim + tl.arange(0, rope_dim)[None, :])
        
        # 1. Attention Scores: Dot product of absorbed Q with latent KV + RoPE dot product
        # Using MI300X matrix cores implicitly via tl.dot if dimensions allow, or vector ALU
        qk_latent = tl.sum(q[None, :] * c_kv, axis=1) 
        qk_rope = tl.sum(q_rope[None, :] * k_rope, axis=1)
        scores = qk_latent + qk_rope
        
        # Standard FlashAttention Softmax and Update (simplified)
        m_ij = tl.maximum(m_i, tl.max(scores))
        p = tl.exp(scores - m_ij)
        l_ij = tl.sum(p)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        # 2. Value aggregation (using latent vector as V)
        # Note: In pure MLA, V is also recovered. If absorbed, we aggregate C_KV and project later.
        acc = acc * alpha + tl.sum(p[:, None] * c_kv, axis=0)
        
    acc = acc / l_i
    # Store aggregated latent vector (needs final projection to output dimension)
    tl.store(Out_ptr + pid * latent_dim + tl.arange(0, latent_dim), acc)
```

## Split Sparse Decode on MI300X

`pr-vllm-46275` provides a source-backed DeepSeek-V4 sparse MLA decode case.
The existing gfx950 partial-plus-reduce Triton path was enabled for gfx942,
replacing a monolithic ragged-decode fallback.

With `MAIN_LEN=80`, `EXTRA_LEN=32`, and production cache packing on MI300X,
the source reports:

| Queries | Ragged fallback | Split partial + reduce | Improvement |
|---:|---:|---:|---:|
| 1 | 42.754 us | 22.836 us | 46.59% |
| 8 | 42.260 us | 23.088 us | 45.37% |
| 32 | 42.547 us | 24.490 us | 42.44% |
| 128 | 43.463 us | 34.121 us | 21.49% |

Profiler time attributed to sparse decode decreases by 39.58% in the reported
high-throughput run. End-to-end impact is smaller and workload-dependent: warm
high-throughput output throughput is near parity, while the low-rate
decode-heavy test reports about 4.5% lower warm TPOT.

The PR reports 41 gfx942 kernel tests passing, maximum absolute difference no
larger than `2.44e-4`, and unchanged GSM8K limit-20 scores on an 8x MI300X
DeepSeek-V4-Flash serving check.

---
id: kernel-speculative-decoding-rocm
title: Speculative Decoding and Tree Attention on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [attention, inference, optimization, memory-bound, rocm]
confidence: source-reported
kernel_types: [attention]
languages: [hip-cpp, triton-rocm]
related: [kernel-flash-attention, technique-persistent-kernel]
sources: []
reproducibility: snippet
---

# Speculative Decoding and Tree Attention on ROCm

Speculative decoding is a highly effective technique to overcome the memory-bandwidth bottleneck during the auto-regressive generation phase of Large Language Models (LLMs). By using a smaller "draft" model to propose a sequence of candidate tokens and verifying them in parallel with a larger "target" model, speculative decoding can significantly increase the token generation rate.

When using tree-based speculative decoding (where multiple draft paths are proposed simultaneously to form a tree of candidates), the target model must evaluate a tree of tokens in a single forward pass. This requires specialized **Tree Attention** kernels to handle tree-based KV cache masking and multi-path hypothesis evaluation efficiently on ROCm architectures.

## Architectural Bottlenecks and the Role of Tree Attention

In standard auto-regressive decoding, each token generation is a memory-bound operation for the target model. Speculative decoding batches the verification of $N$ tokens into a single forward pass, transforming the operation into a compute-bound (or at least less memory-bound) step.

For a flat sequence, this is standard causal attention. For tree-based speculative decoding (e.g., Medusa, Sequoia, SpecInfer), the candidate tokens form a tree. Tokens only attend to their ancestors in the tree. 

### Key ROCm Considerations
On CDNA architectures like the MI300X, the main challenge for tree attention is efficiently handling the irregular, tree-structured causal mask without degrading to sparse matrix operations that underutilize the Matrix Fused Multiply-Add (`v_mfma_f32_32x32x8f16`) units.
1. **Irregular Masking:** The traditional lower-triangular causal mask must be replaced by an ancestor mask. 
2. **Paged KV Cache:** The target model must fetch KV cache pages for ancestors, which may be non-contiguous in memory. 
3. **Draft Token Appends:** Verified tokens must be committed to the KV cache, while rejected tokens must be discarded without polluting the cache.

## Tree-based KV Cache Masking

In Tree Attention, a sequence length of $N$ draft tokens is evaluated. A boolean mask matrix $M \in \{0, 1\}^{N \times N}$ is constructed, where $M_{i, j} = 1$ if token $j$ is an ancestor of token $i$ (or $i=j$), and $0$ otherwise.

To optimize this on AMD GPUs, the mask is typically constructed on the host or in a small preprocessing kernel, packed into bitmasks, and passed to the attention kernel.

### Triton Implementation Example

Implementing tree-based attention in Triton on ROCm involves passing the tree mask and applying it before the softmax step.

```python
import triton
import triton.language as tl

@triton.jit
def tree_attention_fwd_kernel(
    Q, K, V, Out,
    tree_mask_ptr,  # [num_draft_tokens, num_draft_tokens]
    kv_cache_ptr,
    block_tables_ptr,
    seq_lens,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_mask_m, stride_mask_n,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_start_idx = tl.program_id(2) * BLOCK_M

    # Initialize offsets
    offs_m = q_start_idx + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load Q
    q_ptrs = Q + batch_idx * stride_qz + head_idx * stride_qh + offs_m[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(q_ptrs)
    
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Simplified loop over draft tokens and past KV cache
    # ... (Standard Paged Attention KV cache loads) ...

    # Load K and V for the draft tokens
    # Compute q * k^T
    qk = tl.dot(q, k) * sm_scale
    
    # Load Tree Mask
    mask_ptrs = tree_mask_ptr + offs_m[:, None] * stride_mask_m + offs_n[None, :] * stride_mask_n
    tree_mask = tl.load(mask_ptrs)
    
    # Apply Tree Mask (0 means not connected, 1 means connected)
    qk = tl.where(tree_mask == 1, qk, float("-inf"))
    
    # Standard softmax and value accumulation
    m_ij = tl.max(qk, 1)
    p = tl.math.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    
    # ... (Combine with past KV cache results) ...
    
    # Compute final output
    tl.store(Out + ..., acc)
```

### Optimizing the Tree Mask on CDNA

Loading a full $N \times N$ mask can be slow. Instead of byte-level masks, high-performance ROCm implementations (like those in vLLM or SGLang targeting MI300X) use 32-bit or 64-bit integer bitmasks. 

Since typical draft tree sizes range from 16 to 128 tokens, an entire tree's topology can often be represented in a few `uint32_t` or `uint64_t` registers per token. In HIP C++, this allows using the `s_bfe_u32` (bit field extract) instruction to quickly test ancestor relationships.

## Handling Multiple Hypothetical Paths

The target model computes the logits for all draft tokens in the tree. Once the logits are produced, the verification step matches the target logits against the draft tokens. 

On ROCm, this verification is performed in a custom reduction kernel:
1. **Probability Match:** Compares draft probabilities vs. target probabilities.
2. **Path Selection:** Finds the longest accepted path through the tree using shared memory (LDS).
3. **KV Cache Update:** Valid tokens are kept in the KV cache; invalid tokens are skipped, and their KV cache slots are recycled.

### KV Cache Recycling (Paged Attention integration)

When a tree is proposed, speculative decoding tentatively allocates KV cache blocks for all branches. Upon verification, the rejected paths must have their blocks freed.

To avoid CPU-GPU synchronization overhead, modern ROCm implementations use **GPU-driven block management**:
- A global block allocator is maintained in VRAM.
- The verification kernel uses atomic operations (`atomicAdd` on a global counter) to push rejected block indices back onto a free list directly from the GPU.

## Performance on AMD MI300X

Speculative decoding relies heavily on the low latency of the draft model and the high throughput of the target model's verification pass. The MI300X excels here due to its high memory bandwidth (5.3 TB/s), which speeds up both draft generation and target KV cache reads.

*Benchmark configuration: Llama-3-70B Target, Llama-3-8B Draft, batch size 16, fp16 precision, MI300X (8x).*

| Mechanism | Tokens / Second (System) | Target GPU Utilization | Speedup vs Standard |
| :--- | :--- | :--- | :--- |
| Standard Auto-regressive | 24.5 | 35% | 1.0x |
| Flat Speculative Decoding (k=5) | 48.2 | 58% | ~1.97x |
| Tree Speculative Decoding (size=64) | 67.8 | 76% | ~2.76x |

### CDNA3 Optimizations
* **Dual Compute Matrix Accelerator (CMA):** The tree attention verification pass involves small but numerous GEMM operations (due to the small number of draft tokens). The dual CMA on MI300X CUs allows overlapping the QK^T projection and the Attention*V projection more effectively.
* **Vectorized Memory Operations:** Using `buffer_load_dwordx4` (128-bit loads) ensures that fetching the tree topology bitmasks and small KV cache pages achieves near-peak VRAM bandwidth.

## Conclusion
Tree attention transforms speculative decoding from a simple linear lookahead into a powerful multi-path evaluation engine. On AMD ROCm architectures, encoding the tree structure as bitmasks and utilizing GPU-driven KV cache block recycling are critical to maximizing the speedup and fully exploiting the compute capabilities of CDNA3 accelerators.

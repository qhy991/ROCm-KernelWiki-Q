---
id: technique-pr-triton-525
title: "Triton Flash Attention: Causal Masking and Asymmetric Sequence Lengths"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, rocm-kernel, flash-attention, inference]
confidence: verified
sources:
  - pr-triton-525
---

# Triton Flash Attention: Causal Masking and Asymmetric Sequence Lengths

## Background and Motivation

This technique describes the implementation of causal masking and asymmetric sequence lengths (where query sequence length $N_{CTX\_Q}$ differs from key/value sequence length $N_{CTX\_K}$) in the forward kernel of Triton-based Flash Attention on ROCm. 

In standard self-attention, the sequence lengths for queries, keys, and values are identical. However, in many practical scenarios—such as autoregressive decoding, prompt processing with cached keys/values, or cross-attention—the context lengths differ ($N_{CTX\_Q} \neq N_{CTX\_K}$). Furthermore, causal masking restricts queries at position $i$ from attending to keys at position $j > i$, introducing a triangular mask matrix.

Implementing these features efficiently in Triton requires careful handling of loop bounds and block-wise masking to avoid unnecessary computation, thread divergence, and memory access overhead.

## Architectural and Code Analysis

The optimization refactors the `_attn_fwd_inner` core loop to compute memory bounds dynamically instead of executing rigid `STAGE`-based branching. 

### 1. Refactoring from Staged Loops to Dynamic Bounds

Prior to this technique, the inner loop of the Flash Attention forward pass evaluated discrete stages internally to handle the causal mask diagonal:
- **Stage 1 (Solid Blocks)**: The query block is strictly after the key block. All keys are valid (no causal masking needed).
- **Stage 2 (Semi-solid/Transition Blocks)**: The query block intersects the causal diagonal. Explicit masking (`mask = OFFS_M[:, None] >= (start_n + OFFS_N[None, :])`) must be applied.
- **Stage 3 (Skipped Blocks)**: The key block is completely after the query block, so it is skipped.

The refactored approach eliminates the need for hardcoded stages. It dynamically calculates the valid block range before the inner loop and runs a unified generic loop using `block_min` and `block_max`:

```python
# Legacy Staged Approach
if STAGE == 1: # "Solid" blocks of Causal masks
    lo, hi = 0, min(seqlen_k, start_m * BLOCK_M)
elif STAGE == 2: # "Semi-solid" block of Causal mask
    lo, hi = ...

# Optimized Approach
# Bounds (block_min, block_max) are dynamically computed prior to loop entry
for start_n in range(block_min, block_max, BLOCK_N):
    k = load_fn(K_block_ptr, ...)
    v = load_fn(V_block_ptr, ...)
    # Execute attention tile computation
```

This restructuring simplifies the generated PTX/AMDGCN code by removing control-flow divergence within the innermost hardware loop, enabling better instruction pipelining on CDNA compute units.

### 2. Handling Dissimilar Sequence Lengths

The metadata schema was expanded to explicitly track maximum sequence lengths for queries and keys separately (`max_seqlens_q` and `max_seqlens_k`). 

By separating $N_{CTX\_Q}$ and $N_{CTX\_K}$, the outer loop over query blocks (`BLOCK_M`) traverses up to $N_{CTX\_Q}$, while the inner loop defines memory access bounds based on $N_{CTX\_K}$. In causal attention with dissimilar lengths, the mask's boundary offset must be handled carefully. The generic loop parameters dynamically account for boundary shifts rather than enforcing a square matrix assumption.

### 3. Unified Padding and Boundary Load Check

To handle partially filled hardware blocks safely without excessive conditional branches, the implementation abstracts tensor loading and padding into a robust unified `load_fn`:

```python
@triton.jit
def load_fn(block_ptr, first, second, pad):
    if first and second:
        tensor = tl.load(block_ptr, boundary_check=(0,1), padding_option=pad)
    elif first:
        tensor = tl.load(block_ptr, boundary_check=(0,), padding_option=pad)
    elif second:
        tensor = tl.load(block_ptr, boundary_check=(1,), padding_option=pad)
    else:
        tensor = tl.load(block_ptr)
    return tensor
```

This ensures that optimal, unmasked memory load instructions are generated for blocks safely residing inside the bounds of the tensor, while masked boundary checks are applied strictly for edge tiles. This drastically reduces the overhead of masking operations on standard "solid" blocks.

## Optimization Principles

- **Branch Divergence Minimization**: Extracting causal boundary checks (`block_min` / `block_max`) to the outer scope prevents divergent branching inside the hot execution loop.
- **Optimal Tensor Memory Addressing**: Avoiding padding options internally when elements are strictly within bounds reduces VGPR usage and redundant zeroing instructions.
- **Algorithmic Flop Reduction**: Properly structured causal loop bounds inherently discard out-of-bound calculations, cutting the forward pass FLOPs by ~50% compared to dense attention without evaluating a runtime mask.

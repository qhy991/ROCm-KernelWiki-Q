---
id: technique-pr-triton-567
title: "MQA / GQA KV Head Index Calculation Fix in Triton"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - triton
  - attention
  - flash-attention
  - memory-bound
  - bandwidth
  - rocm-kernel
  - optimization
confidence: inferred
sources:
  - pr-triton-567
---

# MQA / GQA KV Head Index Calculation Fix in Triton

## Summary
PR #567 in the ROCm Triton repository addresses a critical index calculation bug for Key/Value (KV) heads when utilizing **Multi-Query Attention (MQA)** and **Grouped-Query Attention (GQA)**. Correct index mapping is necessary to ensure the proper attention heads are retrieved from the KV cache and to prevent out-of-bounds memory access.

## Context: Multi-Query and Grouped-Query Attention
In standard **Multi-Head Attention (MHA)**, the number of Query (Q) heads is equal to the number of Key (K) and Value (V) heads. This 1:1 mapping implies that for any head index $i$, $Q_i$ attends to $K_i$ and $V_i$.
During the decoding phase of Large Language Models (LLMs), attention is severely memory-bound due to the low arithmetic intensity of reading the entire KV cache from High-Bandwidth Memory (HBM) to compute a single token output.

To alleviate this bandwidth bottleneck, models employ **MQA** and **GQA**:
- **Multi-Query Attention (MQA)**: Uses a single shared KV head for all Q heads.
- **Grouped-Query Attention (GQA)**: Divides Q heads into groups, where each group shares a single KV head. 

In GQA, the number of Q heads is a multiple of the number of KV heads. Let $G$ be the group size (i.e., the number of Q heads sharing one KV head), given by $G = N_{Q} / N_{KV}$. The corresponding KV head index for a given Q head index $i$ must be computed as:
$$ \text{KV\_head\_idx} = i \mathrel{//} G $$

## The Defect and Implementation Details
In standard MHA Triton kernels, block pointers or memory offsets for Q, K, and V all commonly use the same head index (`q_head_idx == kv_head_idx`) derived from the grid mapping (e.g., `tl.program_id`). 

When adapting these kernels to support MQA/GQA, if the head index is not explicitly scaled down for the K and V block pointers, the kernel will:
1. **Read incorrect KV heads**, leading to corrupted attention outputs.
2. **Read out-of-bounds memory** if $i \ge N_{KV}$, causing hardware exceptions (memory faults).

### Triton Offset Calculation Correction
The fix correctly calculates the mapping from the Query head ID to the KV head ID using integer division. In Triton, this typically manifests during the setup of memory offsets:

```python
# Before (Standard MHA assumption)
head_idx = tl.program_id(1)
offset_q = ... + head_idx * stride_qh
offset_k = ... + head_idx * stride_kh  # Bug: Out of bounds for GQA
offset_v = ... + head_idx * stride_vh  # Bug: Out of bounds for GQA

# After (Corrected for MQA / GQA)
q_head_idx = tl.program_id(1)
group_size = num_q_heads // num_kv_heads
kv_head_idx = q_head_idx // group_size

offset_q = ... + q_head_idx * stride_qh
offset_k = ... + kv_head_idx * stride_kh
offset_v = ... + kv_head_idx * stride_vh
```

## Performance and Memory Bounds Impact
While primarily a correctness fix, proper GQA/MQA indexing has profound performance implications on AMD CDNA architectures:
- **HBM Bandwidth Reduction**: Sharing KV heads decreases the total amount of global memory traffic by a factor of the group size $G$. 
- **LDS and VGPR Efficiency**: Reduced unique KV loads allow for smaller Local Data Share (LDS) requirements and lower Vector General-Purpose Register (VGPR) pressure, improving theoretical warp occupancy and allowing larger sequence lengths to fit on-die.
- **Cache Hit Rates**: Shared KV reads increase L2 cache hit rates across CUs that process Query heads belonging to the same group.

## Best Practices for Triton Kernel Porting
When writing or porting Triton kernels (like FlashAttention) for ROCm:
1. **Decouple Q and KV grids**: Always treat Q head IDs and KV head IDs as distinct variables.
2. **Floor Division**: Use explicit floor division `tl.program_id(dim) // group_size` to ensure stable block resolution.
3. **Stride Validation**: Validate that memory strides multiplied by `kv_head_idx` fall within the allocated tensor bounds.

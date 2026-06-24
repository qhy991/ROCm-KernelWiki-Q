---
id: technique-pr-triton-551
title: "On-the-fly ALiBi Generation in Triton Flash Attention"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
kernel_types: [flash-attention]
languages: [triton-rocm]
tags:
  - optimization
  - memory
  - fused-kernel
  - rocm-kernel
confidence: inferred
sources:
  - pr-triton-551
---

# On-the-fly ALiBi Generation in Triton Flash Attention

## Context and Intent
Attention with Linear Biases (ALiBi) is a positional encoding method that adds a static, non-learned bias to the query-key dot product (the $QK^T$ matrix) before the softmax operation. The bias is directly proportional to the relative distance between the query and key tokens, scaled by a head-specific slope. 

Implementing this natively by pre-computing the bias and materializing an $N \times N$ matrix in global memory is extremely inefficient, scaling quadratically with sequence length $O(N^2)$. PR #551 in `ROCm/triton` implements ALiBi directly inside the Flash Attention forward pass kernel, calculating and applying these biases dynamically within SRAM (registers).

## Optimization Technique: Inline Computation

Instead of loading bias values from High Bandwidth Memory (HBM), the Triton kernel dynamically constructs the ALiBi bias values inside the innermost loop (`_attn_fwd_inner`).

### 1. Once-per-Block Head Slope Load
The head-specific slope is loaded just once per thread block prior to entering the attention loop. This limits the ALiBi memory footprint entirely to the `alibi_slopes` vector (size $Batch \times Heads$), effectively requiring only a single scalar read per attention block computation:
```python
if USE_ALIBI != 0:
    a_offset = off_z * stride_az + off_h_q * stride_ah 
    alibi_slope = tl.load(alibi_slopes + a_offset)
```

### 2. Relative Position Math
Inside the loop, the absolute relative position between queries and keys is formulated geometrically using `start_m` and `start_n` coordinates:
```python
global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
global_n_positions = start_n + tl.arange(0, BLOCK_N)

# Account for varying sequence lengths in the batch
relative_pos_block = global_m_positions[:,None] + actual_seqlen_k - global_n_positions[None,:] - actual_seqlen_q
relative_pos_block = tl.abs(relative_pos_block)
```

### 3. Log2(e) Base Scaling
Flash Attention in Triton generally calculates probabilities using base-2 exponentiation (`tl.math.exp2(qk)`) rather than natural exponentiation (`tl.math.exp`) because it is natively faster on the hardware. To ensure mathematical equivalence with standard ALiBi, the computed bias is scaled by $\log_2(e) \approx 1.442695$:
```python
alibi_block = -1 * alibi_slope * relative_pos_block
qk += (alibi_block * 1.44269504089)
```

## Memory and Performance Bounds

- **Memory Bound Avoidance:** By constructing ALiBi mathematically, the kernel avoids the $O(B \times H \times N_q \times N_k)$ memory bound that plagues un-fused ALiBi implementations. The only HBM reads are the $O(B \times H)$ `alibi_slopes`, representing an immense reduction in memory traffic.
- **ALU Utilization:** The operation trades a heavy memory load for a small number of inexpensive integer arithmetic instructions (add, subtract, abs) and FP multiplication. Given that Attention is inherently memory bound (or bounded by MFMA tensor-core compute for the QK / PV multiplications), these standard vector ALUs run "for free" in the background, not slowing down the dominant execution paths.
- **Register Pressure:** This technique requires additional local tile allocations (`global_m_positions`, `global_n_positions`, `relative_pos_block`) which map onto VGPRs. While minor, Triton compilers must accommodate the extra live ranges, which can marginally impact wavefront occupancy if register usage is hovering precisely on an occupancy boundary limit. However, fused bias accumulation ensures peak overall system efficiency.

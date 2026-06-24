---
id: technique-pr-triton-576
title: "ALiBi Integration in Triton Attention Backward Pass"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, optimization, memory-bound, fused-kernel, vgpr]
kernel_types: [attention, flash-attention]
languages: [triton-rocm]
confidence: inferred
sources: [pr-triton-576]
---

# Analysis of PR #576 in ROCm/triton: ALiBi Backward Pass

## Architectural Context and Intent

Pull Request #576 in `ROCm/triton` implements the backward pass for **Attention with Linear Biases (ALiBi)** within Triton-based attention kernels. ALiBi is a highly effective positional encoding technique that removes the need for explicit positional embeddings by adding a static, distance-proportional bias directly to the attention logits. 

In standard Flash Attention, the forward pass computes $O = \text{softmax}(QK^T + M)V$. For ALiBi, the mask $M$ incorporates the positional bias: $M_{i,j} = m \cdot (i - j)$, where $m$ is a head-specific scalar slope. Implementing the backward pass in a memory-efficient manner requires recomputing these biased attention scores on-the-fly to avoid materializing the $O(N^2)$ intermediate matrix in High Bandwidth Memory (HBM).

## Deep Technical Analysis

### 1. Memory Bounds and Recomputation
Flash Attention relies on exact recomputation of the attention matrix during the backward pass to drastically reduce HBM traffic. Without ALiBi fused into the kernel, users might have to explicitly materialize the bias matrix in memory, which would degrade performance from being compute-bound (or fast-memory bound) to strictly HBM bandwidth-bound.

By integrating ALiBi natively into the Triton backward kernel:
* **HBM Bandwidth Savings**: The $N \times N$ ALiBi bias matrix is never loaded.
* **On-the-fly Generation**: Bias values are generated inside the compute units (CUs) using fast integer arithmetic on the thread's tile indices (`row_idx` and `col_idx`).

### 2. Register Tiling and Occupancy (VGPR Pressure)
Incorporating ALiBi arithmetic in the innermost loop of the backward pass adds pressure to the Vector General Purpose Registers (VGPRs). CDNA architectures (CDNA2, CDNA3) provide massive throughput through Matrix Core (MFMA) instructions, but require careful register management to maximize wavefront occupancy.
* **Index Arithmetic**: Computing the relative distance $i - j$ requires tracking absolute sequence positions across blocks. This introduces auxiliary integer registers.
* **Broadcasting Slopes**: Each attention head has a distinct scalar slope $m$. This scalar must be broadcasted and multiplied across the tile.
* **Mitigation**: In Triton, the compiler schedules these element-wise bias additions closely with the `tl.dot` output accumulators to minimize the live range of the ALiBi tensor, keeping VGPR usage within acceptable limits for high occupancy.

### 3. Precision and Accumulator Flow
The core operation in the backward pass involves gradients of the attention scores ($dP$ and $dS$). 
* The ALiBi bias is applied to the raw $QK^T$ logits.
* Since the softmax derivative requires the softmax output from the forward pass, the recomputed logits must exactly match the forward pass, including the ALiBi shift. 
* The addition of the ALiBi penalty is typically performed in FP32 inside the MFMA accumulators before downcasting (if applicable) or before applying the softmax operator.

## AMD CDNA Implications
For CDNA hardware (MI250X, MI300X):
* **MFMA Scheduling**: The element-wise addition of the ALiBi bias must not pipeline-stall the dense MFMA instructions processing the GEMM portions ($QK^T$ and $dV$). Optimal scheduling hides the latency of the scalar-vector bias addition behind the MFMA latency.
* **LDS Utilization**: Since ALiBi slopes are small scalars per head, they can be kept in VGPRs or SGPRs rather than consuming valuable Local Data Share (LDS) capacity, keeping LDS fully available for Q, K, V, and their gradients.

## Summary of Optimizations
- **Fused-Kernel Architecture**: Keeps intermediate $N \times N$ bias states inside the register file.
- **Compute vs. Memory**: Trades a negligible amount of ALU compute (multiplication by slope and addition) for massive $O(N^2)$ HBM read/write reductions.
- **Gradient Stability**: Ensures numerical parity between forward and backward attention shapes by enforcing the ALiBi penalty correctly during $QK^T$ recomputation.

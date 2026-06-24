---
id: technique-pr-triton-563
title: "PR Insight: triton #563 - New Base Backwards Kernel"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, optimization, memory-bound, training]
kernel_types: [attention]
languages: [triton-rocm]
hardware_features: [lds, mfma]
confidence: inferred
sources:
  - pr-triton-563
---

# Analysis of PR #563 in triton: New Base Backwards Kernel

## Context and Intent
PR #563 introduces a foundational "base backwards kernel" for ROCm/triton. Backwards kernels compute the gradients required during the backpropagation phase of training. In Triton, these are heavily optimized custom kernels, typically for memory-bound operations like FlashAttention, MLPs, or specialized fused layers. 

The intent of this PR is architectural: establishing the core structure, tiling logic, and baseline memory movement strategies for a backwards pass without integrating advanced features yet. By separating the "base" implementation from subsequent "feature commits", the author ensures that the foundational scheduling—how work is divided among Compute Units (CUs) and wavefronts—is robust and stable.

## Architectural & Memory Analysis

Backwards kernels present unique challenges compared to forward kernels due to increased register pressure and memory bandwidth demands.

### Memory Bound Characteristics
1. **High HBM Traffic**: A typical backwards pass must load the input tensors (e.g., $Q, K, V$), the output tensor ($O$), the output gradient ($dO$), and auxiliary statistics (e.g., softmax $LSE$). It then computes and writes the input gradients ($dQ, dK, dV$). This heavy read/write ratio pushes the kernel deep into the **memory-bound** regime.
2. **LDS (Shared Memory) Utilization**: To mitigate HBM bottlenecks, the base kernel relies heavily on Local Data Share (LDS). Tensors are loaded into LDS in blocks (e.g., `BLOCK_M`, `BLOCK_N`), where MFMA (Matrix Fused Multiply-Add) instructions can consume them repeatedly for gradient accumulations. Proper padding is likely established here to avoid bank conflicts.
3. **Register Tiling**: AMD ROCm GPUs (like MI250X and MI300X) require careful VGPR (Vector General Purpose Register) allocation. The backwards kernel must balance keeping intermediate gradients in VGPRs versus spilling to LDS, as spilling directly limits occupancy and degrades performance.

### Optimization Techniques
- **Recomputation vs. Stashing**: To save memory bandwidth, base backwards kernels often recompute certain forward pass values (like attention scores) on the fly rather than reading them from HBM, exchanging compute cycles for memory savings.
- **Tiling Strategy**: The loop order (e.g., looping over the $K$ and $V$ sequence length in the outer loop while keeping $Q$ blocks in registers) is critical. The base kernel establishes this block-level scheduling.
- **MFMA Scheduling**: Matrix multiplications (like computing $dQ = dO \times V^T$) rely on optimal MFMA instruction interleaving. The base kernel sets up the shapes for these operations (e.g., using $32 \times 32$ or $16 \times 16$ MFMA blocks) suitable for CDNA architectures.

## Future Progression
As noted in the PR description ("Feature commits to follow"), this base kernel serves as a structural scaffold. Expected future enhancements might include:
- **Block-scale quantization** or FP8/FP6 format support (specifically for cdna4).
- **Persistent kernels** using Global Wave Sync (GWS) for cross-CU coordination.
- **Double buffering / Asynchronous memory copies** (`async-copy`) to overlap LDS data movement with MFMA compute.

## Reproducibility and Usage
Developers looking to build custom backwards kernels in `triton-rocm` should study this base implementation as a reference for handling standard memory layout, block synchronization, and gradient accumulation on AMD hardware.

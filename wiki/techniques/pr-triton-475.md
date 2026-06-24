---
id: technique-pr-triton-475
title: "PR Insight: triton #475 - FP8 QK Flash Attention Integration"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
kernel_types:
  - flash-attention
  - attention
languages:
  - triton-rocm
tags:
  - optimization
  - memory-bound
  - fp8
  - rocm-kernel
  - fused-kernel
  - flash-attention
confidence: inferred
sources:
  - pr-triton-475
---

# Analysis of PR #475 in ROCm/triton

## Summary
This PR introduces an FP8 Flash Attention (FA) implementation to the `06-fused-attention-fwd-transV.py` tutorial/kernel within ROCm's Triton fork. The crucial innovation is computing **only the first GEMM (`Q @ K^T`) in FP8**, while preserving higher precision for the subsequent softmax and second GEMM operations. 

## Architectural & Performance Analysis

### 1. Mixed-Precision Flash Attention
Flash Attention comprises two core Matrix Multiplications (GEMMs):
- **GEMM 1**: `S = Q @ K^T`
- **GEMM 2**: `O = P @ V` (where `P = softmax(S)`)

By restricting FP8 to the first GEMM, this implementation balances extreme performance with numerical stability:
- **Compute Optimization**: Matrix operations in FP8 have significantly higher throughput on architectures like CDNA3 (MI300 series) compared to FP16/BF16 hardware. Accelerating the first GEMM reduces the primary compute bottleneck.
- **Accuracy Preservation**: Softmax and the second GEMM (`P @ V`) are highly sensitive to quantization errors. `P` represents a probability distribution; quantizing it to FP8 typically destroys dynamic range. Retaining FP16/BF16 or FP32 for the second GEMM preserves output fidelity.

### 2. Memory Bandwidth & SRAM (LDS) Optimization
Flash Attention is fundamentally bounded by memory bandwidth and SRAM (LDS) capacity. 
- **Reduced Memory Traffic**: Loading `Q` and `K` as 8-bit floats halves the HBM bandwidth requirements for these tensors compared to 16-bit equivalents.
- **LDS Pressure Reduction**: Storing FP8 blocks in Local Data Share (LDS) consumes half the space. This reduction in shared memory footprint can enable larger tile sizes (e.g., larger `BLOCK_M` or `BLOCK_N`), which inversely reduces the number of outer loop iterations and minimizes redundant loads from HBM.

### 3. Transposed Value Matrix (`transV`)
The targeted file `06-fused-attention-fwd-transV.py` indicates that the Value matrix `V` is loaded in a transposed layout.
- In the second GEMM (`P @ V`), `P` is an `M x N` matrix and `V` is an `N x d` matrix. 
- Storing or loading `V` in a transposed format often aligns the contiguous memory dimension with the inner product dimension or aligns with native hardware matrix core instructions (MFMA on ROCm), thereby avoiding LDS bank conflicts during shared memory reads and improving memory coalescing.

### 4. ROCm / CDNA Implications
While CDNA2 (MI200) lacks native FP8 matrix cores, it supports packed int8 or custom software emulation depending on Triton's compiler backend mapping. On CDNA3 (MI300 series) and CDNA4, FP8 is a first-class citizen with native MFMA (Matrix-Fused-Multiply-Add) instructions delivering peak computational TFLOPS. This kernel serves as a critical optimization for unlocking the MI300's peak throughput for Large Language Model (LLM) serving and training.

## Conclusion
This PR represents a strategic mixed-precision optimization for Flash Attention on AMD GPUs. By targeting the `Q @ K^T` operation for FP8 acceleration, it minimizes memory traffic and maximizes compute throughput where it is safest, avoiding the catastrophic accuracy degradation associated with FP8 Softmax and Value accumulations.

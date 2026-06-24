---
id: technique-pr-triton-669
title: "Tianxing/fa int8: INT8 Flash Attention in Triton for ROCm"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - mfma
  - flash-attention
  - int8
  - quantization
  - memory-bound
  - rocm-kernel
  - optimization
confidence: inferred
sources:
  - pr-triton-669
---

# Analysis of PR #669: Tianxing/fa int8 in ROCm Triton

## Context and Intent
PR #669 in the `ROCm/triton` repository, titled "Tianxing/fa int8", introduces INT8 quantization support for Flash Attention in Triton. Flash Attention is a highly optimized algorithmic technique designed to compute the self-attention mechanism in Transformer models while minimizing High Bandwidth Memory (HBM) read/write operations by heavily utilizing the Local Data Share (LDS).

The primary intent of this implementation is to drastically reduce the memory footprint and alleviate the memory bandwidth bottleneck of Flash Attention by quantizing the input matrices (Q, K, V) into 8-bit integer formats (`INT8`). This compression is particularly impactful for Large Language Model (LLM) inference, effectively turning a memory-bound operation into a more compute-bound operation.

## Optimization Techniques

### 1. INT8 Quantization for Memory Bandwidth Relief
By scaling down from 16-bit floating-point (FP16/BF16) to 8-bit integers (INT8), the kernel halves the memory payload fetched from HBM per token. Since standard Flash Attention is fundamentally memory-bound at sequence lengths commonly seen in production, this 50% reduction in data size provides nearly a 2x theoretical boost to HBM bandwidth-limited throughput.

### 2. MFMA Instruction Utilization
AMD's Matrix Fused Multiply-Add (MFMA) instructions natively support INT8 data types. The Triton compiler's lowering passes translate the INT8 dot products in the attention computation into these specialized INT8 `v_mfma` instructions. On architectures like CDNA2 and CDNA3, INT8 matrix operations yield up to double the peak arithmetic throughput (TOPS) compared to FP16/BF16, accelerating the core block-level GEMM computations within the attention loop.

### 3. LDS and Register Tiling Optimization
Operating on INT8 data allows the kernel to fit twice as many elements into the same LDS allocation. This increased data density implies that larger block sizes (e.g., `BLOCK_M` and `BLOCK_N`) can be used during Triton's tile-based programming schedule without overflowing the physical LDS limits per Compute Unit (CU). The larger tiles inherently increase the kernel's arithmetic intensity, reducing the frequency of global memory fetches and further hiding latency.

## Performance and Memory Bounds

- **Memory Bound vs Compute Bound**: Traditional Attention is strictly memory-bound. Quantizing the inputs to INT8 shifts the bottleneck. While it still heavily taxes memory bandwidth for long sequences, the reduced IO overhead allows the kernel to hit much higher operational intensities, getting closer to the compute bounds of the GPU.
- **De-quantization Overhead**: To maintain numerical stability for the softmax operation, the accumulators are typically retained in Int32 or cast back to FP16/FP32. The scaling factors must be loaded alongside the input tiles to dequantize the values before the softmax function is applied. Managing these scale loads optimally is crucial to prevent them from becoming a new memory bottleneck.

## Architecture Specifics
- **CDNA3 (`gfx940`, `gfx942`)**: Benefiting from dual Compute Matrix Accelerators (dual-cma) per CU, CDNA3 executes INT8 MFMAs with immense throughput. This PR aligns perfectly with MI300X capabilities, utilizing its high INT8 instruction rate to maximize Flash Attention performance in inference environments.

## Conclusion
The addition of INT8 Flash Attention in ROCm Triton addresses the most severe bottleneck in modern LLM deployment: memory bandwidth. By compressing the Q, K, and V tensors and exploiting AMD's native INT8 matrix core capabilities, this PR acts as a key enabler for high-throughput, low-latency Transformer inference on CDNA GPUs.

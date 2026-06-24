---
id: technique-fa3-asynchrony-ck
title: "FlashAttention-3 Asynchronous Ping-Pong Buffering via CK"
type: wiki-technique
confidence: verified
architectures:
  - cdna3
kernel_types:
  - flash-attention
tags:
  - flash-attention-3
  - composable_kernel
  - asynchronous-execution
  - ping-pong-buffering
sources:
  - pr-flash-attention-rocm-117
---

# FlashAttention-3 Asynchronous Ping-Pong Buffering via CK

## The Evolution of FlashAttention-3
FlashAttention-2 bound the algorithm tightly to the GPU's SRAM (LDS in AMD parlance), minimizing HBM reads/writes. However, FA2 still suffered from pipeline bubbles where the Matrix Multiply units (MFMA/WMMA) would idle while waiting for the next block of Keys and Values to load from HBM into LDS.

FlashAttention-3 solves this by introducing aggressive **Asynchrony** and **Low-precision (FP8)**. 

## Composable Kernel (CK) Implementation on ROCm
To realize FA3 on AMD CDNA3 architecture, the `ROCm/flash-attention` backend relies on an updated Composable Kernel (CK) library.

### Ping-Pong LDS Scheduling
Instead of a monolithic blocking load, the CK backend partitions the Local Data Share (LDS) into dual buffers (Ping and Pong). 
- While the MFMA instructions are multiplying Query vectors against the **Ping** buffer of Keys...
- Asynchronous DMA instructions are simultaneously fetching the next block of Keys from HBM into the **Pong** buffer.
- This overlapping is synchronized via careful placement of `s_waitcnt vmcnt` and `s_wait_dscnt` intrinsics.

### Persistent Thread Blocks
To further reduce bubbles, the CK update facilitates persistent kernels. Instead of the hardware scheduler spinning up new waves for every tile, waves remain alive and atomically pull the next tile coordinate from a global queue. This ensures that the pipeline never drains and refills during the attention pass.

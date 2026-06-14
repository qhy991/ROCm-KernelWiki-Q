---
id: kernel-gemm-rocm
title: "GEMM Implementation on AMD CDNA"
type: wiki-kernel
architectures: [cdna1, cdna2, cdna3]
tags: [gemm, mfma, lds, register-tiling]
confidence: verified
sources: []
kernel_types: [gemm]
languages: [hip-cpp, ck-dsl]
reproducibility: snippet
---

# GEMM Implementation on AMD CDNA

General Matrix Multiplication (GEMM) on AMD CDNA architectures fundamentally relies on the Matrix Fused Multiply-Add (MFMA) instructions to achieve peak throughput.

## Hierarchical Tiling Strategy

A high-performance GEMM in HIP follows a strict tiling hierarchy:
1. **Grid Level**: The output matrix $C$ is divided into Thread Block Tiles. Each Compute Unit (CU) computes one or more of these tiles.
2. **Block Level**: The thread block loads tiles of $A$ and $B$ from Global Memory into the Local Data Share (LDS).
3. **Wavefront Level**: The 64-thread wavefront loads sub-tiles from LDS into Vector General-Purpose Registers (VGPRs).
4. **Instruction Level**: The `v_mfma` instructions multiply the registers and accumulate the result into a separate set of VGPRs.

## The MFMA Instruction (Matrix Core)

Unlike NVIDIA's `mma.sync` which requires a full warp, AMD's MFMA operates on a single Wavefront (64 threads). 

Example intrinsic in HIP:
```cpp
// 32x32x8 FP16 -> FP32 MFMA
D_reg = __builtin_amdgcn_mfma_f32_32x32x8f16(A_reg, B_reg, C_reg, cbsz, abid, blgp);
```

### Register Pressure

MFMA outputs (the C matrix accumulators) require a large number of VGPRs. For a `32x32` block, a single wave requires 32 VGPRs just for the C accumulators (FP32). Balancing the tile size to maximize MFMA throughput while avoiding register spilling is the primary challenge in ROCm GEMM tuning.

## Double Buffering & Prefetching

To hide the latency of Global Memory and LDS reads, ROCm GEMM implementations use double buffering:
- Load $A_{k+1}$ and $B_{k+1}$ into LDS while computing MFMA on $A_k$ and $B_k$.
- Use `__builtin_amdgcn_s_waitcnt` to control synchronization explicitly, avoiding over-synchronization.

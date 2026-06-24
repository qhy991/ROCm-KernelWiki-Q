---
id: pr-triton-571
title: "Deep Analysis of Triton Perf Kernels Suite (PR #571)"
type: source-pr
repo: ROCm/triton
pr: 571
author: vgokhale
date: '2024-05-06'
url: https://github.com/ROCm/triton/pull/571
source_category: upstream-code
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, triton, rocm, memory-bound, bandwidth, hbm, library, scheduling]
kernel_types: [gemm, attention, flash-attention]
languages: [triton-rocm, python]
captured_at: '2026-06-23'
status: merged
inclusion_reason: "Comprehensive suite of optimized Triton kernels for AMD GPUs"
confidence: inferred
---

# Deep Analysis of Triton Perf Kernels Suite (PR #571)

## Overview

PR #571 in `ROCm/triton` updates the documentation (`python/perf-kernels/README.md`) for a suite of highly-optimized Triton kernels tailored for AMD Instinct series GPUs (CDNA2/CDNA3/CDNA4). While this specific PR is a documentation update, it reveals the architecture and availability of several critical machine learning kernels optimized natively in Triton for AMD architectures. 

These performance kernels represent reference implementations of state-of-the-art optimization techniques, targeting both memory bandwidth and computational bottlenecks across different stages of large language model (LLM) execution.

## Featured Kernels and Architectural Analysis

### 1. Flash Attention (`flash-attention.py`)
This script implements a Triton-native forward kernel for Flash Attention v2.

- **Intent**: To optimize the standard attention mechanism by avoiding large intermediate $N \times N$ materializations in HBM (High Bandwidth Memory).
- **Optimization Techniques**: 
  - **Memory-Bound Alleviation**: Utilizing an online softmax calculation and tiling techniques, the kernel heavily utilizes LDS (Local Data Share) to fuse the $Q \times K^T$ and $P \times V$ operations.
  - **Flexibility**: Supports arbitrary sequence lengths for Queries (Q) and Key/Values (KV), as well as arbitrary head dimensions. 
  - **Advanced Features**: Integrates Autoregressive (causal) masking, Multi-Query Attention (MQA), Grouped-Query Attention (GQA), ALiBi biases, and custom Matrix bias.
- **Architectural Match**: Flash Attention directly benefits from maximizing the usage of the CDNA compute units' massive VGPR files and LDS memory capabilities to keep intermediate attention matrices on-chip.

### 2. Flash Decoding (`06-attention-decode.py`)
A specialized variant of the attention kernel optimized specifically for the decoding phase of autoregressive generation.

- **Intent**: In the decoding phase, standard Flash Attention underutilizes the GPU because the sequence length of the query is 1 (or very small), resulting in limited parallelization across the batch/head dimensions.
- **Optimization Techniques**:
  - **Split-K / Sequence Parallelism**: Flash Decoding splits the keys and values across the sequence dimension, computing partial attention outputs and intermediate log-sum-exp values across different thread blocks (or waves).
  - **Occupancy Tuning**: This significantly increases the active waves/CUs when processing long context with small batch sizes, overcoming compute-unit starvation.
- **Memory Bounds**: Decoding is highly memory-bound on KV cache reads. By splitting the work across the sequence dimension, it maximizes instantaneous HBM read bandwidth.

### 3. Stream-K GEMM (`03-matrix-multiplication-stream-k.py`)
A GEMM (General Matrix Multiplication) kernel implementing the Stream-K scheduling algorithm.

- **Intent**: Standard tiled GEMM algorithms (like Tile-K) often suffer from "tail effects" or quantization inefficiencies where the grid size doesn't perfectly match the number of available SMs (or CUs on AMD GPUs), leading to idle compute resources.
- **Optimization Techniques**:
  - **Workload Distribution**: Stream-K uniformly divides the total iterations of the GEMM loop across all available CUs, regardless of the matrix dimensions.
  - **Partial Tiles**: Threads compute contiguous fractional chunks of the output. If a CU finishes its chunk, it seamlessly starts processing the next logical chunk, even if it spans across output tiles.
- **Performance Impact**: Stream-K achieves near-perfect CU utilization and predictable execution times, highly beneficial for odd-sized matrices where standard tiling creates severe load imbalance.

### 4. Multi-Datatype GEMM (`03-matrix-multiplication-all-types.py`)
A comprehensive GEMM kernel demonstrating Triton's datatype support for AMD GPUs.

- **Supported Types**: Includes legacy datatypes (`int32`, `fp32`), common mixed-precision types (`fp16`, `bf16`), integer types (`int8`), and next-generation 8-bit floating point formats (`f8 e5m2` and `f8 e4m3`).
- **Hardware Integration**: The kernel relies heavily on AMD's Matrix Fused Multiply-Add (`mfma`) hardware features. The support for `f8` datatypes indicates direct targeting of CDNA3/CDNA4 architectures which natively support fp8 precision for increased throughput and reduced register pressure.

### 5. HBM Bandwidth Benchmark (`hbm-bw-test.py`)
A baseline script to measure peak achievable HBM bandwidth.

- **Intent**: Diagnosing hardware health and providing an empirical roofline limit for memory-bound kernels.
- **Implementation**: Typically executes vectorized loads and stores with minimal arithmetic operations to saturate the memory controllers and measure peak VRAM throughput in GB/s.

## Conclusion

The `perf-kernels` directory in the `ROCm/triton` repository acts as a blueprint for optimizing deep learning workloads on AMD Instinct GPUs. By providing reference implementations for Stream-K scheduling, Flash Decoding, and diverse precision GEMMs, AMD exposes the necessary building blocks for maximizing hardware utilization. The overarching theme is mitigating memory bandwidth constraints (via Flash Attention/Decoding) and solving compute load-imbalance (via Stream-K scheduling).

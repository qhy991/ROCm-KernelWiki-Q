---
id: lang-hip-cpp
title: "HIP C++ Programming Guide"
type: wiki-language
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [hip-cpp]
confidence: verified
sources: []
---

# HIP C++ Programming Guide

HIP (Heterogeneous-Compute Interface for Portability) is AMD's primary C++ dialect and runtime API for GPU programming. It is designed to be syntactically similar to CUDA C++, allowing developers to write code that can be compiled for both AMD ROCm and NVIDIA platforms.

## Core Concepts

1. **Host vs Device Code**: Like CUDA, HIP separates code executed on the CPU (`__host__`) from code executed on the GPU (`__device__`, `__global__`).
2. **Execution Configuration**: Kernel launches use the `<<<blocks, threads, shared_mem, stream>>>` syntax.
3. **Memory Management**: HIP provides `hipMalloc`, `hipFree`, and `hipMemcpy` APIs that closely mirror `cudaMalloc`, `cudaFree`, and `cudaMemcpy`.

## The `hipcc` Compiler

`hipcc` is the compiler driver for HIP. Under the hood, it uses Clang/LLVM to compile HIP C++ code into AMD GPU machine code (AMDGCN ISA).

```bash
# Compiling a HIP source file
hipcc -O3 --offload-arch=gfx90a my_kernel.cpp -o my_kernel
```

## Hardware Mapping

In HIP C++ on AMD CDNA:
- `threadIdx.x` maps to a single lane.
- A `warp` (NVIDIA terminology) is called a **Wavefront** and consists of 64 threads on AMD.
- `__shared__` memory maps to the Local Data Share (LDS).

## Key Differences from CUDA

While syntax is similar, the underlying hardware behavior differs:
- **Warp Size**: 64 (AMD) vs 32 (NVIDIA). Code assuming `warpSize == 32` will break or perform poorly. Always use the `warpSize` macro.
- **Matrix Cores**: AMD uses MFMA instructions instead of NVIDIA's Tensor Cores (`mma.sync`). These operate on the entire 64-thread wavefront.

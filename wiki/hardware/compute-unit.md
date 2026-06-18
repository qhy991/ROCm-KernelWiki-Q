---
id: hw-compute-unit
title: Compute Unit (CU) Microarchitecture
type: wiki-hardware
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [hardware, compute-unit, wavefront, mfma, lds]
confidence: source-reported
hardware_features: [compute-unit, wavefront, mfma, lds]
related: []
sources: []
cuda_equivalent: streaming_multiprocessor
---

# Compute Unit (CU) Microarchitecture

The Compute Unit (CU) is the fundamental building block of AMD GPU architectures, directly analogous to the NVIDIA Streaming Multiprocessor (SM). In the CDNA architecture family (powering the MI100, MI250X, MI300X, and MI350X), the CU is heavily optimized for dense compute, high-bandwidth memory operations, and massive matrix math throughput.

## Overview and Execution Model

Unlike the RDNA architecture which uses Workgroup Processors (WGPs) as the primary scheduling block, the CDNA architecture retains the **Compute Unit (CU)** as its atomic compute structure. A single CU contains all the necessary resources to schedule, execute, and retire multiple concurrent thread blocks (workgroups).

In the CDNA execution model, threads are grouped into **Wavefronts**. CDNA exclusively uses a `wave64` execution mode, meaning each wavefront consists of exactly 64 threads. 

### SIMD Layout

The arithmetic logic units inside a CDNA CU are divided into multiple SIMD (Single Instruction, Multiple Data) arrays:

*   **4x SIMD16 Arrays:** Each CU contains four 16-wide SIMD execution units. 
*   **Execution Cycles:** Because a wavefront contains 64 threads, and the SIMD width is 16, it takes **4 clock cycles** to issue a single vector instruction for an entire wavefront (`64 / 16 = 4`). 
*   **Wavefront Pools:** Each SIMD unit maintains its own pool of wavefronts. The CU scheduler cycles through these wavefronts, issuing instructions to the SIMD units to hide memory and instruction latencies.

## Hardware Components

A CDNA Compute Unit consists of several specialized ALUs and memory structures designed to operate in parallel.

### 1. Vector ALU (VALU)
The VALU is responsible for executing vector instructions where each thread in the wavefront has unique data. This includes standard floating-point operations (`v_add_f32`, `v_fma_f32`), integer operations, and type conversions. The 4x SIMD16 units primarily comprise the VALU.

### 2. Scalar ALU (SALU)
A distinguishing feature of AMD architectures is the dedicated **Scalar ALU**. 
*   **Uniform Execution:** The SALU handles operations that are uniform across the entire wavefront, such as control flow (branching), loop counters, and base address calculations.
*   **Parallel Issue:** SALU instructions (e.g., `s_add_u32`, `s_mul_i32`) are issued and executed independently of the VALU. This allows the CU to perform address arithmetic and control flow without stealing cycles from the VALUs performing heavy math.

### 3. Matrix Core (MFMA)
Introduced in CDNA1, Matrix Fused Multiply-Add (MFMA) units accelerate dense matrix multiplications. 
*   **CDNA1/CDNA2:** The CU contains dedicated MFMA units capable of executing massive cross-lane matrix instructions (e.g., `v_mfma_f32_32x32x8f16`), significantly boosting FLOPS for AI and HPC workloads.
*   **CDNA3:** Introduces the Dual Compute Matrix Accelerator (CMA) per CU, further scaling matrix operation throughput and introducing new data types (like FP8/BF8).

### 4. Local Data Share (LDS)
The Local Data Share (LDS) is the AMD equivalent of CUDA Shared Memory. 
*   **Size:** Each CU is equipped with **64 KB** of dedicated LDS. 
*   **Throughput:** LDS provides ultra-low latency, high-bandwidth software-managed caching for data shared among threads within the same workgroup. 
*   **Instructions:** Accessed via specific LDS instructions like `ds_read_b128` (reading 128 bits per lane) and `ds_write_b128`. Avoiding bank conflicts in the 32-bank LDS is critical for achieving peak memory throughput.

### 5. Register Files
To support a massive number of in-flight wavefronts and hide memory latency, the CU contains massive register files:
*   **Vector General-Purpose Registers (VGPRs):** Up to 512 32-bit VGPRs per thread (arch-dependent), capable of holding hundreds of kilobytes of state per CU. High VGPR usage reduces occupancy (number of active waves) but is often necessary for matrix workloads (e.g., Register Tiling).
*   **Scalar General-Purpose Registers (SGPRs):** Dedicated registers for the SALU, storing base pointers and constants, relieving pressure on the VGPR file.

## Scheduling and Occupancy

The Compute Unit schedules wavefronts to maximize hardware utilization. When a wavefront executes a high-latency instruction (such as a `buffer_load_dwordx4` from global memory or waiting on an MFMA instruction to complete), the scheduler immediately switches to another active wavefront in the pool.

### Occupancy Limits
Occupancy (the ratio of active wavefronts to the maximum supported by the CU) is bottlenecked by three primary resources:
1.  **VGPR Allocation:** Kernels allocating many VGPRs (e.g., >128 VGPRs per thread) will limit how many wavefronts can reside on the CU.
2.  **LDS Usage:** Since the CU only has 64 KB of LDS, a workgroup requesting 32 KB of LDS will limit the CU to hosting at most 2 workgroups simultaneously.
3.  **Workgroup Size:** The number of threads per workgroup dictates how many wavefronts are scheduled together.

## Code Example: HIP C++ Mapping to CU Resources

When writing HIP C++ code, the compiler maps software constructs directly to the CU's physical resources:

```cpp
#include <hip/hip_runtime.h>

// Launch configuration determines CU occupancy
// e.g., blockDim.x = 256 -> 4 Wavefronts per Workgroup
__global__ void cu_architecture_demo(const float* __restrict__ in, float* __restrict__ out) {
    // 1. Thread and Wavefront IDs
    int tid = threadIdx.x; // Maps to a specific lane in the 4x SIMD16 (0-63 for wave64)
    int bid = blockIdx.x;  // Maps to a Workgroup running on a specific CU
    
    // 2. Shared Memory allocates from the 64KB CU LDS
    __shared__ float lds_scratchpad[1024]; 
    
    // 3. Uniform operations compile to SALU instructions (e.g., s_add_u32)
    // The base address calculation is done by the SALU once per wave
    int global_idx = bid * blockDim.x + tid;
    
    // 4. Memory loads compile to Vector Memory instructions (e.g., global_load_dword)
    float val = in[global_idx]; 
    
    // 5. Vector math compiles to VALU instructions (e.g., v_add_f32)
    // Executes over 4 cycles across the SIMD16 unit
    val = val * 2.0f + 1.0f; 
    
    // 6. LDS write (ds_write_b32)
    lds_scratchpad[tid] = val;
    
    // 7. Barrier synchronizes all wavefronts in the Workgroup running on this CU
    __syncthreads(); 
    
    out[global_idx] = lds_scratchpad[tid];
}
```

## Performance Implications

Understanding the CU microarchitecture is vital for kernel optimization on ROCm:
*   **Vectorization:** Using `float4` or `int4` data types compiles to 128-bit memory instructions (`buffer_load_dwordx4`), maximizing memory bus utilization.
*   **Register Spilling:** Exceeding VGPR limits causes the compiler to "spill" registers to slow global memory (scratch memory), drastically reducing performance.
*   **Latency Hiding:** Kernels must have sufficient occupancy (enough active wavefronts) to keep the 4x SIMD16 units and MFMA units busy while other wavefronts wait on global memory fetches.

---
id: hw-dual-cma
title: Dual CMA (Compute Matrix Array) Engines in CDNA4
type: wiki-hardware
architectures: [cdna3, cdna4]
tags: [hardware, mfma, fp8, fp16, compute, isa, mi300x]
confidence: source-reported
hardware_features: [dual-cma, mfma, scaled-mfma]
related: []
sources: []
cuda_equivalent: null
---

# Dual CMA (Compute Matrix Array) Engines

The **Dual CMA (Compute Matrix Array)** engine is a major architectural enhancement introduced in the AMD CDNA™ architecture, specifically maturing in CDNA4 (MI350X series) to massively accelerate matrix multiplication operations. By fundamentally changing how the matrix arithmetic logic units (ALUs) are organized within the Compute Unit (CU), the Dual CMA design doubles the MFMA (Matrix Fused Multiply-Add) throughput for low-precision formats such as FP8 and FP16 compared to the preceding CDNA3 architecture (MI300X).

## Architecture Overview

In the AMD GPU execution model, the core computational workhorse is the Compute Unit. A CDNA Compute Unit issues instructions to both vector ALUs (for pointwise operations) and Matrix Cores (for tensor operations). In CDNA3, each CU possessed a unified Compute Matrix Array that processed `v_mfma` instructions. 

With CDNA4, the architecture splits and expands this resource into **Dual CMA engines** per CU. This dual configuration allows a single CU to dispatch and execute twice as many matrix operations per clock cycle for specific precisions. 

### CDNA3 vs CDNA4 Matrix Throughput per CU

| Precision | CDNA3 (MI300X) Ops / Clock / CU | CDNA4 (MI350X) Ops / Clock / CU | Improvement |
|-----------|---------------------------------|---------------------------------|-------------|
| **FP64**  | 128                             | 128                             | 1x          |
| **FP32**  | 256                             | 256                             | 1x          |
| **FP16 / BF16** | 1024                      | 2048                            | **2x**      |
| **FP8 / BF8**   | 2048                      | 4096                            | **2x**      |
| **FP6 / FP4**   | N/A                       | 8192 (with block scaling)       | New         |

*Note: Theoretical dense throughput. Sparse operations can further double these figures.*

Because of the Dual CMA, CDNA4 achieves a drastic generational leap for inference and training workloads that heavily rely on FP16, BF16, and FP8 precision, resolving the compute bottleneck in Large Language Model (LLM) serving.

## Instruction Set Architecture (ISA) Details

The Dual CMA engines are exposed to the programmer via the `v_mfma` (Matrix Fused Multiply-Add) family of instructions. Under the hood, the hardware schedules these instructions across the two arrays to maximize utilization.

### ISA Register Usage
When programming for Dual CMA, understanding register allocation is critical. A typical 32x32x8 FP16 MFMA instruction (`v_mfma_f32_32x32x8f16`) requires:
- **Input A (Matrix A):** 4 VGPRs (containing 8 FP16 elements per thread across a wavefront)
- **Input B (Matrix B):** 4 VGPRs
- **Output (Accumulator C/D):** 16 VGPRs (for FP32 accumulation) or 8 VGPRs (for FP16 accumulation)

In CDNA4, due to the doubled throughput, kernel developers must aggressively tile their computations. Using the new `scaled-mfma` instructions for FP8/FP6 (e.g., `v_mfma_f32_16x16x32_f8f6f4`), the register footprint remains similar per instruction, but the instruction issue rate can be twice as fast. This puts immense pressure on the 256 VGPRs available per SIMD and requires careful **occupancy tuning** and **register tiling**.

```asm
// CDNA3: Single issue per cycle (conceptual)
v_mfma_f32_16x16x16f16 v[0:3], v[4:7], v[8:11], v[0:3] 

// CDNA4: Dual CMA can process two such instructions or a doubled-width instruction in the same time
v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[4:7], v[8:11], v[0:3]
```

## Performance Implications and Tuning

### 1. Register File and LDS Pressure
With the ALUs digesting FP16 and FP8 at twice the speed, the data feeding mechanism becomes the bottleneck. The **Local Data Share (LDS)** bandwidth and **VGPR** read/write bandwidth must be optimized.

- **CDNA4 LDS capacity:** 64KB per CU.
- Developers must use `ds_read_b128` (16-byte vectorized loads) to feed the Dual CMA engines fast enough.
- The **LDS-transpose** feature in CDNA4 is often paired with Dual CMA. It allows data to be read from LDS and transposed on-the-fly without requiring additional ALU instructions, saving register space and cycles.

### 2. Code Example: HIP C++ using Dual CMA efficiently
To leverage the Dual CMA, you don't typically write new code—the hardware executes existing `__builtin_amdgcn_mfma_*` builtins faster. However, block sizes and unroll factors should be increased to feed both arrays.

```cpp
#include <hip/hip_runtime.h>

__global__ void dual_cma_gemm_fp16_kernel(const half* A, const half* B, float* C) {
    // 32x32x8 FP16 MFMA built-in
    // Accumulator requires 16 FP32 registers (v[0:15])
    float c_acc[16] = {0.0f}; 
    
    // In CDNA4, this inner loop executes effectively twice as fast
    // Developers should unroll 2x or 4x more than in CDNA3 to hide latency
    #pragma unroll 4
    for(int k = 0; k < K_DIM; k += 8) {
        // Load A and B from LDS (assumed 128-bit loads)
        // A single ds_read_b128 fetches 8 half-precision values
        half4 a_val = load_matrix_A_from_lds();
        half4 b_val = load_matrix_B_from_lds();
        
        // __builtin_amdgcn_mfma_f32_32x32x8f16
        // Utilizing Dual CMA under the hood
        // The hardware will dispatch back-to-back mFMA ops to both CMA arrays
        c_acc = __builtin_amdgcn_mfma_f32_32x32x8f16(
            a_val, b_val, c_acc, 
            0, 0, 0 // CB, CBSZ, ABID modifiers
        );
    }
    
    // Store C...
}
```

## Summary
The Dual CMA architecture in CDNA4 represents a massive structural shift toward low-precision math density. By doubling the FP8 and FP16 throughput per CU relative to CDNA3 (MI300X), it fundamentally shifts the performance bottleneck from compute to memory bandwidth and register allocation. Developers porting from CDNA3 to CDNA4 should double their inner loop unrolling and utilize vectorized `ds_read_b128` loads and LDS-transpose features to keep the Dual CMA engines fully fed.

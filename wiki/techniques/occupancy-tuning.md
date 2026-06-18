---
id: technique-occupancy-tuning
title: Wavefront Occupancy Tuning
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [occupancy, vgpr, memory, memory-bound, optimization, bandwidth, mi300x, compute]
confidence: source-reported
techniques: [register-tiling, bank-conflict-padding]
hardware_features: [wavefront, vgpr, lds, compute-unit]
kernel_types: [gemm, attention, flash-attention, reduction]
related: []
sources: []
reproducibility: snippet
---

# Wavefront Occupancy Tuning

**Wavefront Occupancy** refers to the number of active wavefronts executing concurrently on a Compute Unit (CU) relative to the hardware's maximum capacity. While traditionally high occupancy is considered critical to hide memory latency (Thread-Level Parallelism or TLP), modern CDNA architectures running dense math (like GEMM or FlashAttention) often achieve peak performance at **low occupancy** by maximizing Register-Level Parallelism (Instruction-Level Parallelism or ILP).

This guide covers how to balance Vector General Purpose Registers (VGPRs), Local Data Share (LDS), and wavefront counts on AMD CDNA hardware.

## 1. Hardware Limits (CDNA2 & CDNA3)

To tune occupancy, it is critical to understand the physical resource limits per Compute Unit (CU) on MI250X and MI300X architectures:

| Resource | MI250X / MI300X Limit | Notes |
| :--- | :--- | :--- |
| **SIMDs per CU** | 4 | Each CU has 4 SIMD units executing instructions independently. |
| **Max Waves per SIMD** | 8 | The absolute hardware limit for active waves. |
| **Max Waves per CU** | 32 | Calculated as `4 SIMDs × 8 waves`. |
| **VGPR File Size** | 512 KiB per CU | Divided into 128 KiB per SIMD. Equal to 32,768 32-bit registers per SIMD. |
| **Max VGPRs per Thread**| 512 | A single thread can address up to 512 VGPRs. |
| **LDS Capacity** | 64 KiB per CU | Shared across all wavefronts running on the same CU. |

## 2. The Occupancy Math

Effective occupancy is determined by the most constraining resource required by your kernel: VGPRs, LDS, or the hardware limit.

### VGPR-Limited Occupancy
Each SIMD has 32,768 VGPRs. Since a wavefront consists of 64 threads, the maximum number of waves per SIMD based on VGPR usage is:
```text
Waves_per_SIMD_VGPR = floor(32768 / (VGPRs_per_thread * 64))
                    = floor(512 / VGPRs_per_thread)
```
- **If your kernel uses 256 VGPRs/thread**: `floor(512 / 256) = 2` waves per SIMD (8 waves per CU).
- **If your kernel uses 128 VGPRs/thread**: `floor(512 / 128) = 4` waves per SIMD (16 waves per CU).
- **If your kernel uses 64 VGPRs/thread**: `floor(512 / 64) = 8` waves per SIMD (32 waves per CU).

> [!NOTE]
> Registers are allocated in discrete granular blocks (typically 8 or 16 registers). A kernel using 129 registers might be rounded up to the next block boundary, potentially dropping occupancy down to 2 waves per SIMD.

### LDS-Limited Occupancy
LDS is allocated per workgroup (block in CUDA). The number of workgroups that can reside on a CU is:
```text
Workgroups_per_CU = floor(64 KiB / LDS_per_workgroup)
```
The resulting wave occupancy is:
```text
Waves_per_CU_LDS = Workgroups_per_CU * Waves_per_workgroup
```
For example, if a workgroup requires 32 KiB of LDS and contains 4 waves, the CU can fit `floor(64 / 32) = 2` workgroups, resulting in `2 * 4 = 8` waves per CU.

### Final Occupancy
The final achievable waves per CU is simply:
```text
Min( Waves_per_SIMD_VGPR * 4, Waves_per_CU_LDS, 32 )
```

## 3. The Balance: Occupancy vs. Register Tiling

On CDNA, maximizing occupancy is **not** the universal goal. 

For **compute-bound kernels** (like GEMM and Matrix Core operations):
- Developers aggressively use **Register Tiling**. By loading large tiles of matrices directly into VGPRs, they drastically reduce expensive LDS reads/writes.
- This pushes VGPR usage to the maximum (e.g., 256 or 512 VGPRs per thread).
- **Result:** Occupancy drops to 1 or 2 waves per SIMD. Memory latency is hidden by issuing independent `v_mfma` (Matrix Fused Multiply-Add) and global load instructions, relying entirely on the compiler and hardware scheduler to exploit ILP.

For **memory-bound kernels** (like Softmax, LayerNorm, or element-wise ops):
- Register tiling provides little benefit.
- The goal is to keep VGPR usage low (e.g., under 64 or 128) to achieve 4–8 waves per SIMD.
- **Result:** High TLP allows the GPU to instantly swap to a different wavefront while waiting for HBM (High Bandwidth Memory) fetches.

## 4. Optimization Strategies

### 1. Controlling Compiler Register Allocation
Use the `__launch_bounds__` attribute to force the compiler to spill or optimize registers to meet a target occupancy.

```cpp
// Hints the compiler to constrain VGPR usage to allow at least 
// (256 threads / 64) = 4 waves per workgroup, and minimum 2 workgroups per CU.
// This aims for 8 waves per CU (2 waves per SIMD).
__global__ void __launch_bounds__(256, 2) 
my_gemm_kernel(float* A, float* B, float* C) {
    // ...
}
```

### 2. Tuning LDS Padding
Bank conflicts in LDS can destroy performance. Developers often pad shared memory arrays. However, padding increases the `LDS_per_workgroup`.
- If your workgroup uses 32 KiB, you can fit 2 workgroups per CU.
- If padding pushes it to 32.5 KiB, you can now only fit **1 workgroup per CU**, instantly halving your occupancy.
> [!TIP]
> Carefully analyze if the cost of dropping occupancy (due to crossing an LDS size boundary) is worth the elimination of bank conflicts.

### 3. Monitoring Resource Usage
Always inspect the compiled kernel to see the exact VGPR, SGPR, and LDS usage. Add `-Rpass-analyze=kernel-resource-usage` to your HIP compiler flags:

```bash
hipcc -O3 -Rpass-analyze=kernel-resource-usage my_kernel.cpp
```
Output will look like:
```text
remark: my_kernel: VGPRs: 128, SGPRs: 32, ScratchSize: 0, Occupancy: 4 waves/SIMD
```

### 4. Grouping Work (Triton / CK)
In higher-level frameworks like Triton or Composable Kernel (CK), tuning block sizes (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`) directly impacts both VGPR and LDS consumption. 
- Larger block sizes require more VGPRs and LDS, lowering occupancy but increasing arithmetic intensity.
- Benchmarking a sweep of block sizes is the standard methodology for finding the optimal point on the Occupancy/ILP curve for a given matrix size.

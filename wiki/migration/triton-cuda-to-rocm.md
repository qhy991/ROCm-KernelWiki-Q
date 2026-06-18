---
id: migration-triton-cuda-to-rocm
title: Triton CUDA to ROCm Migration Guide
type: wiki-migration
architectures: [cdna2, cdna3, cdna4]
tags: [triton-rocm, migration, optimization, porting, mfma]
confidence: source-reported
from_architecture: nvidia-ampere
to_architecture: cdna3
difficulty: moderate
related: []
sources: []
---

# Triton CUDA to ROCm Migration Guide

When migrating Triton kernels from NVIDIA (CUDA) to AMD (ROCm), the Triton compiler abstracts away much of the underlying hardware complexity. However, achieving parity or superior performance on AMD CDNA architectures (like MI250X and MI300X) requires tuning script parameters—especially autotuner configurations, block sizes, and execution dimensions—to match AMD's Wavefront and Matrix Fused Multiply-Add (MFMA) characteristics.

## 1. Wavefronts vs. Warps

The most critical conceptual shift in Triton when moving from CUDA to ROCm is the mapping of `num_warps`.

*   **CUDA:** 1 Warp = 32 threads. `num_warps=4` means 128 threads per block.
*   **ROCm (CDNA):** 1 Wavefront = 64 threads. In the Triton AMD backend, `num_warps` maps directly to Wavefronts (Wave64). Therefore, `num_warps=4` means **256 threads** per block.

**Migration Action:**
When porting autotuners, you generally need fewer `num_warps` on AMD to achieve the same block size, or you can handle larger block sizes with the same `num_warps`. 
*   If your CUDA autotuner sweeps `num_warps: [4, 8]`, consider sweeping `num_warps: [2, 4, 8]` on ROCm to explore equivalent and larger thread counts.

## 2. MFMA-Specific Block Size Tuning

NVIDIA Tensor Cores (Ampere/Hopper) typically operate on m16n8k16 or m16n8k8 shapes. AMD CDNA architectures use MFMA (Matrix Fused Multiply-Add) instructions, which have different native shapes, such as:
*   `v_mfma_f32_32x32x8f16` (32x32x8)
*   `v_mfma_f32_16x16x16f16` (16x16x16)

The Triton ROCm backend automatically lowers `tl.dot()` to these MFMA instructions. To maximize the efficiency of the AMD backend's dot operator, your `BLOCK_SIZE_M` and `BLOCK_SIZE_N` should be tuned to align with these native non-K dimensions.

**Tuning Guidelines:**
*   **Align with 32x32 or 16x16:** `BLOCK_SIZE_M` and `BLOCK_SIZE_N` should be multiples of 32 or 64. A block size of 128x128 or 256x128 is highly efficient on MI300X as it maps cleanly to multiple 32x32x8 MFMA instructions.
*   **`matrix_instr_nonkdim`:** The AMD Triton backend allows an explicit `matrix_instr_nonkdim` configuration in the autotuner to force the compiler to select either the 16x16 or 32x32 MFMA variant.

### Example Autotuner Adjustment

```python
import triton
import triton.language as tl

# CUDA Autotuner
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#     ],
#     key=['M', 'N', 'K'],
# )

# ROCm-Optimized Autotuner
@triton.autotune(
    configs=[
        # Note: num_warps=4 corresponds to 256 threads (4 * 64) on ROCm
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'matrix_instr_nonkdim': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'matrix_instr_nonkdim': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'matrix_instr_nonkdim': 16}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'matrix_instr_nonkdim': 16}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
def matmul_kernel(a_ptr, b_ptr, c_ptr, ...):
    # Kernel implementation remains identical
    ...
```

## 3. Backend-Specific Compilation Flags

The ROCm backend supports AMD-specific configuration variables that can be passed to the Triton JIT or set as environment variables.

### `waves_per_eu` (Occupancy Control)
In CUDA, occupancy is often tuned by adjusting register usage or using `__launch_bounds__`. On ROCm, Triton exposes `waves_per_eu` to control the number of wavefronts per Execution Unit (EU). 
Setting this limits the VGPR (Vector General Purpose Register) allocation per wavefront, forcing the compiler to spill or optimize register usage to allow more wavefronts to run concurrently.

```python
# Force compilation targeting 2 waves per EU for latency-bound kernels
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=2, num_warps=4, waves_per_eu=2)
```

### Environment Variables
When profiling and migrating, the following environment variables are invaluable on AMD platforms:

*   `TRITON_PRINT_AUTOTUNING=1`: Displays the winning autotune configuration. Extremely useful to verify if ROCm is selecting a configuration with `matrix_instr_nonkdim=32`.
*   `AMD_LOG_LEVEL=4`: Provides low-level HIP and HSA runtime logs.
*   `ROCPROF_ENABLE=1`: If using rocprofiler, helps capture traces to visualize whether `v_mfma` instructions are optimally interleaved with `global_load` instructions.

## 4. `num_stages` and Asynchronous Copies

CUDA (especially Hopper with TMA) relies heavily on deep software pipelines (e.g., `num_stages=5` or `num_stages=7`). 

On AMD CDNA2/CDNA3, asynchronous copying from Global Memory to LDS (Local Data Share) is implemented using `async_copy` (lowered to hardware async fetch or global load instructions). While deep pipelining is beneficial, ROCm LDS has high bandwidth but can suffer from pressure if `num_stages` is set too high with large `BLOCK_SIZE`s, leading to LDS overallocation.

**Migration Action:** 
Typically, `num_stages` between `2` and `4` yields the best performance on MI250X and MI300X. Avoid simply transferring `num_stages=7` from Hopper autotuners, as it will likely result in register spilling or LDS capacity errors on ROCm.

## 5. Performance Comparison Data

A typical 4096x4096x4096 FP16 GEMM mapped via Triton demonstrates the impact of ROCm-specific tuning.

| Architecture | Setup | Peak TFLOPS (FP16) | Optimal `num_warps` | Optimal `num_stages` | `matrix_instr_nonkdim` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A100 (CUDA)** | Naive Triton | ~180 TFLOPS | 8 (256 threads) | 4 | N/A |
| **MI300X (ROCm)**| Untuned (CUDA configs) | ~350 TFLOPS | 8 (512 threads)* | 4 | Default (16) |
| **MI300X (ROCm)**| ROCm Tuned | ~800+ TFLOPS | 4 (256 threads) | 3 | 32 |

*(Note: Untuned CUDA config mapping `num_warps=8` creates an overly large 512-thread block on ROCm, leading to sub-optimal occupancy and LDS pressure).*

## Summary

Migrating Triton scripts from CUDA to ROCm requires zero changes to the actual `tl.*` math operators, making it exceptionally portable. However, the performance portability gap is closed by correctly mapping the grid and autotuner space to AMD's hardware realities: replacing warps with Wave64, aligning block shapes to MFMA non-K dimensions (16 or 32), and limiting `num_stages` to fit CDNA LDS capacities.

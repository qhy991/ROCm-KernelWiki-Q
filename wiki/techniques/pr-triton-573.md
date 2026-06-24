---
id: technique-pr-triton-573
title: "CI Infrastructure for Triton Performance Kernels and Core AMD Tests"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - optimization
  - triton-rocm
  - rocm
  - hardware
confidence: inferred
sources:
  - pr-triton-573
---

# CI Infrastructure for Triton Performance Kernels and Core AMD Tests

## Context & Intent

Pull Request [#573](https://github.com/ROCm/triton/pull/573) in the ROCm Triton repository introduces a robust GitHub Actions CI pipeline dedicated to executing performance kernels and running `test_core_amd.py`. 

The primary architectural intent of this integration is to establish an automated, zero-regression guardrail for the AMD-specific Triton backend. As Triton optimizes code for AMD CDNA architectures (CDNA2/MI250, CDNA3/MI300), the compiler backend undergoes complex transformations such as instruction scheduling, register allocation, and LDS (Local Data Share) management. This CI setup systematically validates these compiler passes for correctness and performance efficiency.

## Technical Analysis of Core AMD Compiler Tests (`test_core_amd.py`)

The inclusion of `test_core_amd.py` within the CI pipeline emphasizes the strict validation of AMD-specific intrinsics and lowering passes. The Triton AMD backend must accurately translate high-level block operations into optimized LLVM IR targeting the AMDGPU backend.

Key architectural features implicitly validated by these core tests include:

- **Matrix Core Validation (MFMA)**: Ensures that block-level matrix multiplications correctly map to AMD Matrix Fused Multiply-Add (`v_mfma_*`) instructions without correctness or scheduling regressions.
- **LDS Management**: Validates that the compiler correctly handles shared memory layout, including automated swizzling and padding to avoid bank conflicts.
- **Wavefront and Warp Equivalency**: Triton assumes a specific thread-block mapping. On AMD, this translates to 64-thread execution units (Wave64). Core tests ensure that synchronization (e.g., via `gws` or LDS barriers) and cross-lane operations (`dpp`, `bpermute`) are correctly lowered for the AMD wavefront.

## Performance Kernels & Optimization Tracking

The execution of "perf kernels" in CI is a mechanism to track hardware utilization bounds across sequential compiler commits. By running standard deep learning primitive kernels (e.g., GEMM, Flash Attention, Softmax) on actual CDNA hardware within the CI runner, it tracks two fundamental performance dimensions:

### Compute Bounds
For compute-bound kernels like dense GEMM, the CI asserts that the generated AMDGPU assembly achieves maximum arithmetic intensity. Regressions here often indicate:
- Suboptimal **MFMA scheduling** resulting in stalled compute pipelines.
- Excessive **VGPR (Vector General Purpose Register)** pressure, causing occupancy drops or spilling to scratch memory.
- Inefficient **register tiling**.

### Memory Bounds
For memory-bound kernels (e.g., reduction operations, normalization layers), the CI validates the efficiency of memory access patterns. Optimal performance requires:
- High HBM (High Bandwidth Memory) utilization via **vectorized loads** (e.g., 128-bit `global_load_dwordx4`).
- Effective **double-buffering** and asynchronous copy mechanisms from global memory to LDS.

## Conclusion

Automating continuous integration for performance benchmarks and AMD core tests represents a critical step in maturing the `triton-rocm` compiler ecosystem. By anchoring the development process to hardware-validated bounds on CDNA2, CDNA3, and future CDNA4 hardware, the pipeline guarantees that complex optimization techniques remain uncompromised during rapid compiler iteration.

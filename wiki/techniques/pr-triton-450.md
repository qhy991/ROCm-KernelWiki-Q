---
id: technique-pr-triton-450
title: "PR Insight: Triton #450 - Occupancy Info Script"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
hardware_features: [wavefront, lds]
techniques: [occupancy-tuning, register-tiling]
languages: [triton-rocm, python]
tags: [rocm, rocm-kernel, optimization, occupancy, vgpr, memory, memory-bound]
confidence: inferred
sources:
  - pr-triton-450
---

# PR Insight: Triton #450 - Occupancy Info Script

This PR introduces a dedicated utility script to the [ROCm/triton](https://github.com/ROCm/triton/pull/450) repository to calculate and print kernel occupancy information. Understanding occupancy is a critical step in ROCm kernel optimization, particularly when dealing with CDNA architectures where register pressure and local memory constraints heavily influence execution efficiency.

## Motivation and Intent

In Triton, kernels are highly parameterized (e.g., `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `num_stages`). These parameters directly dictate the hardware resources the kernel consumes:
1. **VGPRs (Vector General-Purpose Registers):** Triton's heavy reliance on register tiling means that large block sizes often consume maximum registers (e.g., 256 or 512 VGPRs per thread).
2. **LDS (Local Data Share):** Configured via `num_stages` (software pipelining), larger pipelines linearly increase LDS consumption.

Without visibility into how these block parameters map to exact hardware resource limits, developers are forced into blind trial-and-error grid searches. This PR solves that by providing a script that takes a kernel's resource requirements (VGPRs, SGPRs, LDS size) and outputs its theoretical wavefront occupancy.

## Architectural Relevance (CDNA)

For AMD Instinct accelerators (CDNA2, CDNA3), the CU (Compute Unit) limits dictate the occupancy. The underlying hardware constraints applied by the script involve:

* **Max Waves per SIMD:** 8
* **Max Waves per CU:** 32 (4 SIMDs per CU)
* **VGPR limit per SIMD:** 512 KiB (128 KiB / 32,768 registers per SIMD)
* **LDS limit per CU:** 64 KiB

The tool utilizes the mathematical model of CDNA compute units, calculating:

1. **VGPR-Bounded Occupancy:** `floor(512 / VGPRs_per_thread)` waves per SIMD.
2. **LDS-Bounded Occupancy:** `floor(64 KiB / LDS_per_workgroup) * waves_per_workgroup` waves per CU.

The final theoretical occupancy is bounded by the minimum of these limits and the hard architectural limit of 32 waves per CU. 

## Optimization Workflow Impact

By integrating this script into the development workflow, Triton users can analytically tune their kernels:

1. **Compute-Bound Kernels (GEMM / FlashAttention):** Developers can observe when their register-tiled configurations drop occupancy to 1 or 2 waves/SIMD. They can use the script to verify if decreasing `BLOCK_SIZE` slightly would allow the kernel to jump up an occupancy tier, balancing Instruction-Level Parallelism (ILP) vs. Thread-Level Parallelism (TLP).
2. **Memory-Bound Kernels (Softmax / LayerNorm):** Developers can use the tool to ensure their kernel maintains high occupancy (4-8 waves/SIMD) to effectively hide High Bandwidth Memory (HBM) latency. If the script reports low occupancy, the developer knows to reduce `num_stages` or reduce block sizes to free up LDS and VGPRs.

## Implementation Details

While the exact PR diff varies, such occupancy calculators typically interface with the compiler backend's emitted statistics. The script likely parses the compiled kernel artifacts (e.g., reading metadata from the ROCm `.hsaco` binaries) or hooks into the Triton compiler backend to extract exact register and shared memory allocations. It then translates these raw statistics into human-readable wave and workgroup occupancy metrics based on the target AMD GPU's architectural constants.

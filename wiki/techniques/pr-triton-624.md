---
id: technique-pr-triton-624
title: "Triton Plot Tool: MFMA16 Support Analysis"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - mfma
  - triton-rocm
  - python
  - optimization
  - hardware
  - occupancy
confidence: inferred
sources:
  - pr-triton-624
---

# Triton Plot Tool: MFMA16 Support Analysis

## 1. Intent and Context
PR #624 in the `ROCm/triton` repository introduces support for tracking and visualizing **MFMA16** configurations within Triton's internal plotting and performance analysis tools. As Triton kernels on AMD CDNA architectures heavily rely on Matrix Fused Multiply-Add (MFMA) instructions to achieve peak compute throughput, empowering the benchmarking suite to recognize and plot `16x16` tile variants is critical for kernel tuning.

## 2. Architectural Background: MFMA16
AMD CDNA architectures provide a suite of MFMA instructions that operate on various tile dimensions, such as `32x32`, `16x16`, and `4x4`.

- **Dimensions**: An `MFMA16` instruction (e.g., `v_mfma_f32_16x16x16f16`) computes a $16 \times 16$ output block.
- **Wavefront Mapping**: On AMD GPUs, a wavefront consists of 64 threads (`wave64`). A $16 \times 16$ tile produces 256 output elements. Thus, each thread in the wavefront is responsible for exactly $256 / 64 = 4$ registers in the accumulator matrix.
- **Comparison to MFMA32**: An `MFMA32` instruction ($32 \times 32 = 1024$ elements) requires each thread to hold 16 accumulator registers. While MFMA32 can offer higher instruction-level throughput for compute-bound workloads, it significantly increases VGPR (Vector General Purpose Register) pressure.

## 3. Optimization Techniques and Memory Bounds

### Register Occupancy Tuning
The primary motivation for selecting an `MFMA16` configuration over `MFMA32` is **occupancy-tuning**. 
Kernels like FlashAttention or complex grouped GEMMs often require substantial VGPRs for intermediate state, scaling, and masking. If an `MFMA32` instruction forces the kernel to spill registers or limits the number of active wavefronts per Compute Unit (CU) due to VGPR overallocation, the hardware cannot hide memory latency effectively. 

By switching to `MFMA16`, the register footprint for the accumulator reduces from 16 VGPRs to 4 VGPRs per thread. This reduction directly translates to higher wave occupancy, which is essential for hiding memory latency in **memory-bound** or **LDS-bound** kernels.

### Memory Hierarchy & Latency Hiding
- **LDS Usage**: Smaller MFMA tiles often correspond to smaller block-level matrix tiles, meaning fewer elements need to be prefetched into the Local Data Share (LDS). This reduces LDS pressure and allows more thread blocks to reside concurrently on a single CU.
- **Memory Bound vs. Compute Bound**: For kernels that are bandwidth-bound rather than compute-bound, maximizing the arithmetic intensity per instruction is less critical than maximizing latency hiding. `MFMA16` strikes an optimal balance, keeping the matrix cores fed while allowing enough concurrent waves to saturate the High Bandwidth Memory (HBM) bandwidth.

## 4. Integration with Triton Performance Tools
The "plot tool" updates imply that Triton's auto-tuner and performance visualization scripts can now automatically sweep, parse, and plot performance metrics for `BLOCK_SIZE_M = 16` and `BLOCK_SIZE_N = 16` matrix core selections.

- **Automated Sweeps**: When developers use `triton.testing.perf_report`, the inclusion of MFMA16 enables the tuner to natively plot latency or throughput graphs for these register-light configurations.
- **Visualization**: Identifying whether the bottleneck is VGPR-limited (favoring MFMA16) or instruction-throughput limited (favoring MFMA32) becomes easier when these data points are plotted side by side.

## 5. Summary
Adding MFMA16 support to the Triton plot tool provides crucial visibility into register-efficient kernel configurations. It enables machine learning engineers to better analyze the trade-off between matrix core throughput and CU occupancy on AMD CDNA architectures.

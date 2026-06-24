---
id: technique-pr-triton-599
title: "Triton Parameter Expansion for Targeted Tuning (one_config.py)"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, rocm, rocm-kernel, memory-bound, occupancy, profiling]
confidence: inferred
sources: [pr-triton-599]
---

# Triton Parameter Expansion for Targeted Tuning

## Overview
This wiki entry analyzes the architectural implications and tuning methodologies enabled by PR #599 in the `ROCm/triton` repository, specifically focusing on the updates to `one_config.py`. The primary intent of this modification is to expose new input parameters, augmenting developers' ability to manually profile and benchmark isolated kernel configurations.

## Technical Intent & Mechanism

The `one_config.py` script serves as a standalone testing harness that bypasses the comprehensive `triton.autotune` framework. This allows Triton developers and kernel engineers to inject a *single, deterministic configuration* (`triton.Config`) for testing correctness, compiling specific IR, or obtaining clean profiling traces.

By supporting new input parameters, the script expands the hyperparameter surface area accessible via the command line interface. This likely includes:
* **Tiling Dimensions:** `BLOCK_M`, `BLOCK_N`, `BLOCK_K`.
* **Hardware Threading Defaults:** `num_warps` (controlling Wavefront concurrency per Compute Unit).
* **Software Pipelining:** `num_stages` (dictating the depth of asynchronous copy pipelines and double-buffering).
* **Precise Hardware Knobs:** Specific CDNA constraints such as Matrix Core (MFMA) selection overrides or LDS allocation sizes.

## Architectural Implications for CDNA

Tuning Triton kernels on AMD CDNA architectures (CDNA2, CDNA3, CDNA4) requires carefully balancing compute throughput and memory bandwidth.

### 1. Navigating Memory Bounds vs. Compute Bounds
For **memory-bound kernels** (e.g., skinny GEMMs, Flash Attention memory passes, or custom layer norms), achieving peak HBM bandwidth utilization is critical. The added parameters allow developers to systematically vary `BLOCK_SIZE` arguments. Larger block sizes increase computational intensity but exert heavy pressure on the Local Data Share (LDS) and VGPR files. Fine-grained control enables developers to plot the exact inflection point where a kernel transitions from memory-bound to compute-bound.

### 2. Occupancy Tuning and Register Allocation
CDNA architectures feature deep register files but are sensitive to VGPR allocation limits. 
* Expanding `num_warps` allows developers to test how occupancy scaling affects execution. Higher warps hide latency but reduce available VGPRs per thread.
* Tuning `num_stages` directly impacts the software pipeline. On CDNA3 (MI300X), a deeper pipeline (e.g., `num_stages=4` or `5`) can effectively overlap global memory loads with MFMA compute. However, each additional stage reserves more LDS and VGPRs. The extended `one_config.py` permits probing the maximum sustainable `num_stages` before register spilling occurs.

### 3. MFMA Scheduling & Tile Programming
The ability to configure detailed inputs ensures that the chosen `BLOCK_M`, `BLOCK_N`, and `BLOCK_K` evenly divide into the underlying `v_mfma_*` instruction shapes (e.g., 32x32x8, 16x16x16). Targeted testing ensures that the Triton compiler lowers the configuration to optimal MFMA intrinsic instructions without introducing bank conflicts or redundant padding operations.

> [!TIP]
> **Performance Tip**: When utilizing `one_config.py` for occupancy tuning on MI300X (CDNA3), closely monitor the VGPR allocation. Pushing `num_stages` too high for large `BLOCK_M` and `BLOCK_N` tiles can trigger costly register spilling, abruptly collapsing performance.

## Summary

The enhancements to `one_config.py` in PR #599 provide an essential toolset for granular, single-point kernel optimization. By enabling explicit control over tiling, staging, and threading parameters, kernel engineers can systematically demystify CDNA performance bottlenecks, achieve precise occupancy tuning, and effectively manage memory bounds.

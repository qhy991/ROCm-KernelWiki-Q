---
id: technique-pr-triton-530
title: "Custom Kernel Configuration via Command Line in Triton Benchmarks"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, rocm, rocm-kernel, programming, runtime-api]
languages: [triton-rocm]
hardware_features: [wavefront, lds, mfma]
confidence: inferred
sources:
  - pr-triton-530
---

# Custom Kernel Configuration via Command Line in Triton Benchmarks

## Overview

PR [#530 in ROCm/triton](https://github.com/ROCm/triton/pull/530) introduces a critical capability for performance engineering and tuning: the ability to specify custom kernel configurations (e.g., block sizes, warp counts, and pipeline stages) directly from the command line while running the existing benchmark suite. 

This infrastructure enhancement fundamentally changes how developers profile, test, and tune ROCm Triton kernels on AMD CDNA architectures (CDNA2, CDNA3, CDNA4). Rather than being forced to hardcode specific tuning parameters or rely solely on Triton's `@triton.autotune` sweeps, developers can now dynamically inject targeted configurations.

## Architectural Context and Intent

In Triton programming, performance heavily depends on mapping the iteration space and memory tiles perfectly to the underlying hardware capabilities:

* **`BLOCK_M`, `BLOCK_N`, `BLOCK_K`**: Determine the tile dimensions loaded into SRAM/LDS and computed.
* **`num_warps`**: Dictates the occupancy and the number of threads actively collaborating on a single block. On ROCm, a "warp" conceptually maps to an AMD wavefront (64 threads on CDNA).
* **`num_stages`**: Determines the depth of the software pipeline (double buffering, multi-buffering) and impacts the degree of overlapping between global memory loads and Matrix Fused Multiply-Add (MFMA) compute operations.

By exposing these to the command line, the PR intent is to:
1. **Accelerate Profiling**: Allow rapid binary searches for optimal `num_stages` or `num_warps` without repeatedly editing source code.
2. **Deterministic Benchmarking**: Provide a way to run the entire benchmark suite using a locked, known-good configuration to measure the impact of compiler back-end changes on ROCm.
3. **Isolate Performance Regressions**: Easily verify if a performance drop is due to a bad autotuner choice or an actual compiler regression for a given set of `BLOCK` parameters.

## Detailed Analysis

### 1. Bypassing Autotune Overheads
The standard workflow in Triton relies on the `@triton.autotune` decorator, which compiles and tests multiple variants of a kernel before picking the fastest one. When investigating compiler optimization passes or hardware bottlenecks, the autotuner adds significant noise and compilation overhead. Passing custom configs at the command line allows developers to instantiate precisely the kernel variant they want.

### 2. Retaining the Benchmark Suite
The PR description notes: *"Retains the ability to run the existing benchmark suite."*
This implies that the command-line integration is seamless. It extends `triton.testing.perf_report` or the argument parsers of individual benchmark scripts to optionally override the `configs` list. If a config is provided via CLI, the benchmark suite will execute that specific config while still producing standard throughput/TFLOPS outputs.

### 3. Impact on Hardware Tuning (CDNA Architecture)
For CDNA architectures (e.g., MI250X, MI300X), optimal block sizes and stages vary significantly depending on the shape of the tensors:
* **Matrix Core (MFMA) Scheduling**: Different block sizes trigger different `v_mfma` instructions (e.g., 32x32x8 vs 16x16x16). Command-line overrides allow developers to quickly force the compiler to select specific MFMA instructions and observe the impact on VGPR utilization.
* **Occupancy Tuning**: Changing `num_warps` directly influences how many wavefronts can fit into a single Compute Unit (CU) and how the Local Data Share (LDS) is partitioned. 

## Best Practices & Usage Patterns

When leveraging this technique:
- **Baseline Establishment**: Run the benchmark suite with the custom config pointing to the optimal parameters found in a previous run to isolate compiler changes.
- **Sweep Scripts**: Wrap the benchmark execution in an external bash or Python script to perform custom heuristic sweeps that might be too complex for the built-in autotuner.
- **Hardware-Specific Tuning**: Use the custom config to test the differences between CDNA2 (MI250X) and CDNA3 (MI300X), which may require entirely different `num_stages` due to increased LDS and dual-CMA structures.

## Conclusion
Allowing command-line specification of Triton kernel configurations is a vital runtime infrastructure improvement. It empowers developers and compiler engineers with finer-grained control over kernel execution, facilitating faster iteration cycles and deeper insights into hardware utilization on AMD CDNA GPUs.

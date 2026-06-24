---
id: technique-pr-triton-534
title: "Accurate Autotuning via Graph Capture"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, runtime-api, hip, rocm-kernel]
confidence: inferred
sources: [pr-triton-534]
---

# Accurate Autotuning via Graph Capture in Triton

## Overview

PR [#534](https://github.com/ROCm/triton/pull/534) in the `ROCm/triton` repository ports upstream changes from OpenAI Triton (PR #3306) to integrate **CUDA/HIP Graph** capture into the kernel autotuning framework. This technique aims to significantly improve the accuracy of kernel benchmarking by isolating pure GPU execution time and eliminating host-side launch overhead during the tuning process.

## The Challenge: Host Launch Overhead

Triton relies heavily on its JIT compilation and `@triton.autotune` mechanism to search through a large parameter configuration space (e.g., `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`) to find the optimal kernel settings for a specific workload on target hardware.

When the autotuner benchmarks different configurations, it traditionally issues asynchronous kernel launches from the host CPU. However, the CPU overhead of enqueuing a kernel to the GPU queue—navigating API calls and the driver stack—can consume several microseconds. 

For kernels that execute very rapidly (e.g., small GEMMs, elementwise operations, or low-sequence-length Flash Attention), the kernel's actual execution time might be overshadowed by this host launch overhead. This leads to:

1. **Measurement Noise**: Variability in CPU load or OS scheduling adds unpredictability to the timing metrics.
2. **Suboptimal Selection**: The autotuner may incorrectly select a configuration because its measurement was skewed by launch latency rather than reflecting optimal GPU throughput.

## The Solution: Graph Capture for Replay

By integrating graph capture into the tuning loop, the autotuner delegates the replay of the benchmarked kernel entirely to the GPU runtime:

1. **Capture Mode**: The driver begins capturing GPU operations on a designated stream. The kernel is dispatched, but instead of executing immediately, its launch parameters and memory bindings are recorded into a graph.
2. **Instantiation & Replay**: The recorded graph is instantiated into an executable graph object. The host then launches this entire graph—which can easily encompass a loop of multiple kernel invocations—with a single driver API call.
3. **Accurate Timing**: Because the CPU is removed from the loop during the inner replay, the elapsed time directly represents raw GPU execution.

> [!NOTE]
> By eliminating the host-side dispatch overhead, graph capture allows Triton to accurately tune "launch-bound" configurations where execution times are so short that CPU noise would otherwise invalidate the benchmark.

## ROCm/HIP Implementation Details

While the upstream feature refers to "CUDA graphs", the ROCm Triton backend seamlessly translates these primitives to leverage **HIP Graphs**. The runtime transition involves mapping the graph capture semantics cleanly through Triton's driver layer to the underlying HIP API:

- `cudaStreamBeginCapture` $\rightarrow$ `hipStreamBeginCapture`
- `cudaStreamEndCapture` $\rightarrow$ `hipStreamEndCapture`
- `cudaGraphInstantiate` $\rightarrow$ `hipGraphInstantiate`
- `cudaGraphLaunch` $\rightarrow$ `hipGraphLaunch`

### Implementation Considerations on AMD CDNA Architectures

Integrating HIP graphs within a benchmarking loop requires careful runtime orchestration:

- **Stream Synchronization**: Accurate timing demands that `hipDeviceSynchronize` or `hipStreamSynchronize` bounds the entire graph launch, rather than interleaving synchronizations between individual kernel launches.
- **Memory Allocations**: Graph capture enforces strict rules on memory management. Buffers for inputs, outputs, and workspace memory must be pre-allocated and remain static throughout the capture and replay phases. Dynamic allocation inside a captured stream is heavily restricted.
- **Cache Invalidation**: During iterative autotuning, avoiding L2/Infinity Cache hits from previous executions is crucial for measuring realistic memory-bound performance. A cache-clearing mechanism (such as dispatching a dummy kernel that flushes the caches with orthogonal data) must be carefully integrated so that it operates correctly alongside or within the captured graph without artificially inflating the target kernel's measured time.

## Summary

Integrating HIP graph capture into Triton's autotuner is a foundational runtime optimization for the AMD ROCm ecosystem. It guarantees that the configurations chosen by the autotuner reflect true hardware optimality, completely sidestepping the host API overheads that traditionally mask the performance characteristics of small or launch-bound workloads.

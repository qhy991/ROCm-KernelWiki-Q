---
id: technique-hip-graph-capture
title: HIP Graph Capture and Replay
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [hip, runtime-api, optimization]
confidence: source-reported
techniques: []
hardware_features: []
kernel_types: []
related: []
sources: []
reproducibility: snippet
---

# HIP Graph Capture and Replay

## Overview
**HIP Graphs** (HIP Graph 捕获与回放) are a mechanism in the ROCm runtime designed to minimize the CPU-side overhead of submitting GPU work. By representing a sequence of operations (kernel launches, memory copies, and stream synchronizations) as a Directed Acyclic Graph (DAG), the ROCm runtime can optimize the execution topology and launch the entire graph with a single API call, significantly reducing latency for small, iterative kernels.

This feature is the ROCm equivalent of CUDA Graphs and provides the same conceptual benefits: moving the overhead of defining work out of the critical execution path.

## How HIP Graphs Reduce Kernel Launch Overhead
Standard kernel launches in HIP incur a CPU overhead (typically 2–5 microseconds per kernel) due to runtime validation, queue management, and packet submission to the hardware command processor. When a workload consists of many short-running kernels (e.g., small batch size inference or layer norms in LLMs), the CPU submission time can easily become the bottleneck, leaving the GPU underutilized (often referred to as being "CPU-bound" or "launch-bound").

HIP Graphs solve this by dividing the execution into two phases:
1. **Definition/Capture**: The sequence of operations and their dependencies are recorded into a static DAG.
2. **Instantiation and Launch (Replay)**: The graph is compiled into an executable format (`hipGraphExec_t`) that can be rapidly submitted to the GPU queue with minimal CPU involvement.

Because the command structures are pre-built during the capture phase, the actual submission overhead of the graph is drastically reduced, effectively amortizing the launch cost across all nodes in the graph.

## HIP Graph API
HIP provides two main ways to construct a graph: **Explicit API** (manually adding nodes and dependencies) and **Stream Capture** (recording operations submitted to a stream). Stream capture is the most common and easiest to integrate into existing codebases.

### Stream Capture Workflow
1. **Begin Capture**: Call `hipStreamBeginCapture` on an existing stream. Subsequent operations pushed to this stream (and any joined streams via events) are recorded.
2. **End Capture**: Call `hipStreamEndCapture` to extract the `hipGraph_t`.
3. **Instantiate**: Convert the graph into an executable format using `hipGraphInstantiate`.
4. **Launch**: Execute the graph using `hipGraphLaunch`.
5. **Clean up**: Destroy the executable graph and the graph template when no longer needed.

### Code Example: Stream Capture and Replay

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void runHipGraphExample(float* d_a, float* d_b, float* d_c, int n, int iterations) {
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    // 1. Begin Capture
    CHECK_HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    // These operations are not executed immediately; they are recorded.
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Node 1: Kernel launch
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, d_a, d_b, d_c, n);
    // Node 2: Another dependent kernel
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, d_c, d_b, d_a, n);

    // 2. End Capture
    hipGraph_t graph;
    CHECK_HIP(hipStreamEndCapture(stream, &graph));

    // 3. Instantiate Graph
    hipGraphExec_t graphExec;
    CHECK_HIP(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // 4. Replay Graph multiple times
    for (int i = 0; i < iterations; i++) {
        CHECK_HIP(hipGraphLaunch(graphExec, stream));
    }

    CHECK_HIP(hipStreamSynchronize(stream));

    // 5. Cleanup
    CHECK_HIP(hipGraphExecDestroy(graphExec));
    CHECK_HIP(hipGraphDestroy(graph));
    CHECK_HIP(hipStreamDestroy(stream));
}
```

## Best Practices

To maximize the benefits of HIP Graphs, follow these optimization strategies:

1. **High Replay Count**: The overhead of capturing and instantiating a graph is relatively high. Graphs are only beneficial if they are replayed many times (e.g., thousands of iterations in training, or repeated generation steps in inference).
2. **Update Over Rebuild**: If kernel parameters (like grid dimensions or pointer addresses) change slightly between iterations, use **Graph Updates** (`hipGraphExecUpdate`) instead of destroying and re-capturing the entire graph. Updating an existing `hipGraphExec_t` is substantially faster than full instantiation.
3. **Minimize Host-Device Syncs**: Avoid synchronizations between graph launches. Ensure the graph encapsulates a substantial amount of work.
4. **Use Memory Pools for Graph allocations**: When temporary device memory needs to be allocated inside a graph, standard `hipMalloc` cannot be captured. Instead, use HIP's memory pool API (`hipMallocAsync` with stream ordered allocations) which is graph-compatible.

## Limitations and Pitfalls

### 1. Dynamic Topologies and Control Flow
HIP Graphs are inherently static. You cannot capture operations with dynamic control flow where the number of nodes or their types depend on device-side data (e.g., a data-dependent `if/else` block that launches different kernels). 
*Workaround*: To handle dynamic shapes, update the graph parameters before replay. Conditional nodes exist in newer HIP versions but have limited expressiveness compared to host-side logic.

### 2. Device Synchronization inside Capture
Calling `hipStreamSynchronize()`, `hipDeviceSynchronize()`, or any CPU-blocking memory copy (`hipMemcpy` instead of `hipMemcpyAsync`) during a capture window is strictly prohibited. This will immediately invalidate the capture state and return an error. All operations inside the graph must be asynchronous.

### 3. Multi-Device and P2P Support
While multi-GPU graphs are supported via `hipStreamWaitEvent` across streams on different devices, capturing complex multi-device peer-to-peer (P2P) patterns requires careful management of stream contexts. Failing to set the correct active device (`hipSetDevice`) before a capture call in a multi-GPU setup is a common source of bugs.

## Performance on AMD CDNA Architectures

HIP Graphs drastically reduce the CPU latency overhead, making them critical for keeping high-end accelerators like the MI250X and MI300X fed with work.

| Operation | Typical CPU Latency (MI250X) | Typical CPU Latency (MI300X) |
|-----------|------------------------------|------------------------------|
| Single `hipLaunchKernel` | 3.5 - 5.0 µs | 2.5 - 4.0 µs |
| Stream Capture (Per Node) | 1.0 - 2.0 µs | 0.8 - 1.5 µs |
| Graph Replay (Amortized per node) | **< 0.5 µs** | **< 0.3 µs** |

*Note: Graph replay submits the entire queue of commands directly to the GPU's command processor. The actual latency depends on the complexity of the DAG. For deep, linear DAGs, the launch overhead becomes virtually negligible.*

---
id: technique-kernel-launch-overhead
title: Kernel Launch Overhead Optimization
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, hip, runtime-api]
confidence: source-reported
techniques: [persistent-kernel]
hardware_features: [gws]
kernel_types: [gemm, moe]
related: []
sources: []
reproducibility: snippet
---

# Kernel Launch Overhead Optimization

Kernel launch overhead is a critical factor in performance, particularly for models with many small kernels (e.g., MoE routing, small batch inference, custom fused operations). On AMD ROCm, HIP kernel launches involve multiple software and hardware layers that contribute to latency.

## Components of HIP Kernel Launch Latency

The total latency from calling `hipLaunchKernelGGL` (or `<<<...>>>`) to the first instruction executing on the GPU compute units (CUs) breaks down into:

1. **CPU Overhead (HIP API & Runtime):**
   When the kernel launch API is invoked, the HIP runtime allocates memory for kernel arguments, performs parameter packing, and prepares the metadata.
2. **Driver Queueing (ROCm Thunk & KFD):**
   The runtime communicates with the user-mode driver (ROCR / Thunk) and the Kernel Fusion Driver (amdgpu/KFD). This layer handles queue management and virtual memory validation.
3. **HSA Packet Dispatch (AQL & Doorbell):**
   The final step on the host is creating an HSA Architected Queuing Language (AQL) packet in the user-mode queue and "ringing the doorbell" (writing to a mapped PCIe register). The hardware Command Processor (CP) then schedules the workgroups to the CUs.

*Typical Launch Latencies (approximate, system-dependent):*
* **Standard HIP Launch:** ~3 - 5 μs
* **HIP Graph Replay:** ~1.5 - 2 μs
* **Hardware Execution (Empty Kernel):** ~1 μs

## Optimization Techniques

To mitigate kernel launch overhead, several strategies can be employed.

### 1. HIP Graphs (`hipGraph`)

HIP Graphs capture the execution topology of a sequence of kernels, memory copies, and barriers into a single execution graph. When replayed, HIP Graphs bypass the repeated CPU-side HIP API overhead and driver validations, issuing the pre-compiled AQL packets directly.

**When to use:** Fixed-topology workloads, static shapes, and repeated execution.

**Example: HIP Graph Capture and Replay**

```cpp
hipGraph_t graph;
hipGraphExec_t instance;
hipStream_t stream;
hipStreamCreate(&stream);

// Capture phase
hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
MyKernel1<<<grid1, block1, 0, stream>>>(args...);
MyKernel2<<<grid2, block2, 0, stream>>>(args...);
hipStreamEndCapture(stream, &graph);

// Instantiate once
hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

// Replay multiple times with minimal CPU overhead
hipGraphLaunch(instance, stream);
hipStreamSynchronize(stream);
```

### 2. Batched Kernel Launches

Instead of launching many small independent kernels, workloads can be fused into batched operations. For instance, launching multiple GEMMs can be done using `hipBLASLt` grouped GEMM APIs or custom batched HIP kernels. By handling multiple items inside a single kernel grid, the launch overhead is paid only once.

### 3. Persistent Kernels

Persistent kernels launch a single, long-running grid of thread blocks that continuously pull tasks from a global memory queue. This is particularly useful for dynamic workloads or when host-side synchronization is too costly. It completely hides the launch overhead for subsequent tasks, as the CUs remain occupied.

**Key Requirements for Persistent Kernels on ROCm:**
*   Ensure that the grid size does not exceed the maximum concurrently resident workgroups on the GPU (e.g., MI300X has 304 CUs. Launching too many workgroups might cause deadlocks if tasks have inter-dependencies and are waiting for un-scheduled workgroups).
*   Use `atomicAdd` to fetch task indices.

**Example: Persistent Kernel Pattern in HIP C++**

```cpp
__global__ void persistent_worker_kernel(Task* task_queue, int* task_counter, int num_tasks) {
    while (true) {
        // Fetch next task index atomically
        int task_id = 0;
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            task_id = atomicAdd(task_counter, 1);
        }
        
        // Broadcast task_id to all threads in the workgroup
        // ROCm warp size is 64 (wavefront)
        task_id = __shfl(task_id, 0);

        if (task_id >= num_tasks) {
            break; // No more tasks
        }

        // Execute task
        Task current_task = task_queue[task_id];
        process_task(current_task);
        
        // Ensure memory visibility before moving to next task
        __threadfence();
    }
}
```

## Performance Comparison: MI300X and MI250X

The following table highlights the latency benefits when applying these techniques on recent AMD architectures.

| Architecture | Technique | Launch Latency per Task | CPU Overhead |
|---|---|---|---|
| **MI250X** | Sequential HIP Launches | ~4.5 μs | High |
| **MI250X** | HIP Graphs | ~2.0 μs | Low |
| **MI250X** | Persistent Kernel | < 0.5 μs (mem latency) | Zero (after init) |
| **MI300X** | Sequential HIP Launches | ~3.0 μs | High |
| **MI300X** | HIP Graphs | ~1.5 μs | Low |
| **MI300X** | Persistent Kernel | < 0.2 μs (mem latency) | Zero (after init) |

*Note: Persistent kernel overhead represents the atomic fetch and LDS synchronization latency, not host-to-device dispatch overhead.*

### Best Practices

1. **Avoid over-subscribing Persistent Kernels:** On MI300X, limit persistent workgroups to `(Number of CUs) * (Max Workgroups per CU)` to prevent live-lock.
2. **Combine with Global Wave Sync (GWS):** For cross-CU synchronization within a persistent kernel, use GWS (supported in `cdna2` and newer) to orchestrate data dependencies safely without exiting the kernel.
3. **Limit Stream Synchronizations:** `hipStreamSynchronize` flushes the command queue and causes pipeline bubbles. When using standard launches, prefer events (`hipEventRecord`) and stream dependencies (`hipStreamWaitEvent`) over CPU synchronization.

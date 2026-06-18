---
id: kernel-all-reduce-rocm
title: AllReduce on ROCm
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [synchronization, hardware, memory, bandwidth, mi300x]
confidence: source-reported
kernel_types: [reduction]
languages: [hip-cpp]
related: []
sources: []
reproducibility: snippet
---

# AllReduce on ROCm

The `AllReduce` operation is a foundational collective communication pattern in distributed deep learning training and inference. On AMD ROCm, the RCCL (ROCm Communication Collectives Library) provides highly optimized implementations of AllReduce, leveraging AMD's Infinity Fabric (xGMI) for intra-node communication and RDMA (RoCEv2/InfiniBand) for inter-node communication.

## RCCL Internals

RCCL is AMD's equivalent of NVIDIA's NCCL. It implements various algorithms for collective operations, selecting the most optimal algorithm at runtime based on the message size, system topology, and network capabilities.

### Transport Mechanisms

RCCL supports multiple transport layers to move data between GPUs:
*   **P2P (Peer-to-Peer):** Directly accesses memory on remote GPUs using AMD Infinity Fabric (xGMI) or PCIe.
*   **SHM (Shared Memory):** Used when P2P is not possible or optimal.
*   **NET (Network):** Uses network interfaces (e.g., RoCE, InfiniBand, TCP/IP) for communication between nodes, typically utilizing the `rccl-rdma-sharp-plugins`.

### Threading and Execution Model

RCCL kernels are launched as standard HIP kernels. They utilize dedicated hardware resources to ensure high throughput and low latency:
*   **Persistent Threads:** RCCL often employs persistent kernels that stay active on the GPU and wait for work, reducing the overhead of repeated kernel launches.
*   **LL128 (Low Latency 128-bit):** An optimization protocol that embeds control flags alongside 128-bit data chunks, minimizing synchronization overhead for small messages.

## Topology and Algorithms

The choice of AllReduce algorithm is critical for performance and is highly dependent on the underlying hardware topology.

### Ring Topology

In a Ring AllReduce, GPUs are organized in a logical ring. The operation proceeds in two phases:
1.  **Scatter-Reduce:** Each GPU sends a chunk of data to the next, accumulating partial reductions.
2.  **Allgather:** The fully reduced chunks are propagated around the ring so that all GPUs have the complete result.

**Characteristics:**
*   **Optimal for Bandwidth:** Ring algorithms are bandwidth-optimal, making them suitable for large message sizes.
*   **Latency Bound:** The number of steps scales linearly with the number of GPUs ($2(N-1)$ steps), resulting in higher latency for smaller messages or larger clusters.
*   **xGMI/NVLink Utilization:** High utilization of the ring connections. On MI250X or MI300X systems, the dense xGMI meshes provide excellent ring bandwidth.

### Tree Topology

Tree algorithms organize GPUs in a hierarchical tree structure (e.g., binary tree or double binary tree).
1.  **Reduce (Up phase):** Data flows from the leaves to the root, with intermediate nodes reducing data from their children.
2.  **Broadcast (Down phase):** The final reduced result is broadcast from the root back down to the leaves.

**Characteristics:**
*   **Optimal for Latency:** The number of steps scales logarithmically ($2 \log_2 N$), making it significantly faster for small message sizes.
*   **Bandwidth Bound:** The root nodes can become a bottleneck, especially if the interconnect bandwidth is asymmetric. Double binary trees alleviate this by utilizing two trees in opposite directions.

## AMD xGMI vs. NVLink

Both xGMI (AMD Infinity Fabric) and NVLink (NVIDIA) serve as high-speed intra-node interconnects, but they have architectural differences that impact AllReduce implementations.

*   **xGMI (AMD):** AMD's Infinity Fabric provides a cohesive, cache-coherent interconnect between GPUs and CPUs. On platforms like the MI300X, the topology often features a fully connected or highly dense mesh, allowing any GPU to communicate directly with any other GPU with high bandwidth. This makes it highly efficient for both Ring and Tree algorithms.
*   **NVLink (NVIDIA):** Typically utilizes NVSwitch to provide full-bisection bandwidth across all GPUs in a node.

## Code Example: Custom AllReduce Kernel (HIP C++)

While standard deep learning workloads rely on RCCL, there are scenarios where a custom, fused AllReduce kernel is necessary to avoid the overhead of library calls or to combine communication with computation. Here's a simplified conceptual example of a ring-based reduction step using HIP.

```cpp
#include <hip/hip_runtime.h>

// Simplified conceptual Ring-Reduce step
__global__ void ring_reduce_step_kernel(
    const float* send_buff,
    float* recv_buff,
    int count,
    int rank,
    int world_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Simple element-wise reduction (assuming P2P access to recv_buff of next rank is handled externally via IPC or unified memory mapping)
    if (tid < count) {
        // In reality, synchronization and remote memory access primitives are needed here.
        // This is a highly abstracted representation.
        float local_val = send_buff[tid];
        // Remote write/reduction (requires xGMI/PCIe P2P setup)
        atomicAdd(&recv_buff[tid], local_val); 
    }
}
```

> [!IMPORTANT]
> Writing custom collective kernels requires handling low-level memory coherence, cross-GPU synchronization, and hardware-specific optimizations (like LL128 or memory-mapped queues), which is why standard libraries like RCCL are strongly recommended for production use.

## Performance Considerations on MI300X

The MI300X architecture brings significant enhancements for communication:
*   **High xGMI Bandwidth:** Delivering up to 896 GB/s of bidirectional peak bandwidth per GPU.
*   **Unified Memory Architecture:** CDNA3's APU design (in MI300A) or the cohesive memory layout in MI300X reduces memory copy overheads.

**Typical Tuning Variables (`RCCL_` env vars):**
*   `NCCL_ALGO`: Force a specific algorithm (Ring, Tree, CollNet).
*   `NCCL_PROTO`: Force a specific protocol (Simple, LL, LL128).
*   `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`: Tune the number of parallel communication channels.

### Realistic Performance Expectations

| Operation (MI300X 8-GPU Node) | Message Size | Algorithm | Peak Algorithmic Bandwidth |
| :--- | :--- | :--- | :--- |
| AllReduce | 8 MB | Tree | ~150 GB/s |
| AllReduce | 1 GB | Ring / Double Tree | ~380 GB/s |
| AllGather | 1 GB | Ring | ~400 GB/s |

*Note: Algorithmic bandwidth is calculated as S / t, whereas bus bandwidth is typically higher, e.g., S * 2(N-1) / (N * t) for AllReduce.*

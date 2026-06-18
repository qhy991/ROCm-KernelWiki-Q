---
id: technique-multi-gpu-rccl
title: RCCL Multi-GPU Communication
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [multi-gpu, communication, optimization, memory-management]
confidence: source-reported
techniques: [memory-mapping, overlap-compute-transfer]
hardware_features: [xgmi, infinity-fabric]
kernel_types: [communication]
related: []
sources: []
reproducibility: snippet
---

# RCCL Multi-GPU Communication

RCCL (ROCm Communication Collectives Library) is a standalone library of standard collective communication routines for GPUs, highly optimized for AMD's Infinity Fabric (xGMI) and PCIe interconnects. RCCL is an API-compatible implementation of NVIDIA's NCCL. When dealing with multi-GPU and distributed systems on AMD CDNA architectures (like MI250X and MI300X), leveraging and tuning RCCL alongside raw HIP IPC (Inter-Process Communication) is key to maximizing throughput and hiding communication latency.

## RCCL Architecture and Interconnect Awareness

RCCL relies heavily on the underlying hardware topology to build rings and trees for collective operations. On AMD platforms, the interconnect is typically **xGMI (Infinity Fabric)** for intra-node communication and PCIe/InfiniBand/RoCE for inter-node communication.

### xGMI and Topology Awareness

Modern AMD compute nodes typically group GPUs via Infinity Fabric links. For instance, the MI250X features two GCDs (Graphics Compute Dies) per OAM package, connected by intra-package xGMI links, and cross-package xGMI links to other OAMs. The MI300X connects 8 GPUs within a node in a fully connected or ring/chord topology.

RCCL auto-detects this topology to optimize data routing. You can verify the link structures using:
```bash
rocm-smi --showtopo
```
This shows the weight/hops between devices. RCCL uses this metric to form optimal communication rings.

## Tuning RCCL Environment Variables

Like NCCL, RCCL exposes several environment variables to tune algorithms, routing, and low-level protocol choices.

### Key Environment Variables

* **`NCCL_ALGO`**: Forces the collective algorithm.
  * `Ring`: Forms a ring across GPUs. Best for large message sizes.
  * `Tree`: Forms a double binary tree. Often yields lower latency for small to medium message sizes.
  * `CollNet`: Uses in-network computing if supported (e.g., SHARP).
* **`NCCL_PROTO`**: Communication protocol.
  * `Simple`: Standard sender-receiver protocol with synchronization. Good for large messages.
  * `LL` (Low Latency): Data and flags are interleaved. Best for small messages to reduce synchronization overhead.
  * `LL128`: Uses 128-byte loads/stores. A middle ground between Simple and LL.
* **`NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`**: Controls the number of independent parallel rings/trees. Increasing channels can improve bandwidth utilization on high-bandwidth xGMI links (e.g., MI300X where multi-link bandwidth is extremely high).
* **`HSA_ENABLE_SDMA=1`**: Enables the use of the System DMA engine for copies. Depending on the ROCm version and system configuration, this can offload copy overhead from the compute units (CUs).
* **`NCCL_P2P_DISABLE=0`**: Ensures Peer-to-Peer access is enabled. If set to 1, RCCL falls back to host (pinned memory) staging, severely impacting multi-GPU performance.

### Performance Tuning Matrix for MI300X

| Workload | Message Size | Recommended Settings |
| :--- | :--- | :--- |
| LLM Inference (TP) | Small (< 1MB) | `NCCL_ALGO=Tree`, `NCCL_PROTO=LL` |
| Deep Learning Training (DP) | Large (> 32MB)| `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple` |
| Hybrid / MoE | Mixed | Auto (RCCL internal heuristic tuning) |

## Combining RCCL with HIP IPC

While RCCL is highly optimized for collectives (AllReduce, AllGather, ReduceScatter, Broadcast), it incurs library overhead and restricts execution models to standard collective patterns. For fine-grained, direct GPU-to-GPU memory access (e.g., neighbor exchanges in stencil computations or custom halo exchanges), combining RCCL with **HIP IPC** provides the highest flexibility and performance.

### HIP IPC Basics

HIP IPC allows one GPU process to expose a memory allocation to another GPU process. Under the hood, this establishes a direct xGMI mapping, bypassing the host completely.

```cpp
// Process A (GPU 0)
void* d_ptr;
hipMalloc(&d_ptr, size);
hipIpcMemHandle_t handle;
hipIpcGetMemHandle(&handle, d_ptr);
// Send 'handle' to Process B via sockets, MPI, etc.

// Process B (GPU 1)
void* peer_d_ptr;
// Open the IPC handle
hipIpcOpenMemHandle(&peer_d_ptr, handle, hipIpcMemLazyEnablePeerAccess);
// Now GPU 1 can read/write to peer_d_ptr directly using standard HIP kernels
```

### Hybrid Strategy

A common optimization pattern in large-scale PDE solvers or graph neural networks is the **Hybrid RCCL + IPC approach**:

1. **Global Reductions**: Use RCCL (`ncclAllReduce`) for calculating global scalars (e.g., loss, global gradients, norms). RCCL is extremely efficient at reducing data across the entire fabric.
2. **Local Peer-to-Peer**: Use HIP IPC pointers for direct read/write from neighboring GPUs within the same node.
3. **Overlapping**: Launch the RCCL collective on a separate HIP stream. Simultaneously, launch a custom kernel on the main stream that reads from remote xGMI memory via the opened HIP IPC pointers.

```cpp
// Pseudocode for Hybrid Communication
hipStream_t compute_stream, comm_stream;

// 1. Start RCCL collective asynchronously
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat32, ncclSum, comm, comm_stream);

// 2. Perform direct peer-to-peer data copy or computation via IPC pointers
// 'peer_ptr' was acquired via hipIpcOpenMemHandle
exchange_halo_kernel<<<blocks, threads, 0, compute_stream>>>(local_ptr, peer_ptr);

// 3. Synchronize
hipStreamSynchronize(comm_stream);
hipStreamSynchronize(compute_stream);
```

By explicitly separating the operations, you avoid stalling the execution pipeline. The custom kernel can utilize fine-grained synchronization (like atomic operations over xGMI) on the `peer_ptr`, offering significantly lower latency than calling `ncclSend` and `ncclRecv` for small, fragmented data.

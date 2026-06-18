---
id: hw-infinity-fabric
title: AMD Infinity Fabric (xGMI)
type: wiki-hardware
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [hardware, bandwidth, mi300x, synchronization, cross-cu]
confidence: source-reported
hardware_features: []
related: []
sources: []
cuda_equivalent: nvlink
---

# AMD Infinity Fabric (xGMI)

AMD Infinity Fabric (also known as Global Memory Interconnect or xGMI) is the proprietary high-bandwidth, low-latency interconnect architecture used to connect AMD Instinct GPUs (and CPUs) within a system node. It is the architectural equivalent of NVIDIA's NVLink.

In a multi-GPU system, Infinity Fabric enables fully coherent memory sharing across multiple GPUs, allowing a kernel on one GPU to access the High-Bandwidth Memory (HBM) of another GPU directly without traversing the slower PCIe bus.

## Topology and Bandwidth

The architecture of Infinity Fabric topologies varies significantly between CDNA2 (MI250X) and CDNA3 (MI300X). 

### Performance and Bandwidth Table

| Architecture | Setup | Topology Type | Links per GPU/GCD | Bidirectional Link Bandwidth | Peak Bidirectional Bandwidth (GPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CDNA3 (MI300X)** | 8-GPU UBB Node | Fully Connected Mesh | 7 links | 128 GB/s | 896 GB/s |
| **CDNA2 (MI250X)** | 4-OAM (8-GCD) Node | Heterogeneous (Rings / Direct) | 1 to 4 links per pair | 100 GB/s (single link) | 400 GB/s (intra-OAM) |

### CDNA3: MI300X Fully Connected Mesh
The AMD Instinct MI300X platform introduces a uniform, **fully connected mesh topology** for 8-GPU configurations on a Universal Baseboard (UBB).

* **Links per GPU:** Each MI300X GPU utilizes **7 bidirectional Infinity Fabric links**.
* **Aggregate Bandwidth:** $7 \times 128 \text{ GB/s} = 896 \text{ GB/s}$ peak bidirectional Infinity Fabric bandwidth per GPU.
* **Total System Bandwidth:** Across the 8-GPU node, this uniform fully connected mesh provides a massive 896 GB/s per-GPU capability, effectively eliminating non-uniform NUMA penalties inside a node.

### CDNA2: MI250X Heterogeneous Connectivity
The MI250X uses an OAM (OCP Accelerator Module) form factor that encapsulates two **Graphics Compute Dies (GCDs)** per package. Standard 4-OAM (8-GCD) nodes typically feature a heterogeneous Infinity Fabric layout (often forming connected rings).

* **Intra-Package (GCD to GCD):** The two GCDs within the same OAM are connected via 4 Infinity Fabric links, yielding **400 GB/s bidirectional** bandwidth (200 GB/s each direction).
* **Inter-Package:** Connections between different OAMs use varying numbers of links depending on the routing distance. Some pairs use dual-links (200 GB/s bidirectional), while others use single-links (100 GB/s bidirectional).

## NUMA Distance and Implications for Multi-GPU RCCL

When scaling Deep Learning applications across multiple GPUs using the ROCm Communication Collectives Library (RCCL), the underlying Infinity Fabric topology drastically influences performance. 

### RCCL on MI250X
Because of the heterogeneous link bandwidth (400 GB/s intra-OAM vs 200 GB/s or 100 GB/s inter-OAM), MI250X nodes exhibit highly non-uniform NUMA characteristics.
* **Ring Formation:** RCCL attempts to build fully connected rings that traverse the highest bandwidth links across the nodes. However, the bottleneck of any ring algorithm is bounded by the slowest link in the ring (typically the 100 GB/s single links).
* **Workarounds:** Developers often use hierarchical communication strategies or RCCL's MSCCL (Microsoft Collective Communication Library) plugins to optimize traffic, performing intra-OAM reductions first before executing inter-OAM cross-node steps.

### RCCL on MI300X
The fully connected mesh on MI300X perfectly complements collective communication patterns like All-Reduce and All-to-All.
* **Uniform NUMA:** Every GPU is exactly **1 hop** away from every other GPU at an identical 128 GB/s bidirectional bandwidth. 
* **Algorithmic Efficiency:** RCCL does not have to navigate heterogeneous bottlenecks. Direct All-to-All operations achieve peak efficiency without complex topological routing. 

## Checking Link Status

Developers can inspect the Infinity Fabric topology and link bandwidth using `rocm-smi` or `rocminfo`:

```bash
# View xGMI link status and bandwidth between all GPUs
rocm-smi --showxgmierr --showtopo
```

Example topology output on an 8-GPU MI300X node will display single-hop links between all pairs of nodes `[0-7]`, demonstrating the fully connected mesh.

### Code Example: Peer-to-Peer Access
Under the hood, ROCm maps remote GPU HBM to the local address space. Using the HIP runtime, P2P memory access is configured as follows:

```cpp
int canAccessPeer = 0;
// Check if Device 0 can access Device 1 over Infinity Fabric
hipDeviceCanAccessPeer(&canAccessPeer, 0, 1);

if (canAccessPeer) {
    // Enable P2P access
    hipSetDevice(0);
    hipDeviceEnablePeerAccess(1, 0);

    // After enabling, device 0 can directly read/write device 1 pointers
    // via standard load/store instructions or hipMemcpy
}
```

When accessing remote memory, latency is higher than local HBM, and bandwidth is limited by the Infinity Fabric link capacity between the two specific GPUs.

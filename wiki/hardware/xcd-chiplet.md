---
id: hw-xcd-chiplet
title: XCD Chiplet Architecture
type: wiki-hardware
architectures: [cdna3]
tags: [hardware, mi300x, memory, scheduling]
confidence: source-reported
hardware_features: [compute-unit, dual-cma]
related: []
sources: []
cuda_equivalent: null
---

# XCD Chiplet Architecture (MI300X)

The AMD Instinct MI300X (CDNA3) relies on a deeply disaggregated chiplet architecture to achieve unprecedented scale in memory capacity and compute density. Instead of a single monolithic GPU die, the MI300X is composed of multiple **Accelerator Complex Dies (XCDs)** and **I/O Dies (IODs)** interconnected via Infinity Fabric in a sophisticated 3.5D packaging layout.

## 3.5D Architecture Layout

The MI300X physical topology consists of:
* **8 XCDs (Accelerator Complex Dies):** Each XCD contains the actual compute resources, including 38 active Compute Units (CUs), scalar/vector ALUs, Dual Compute Matrix Accelerators (CMA), and a private **4 MB L2 Cache**.
* **4 IODs (I/O Dies):** The 8 XCDs are 3D-stacked directly on top of 4 base IODs (2 XCDs per IOD). The IODs provide high-speed routing, PCIe/Infinity Fabric external links, and memory controllers.
* **8 HBM3 Memory Stacks:** Placed alongside the IODs on a silicon interposer (2.5D integration), providing a total of 192 GB of capacity.

This interconnected chiplet design results in a highly **Non-Uniform Memory Access (NUMA)** system within a single GPU package.

## Memory Latency and NUMA Effects

Because each XCD has its own private L2 cache and specific proximity to certain HBM3 stacks, memory access latencies vary significantly depending on data placement and execution location:

1. **Local Access:** A workgroup executing on XCD 0 accessing HBM directly attached to its underlying IOD enjoys the highest bandwidth and lowest latency.
2. **Remote Access:** If a workgroup on XCD 0 must read from an HBM stack attached to the IOD for XCD 7, the memory transaction must traverse the on-package Infinity Fabric. This introduces additional cross-die latency ($L_{IF}$).
3. **L2 Cache Isolation:** The 4 MB L2 cache is *private* to each XCD. There is no shared L2 across XCD boundaries. If two thread blocks running on different XCDs access the same global memory address, the data will be fetched from HBM twice, occupying L2 capacity in both XCDs.

## NPS (Nodes Per Socket) Partitioning Modes

To manage the NUMA characteristics, the MI300X firmware and ROCm runtime expose **NPS (Nodes Per Socket)** modes. These modes control how the GPU is logically partitioned to the OS and host applications.

| Mode | Topology | Device Count | Behavior / Use Case |
|---|---|---|---|
| **NPS1** | 1 Domain | 1 Logical Device (304 CUs) | HBM is uniformly interleaved across all 8 stacks. Ideal for massive single models (e.g., 70B+ LLMs) where capacity and simplicity outweigh NUMA latency overheads. |
| **NPS4** | 4 Domains | 4 Logical Devices (76 CUs each) | GPU partitioned by IOD. Each device has 2 XCDs and 48 GB of local HBM3. Recommended for multi-tenant serving (CPX mode) to isolate workloads and reduce cross-die traffic. |
| **NPS8** | 8 Domains | 8 Logical Devices (38 CUs each) | Strict isolation. 1 XCD and 24 GB HBM3 per device. Maximum memory locality and lowest latency, but limits maximum kernel scale. |

## Impact on Kernel Scheduling

The chiplet boundaries severely impact how kernels should be optimized, particularly for memory-bound and synchronization-heavy workloads like Flash Attention or large GEMMs.

### Spatial-Aware Block Scheduling
Default grid scheduling in HIP distributes workgroups (thread blocks) pseudo-randomly across all available CUs. In NPS1 mode, this means sequential blocks may land on entirely different XCDs. 

To maximize L2 cache hit rates and minimize HBM traffic, developers implement **spatially-aware block swizzling**:
* Ensure that blocks processing the same KV-cache sequence (e.g., in attention mechanisms) are scheduled to the same XCD.
* Partition the output matrix into large tiles such that the data footprint fits within the 4 MB L2 cache of a single XCD.

### Persistent Kernels & Cross-CU Synchronization
Persistent kernels that require barrier synchronization across the entire GPU incur significantly higher penalties on MI300X than monolithic GPUs. If a kernel must sync across all 304 CUs, the barrier message must cross the Infinity Fabric between all 8 XCDs. 

> [!TIP]
> Avoid global synchronization within kernels if possible. Prefer localized synchronization within an XCD (e.g., using Grid synchronization conservatively), or rely on kernel boundary termination if global consistency is required.

## Detection in HIP C++

You can detect the current partitioning mode at runtime by querying the number of CUs available to the logical HIP device.

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

void detect_xcd_topology() {
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    
    int cu_count = props.multiProcessorCount;
    
    std::cout << "Device: " << props.name << "\n";
    std::cout << "Total CUs: " << cu_count << "\n";
    
    if (cu_count == 304) {
        std::cout << "Mode: NPS1 (Unified 8 XCDs)\n";
        std::cout << "Warning: Expect high cross-die NUMA latency.\n";
    } else if (cu_count == 76) {
        std::cout << "Mode: NPS4 (Partitioned, 2 XCDs per device)\n";
    } else if (cu_count == 38) {
        std::cout << "Mode: NPS8 (Fully Isolated, 1 XCD per device)\n";
    }
}
```

## Performance Characteristics

The NUMA effects directly translate into observable bandwidth limits. While the theoretical peak HBM3 bandwidth for the MI300X is 5.3 TB/s, achieving this requires perfect localized access across all XCDs without congestion.

| Metric | Measured Bandwidth | Latency Profile |
|--------|--------------------|-----------------|
| Local L2 Cache (per XCD) | ~1.5 TB/s (12 TB/s aggregate) | Ultra-low |
| Local HBM3 (Intra-IOD) | ~5.0 - 5.3 TB/s (aggregate peak) | Baseline HBM |
| Remote HBM3 (Cross-IOD) | ~3.8 - 4.2 TB/s (aggregate) | + $L_{IF}$ Penalty |
| Cross-XCD Sync Penalty | N/A | High |

Understanding these hardware boundaries is essential for writing peak-performance code on CDNA3, particularly when navigating between unified memory simplicity (NPS1) and explicit chiplet-aware programming (NPS4/8).

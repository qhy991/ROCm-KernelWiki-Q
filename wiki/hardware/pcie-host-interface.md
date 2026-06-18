---
id: hw-pcie-xgmi
title: PCIe and xGMI Host Interface
type: wiki-hardware
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [hardware, memory, bandwidth, optimization, mi300x]
confidence: source-reported
hardware_features: []
related: []
sources: []
cuda_equivalent: NVLink / PCIe
---

# PCIe and xGMI Host Interface

In AMD ROCm architectures, data movement between the host (CPU) and device (GPU), as well as peer-to-peer (P2P) transfers between GPUs, forms the backbone of large-scale distributed training and inference workloads. Modern AMD accelerators like the MI300X (CDNA3) heavily rely on advanced PCIe Gen 5 interfaces and xGMI (Infinity Fabric) links, accelerated by dedicated System Direct Memory Access (SDMA) engines.

## Interconnect Hardware Overview

### PCIe Gen 5 x16 Interface
The MI300X OAM (Open Accelerator Module) interfaces with the host CPU using **PCIe Gen 5 x16**. 
- **Theoretical Peak:** 128 GB/s bidirectional bandwidth.
- **Practical Throughput:** Typically peaks around 50-64 GB/s per direction due to PCIe protocol and system overheads.
- **Use Cases:** Initial weight loading, asynchronous stream of activation offloading, and continuous host-device synchronization.

### xGMI (Infinity Fabric)
For inter-GPU communication, AMD utilizes 4th Generation Infinity Fabric (xGMI).
- **Topology:** In an 8-GPU Universal Base Board (UBB 2.0) setup, each MI300X features 7 xGMI links, establishing an all-to-all fully connected mesh.
- **Bandwidth per Link:** Up to 128 GB/s peak theoretical per bidirectional link.
- **Aggregate Bandwidth:** ~896 GB/s total peak theoretical P2P bandwidth per GPU.
- **Coherency:** xGMI provides cache-coherent memory access across GPUs, enabling efficient peer-to-peer `hipMemcpy` and direct loads/stores.

## System Direct Memory Access (SDMA)

Data transfers are orchestrated by **SDMA engines**, which are dedicated hardware modules independent of the Compute Units (CUs).

### Mechanism
- **Offloaded Transfers:** Invoking `hipMemcpyAsync` schedules commands on the SDMA engine via a HIP stream. This allows the GPU's CUs to perform dense math operations (like `v_mfma_f32_32x32x8f16`) simultaneously without stalling on data movement.
- **SDMA vs. Blit Kernels:** ROCm defaults to using SDMA for bulk memory transfers. However, on certain systems, if the SDMA hardware becomes a bottleneck for very specific workloads, developers can force the runtime to use CU-based "Blit" (copy) kernels by setting `HSA_ENABLE_SDMA=0`.
  - *Trade-off:* Blit kernels can sometimes achieve higher peak throughput on the interconnect by utilizing massive CU parallelism, but they consume valuable compute resources and register files, reducing overall arithmetic throughput.

## Optimizing Host-to-Device Communication

To fully saturate the PCIe and xGMI links, correct host memory allocation and transfer patterns are required.

### 1. Pinned (Page-Locked) Memory
Host memory is by default pageable, meaning the OS can swap it out. Transferring from pageable memory requires the driver to first stage the data into a hidden pinned buffer, effectively halving bandwidth and increasing latency.
- **Optimization:** Use `hipHostMalloc` or `hipHostRegister` to allocate or pin host memory. This enables the SDMA engine to perform direct memory access (DMA) directly from host RAM.
- **Performance Impact:** Pinned memory transfers are typically 2-3x faster than pageable memory transfers.

```cpp
// Optimal H2D Transfer setup
float* h_data;
float* d_data;
size_t bytes = N * sizeof(float);

// Allocate pinned memory on the host
hipHostMalloc((void**)&h_data, bytes);
hipMalloc((void**)&d_data, bytes);

// Asynchronous copy leveraging SDMA
hipMemcpyAsync(d_data, h_data, bytes, hipMemcpyHostToDevice, stream);

// Overlap with compute
MyKernel<<<grid, block, 0, stream>>>(d_data);
```

### 2. Avoid Zero-Copy Device Access for Heavy Compute
While `hipHostMalloc` makes host memory directly accessible to the device kernels (zero-copy), accessing host memory from a device kernel over PCIe is extremely slow compared to local HBM3 (which peaks at 5.3 TB/s on MI300X).
- **Best Practice:** Only use zero-copy direct access for tiny control flags or one-off reads. For substantial data, always asynchronously pre-fetch data to device HBM via `hipMemcpyAsync`.

### 3. NUMA Affinity
On modern multi-socket EPYC CPUs, PCIe lanes are localized to specific NUMA nodes. 
- **Optimization:** Ensure the host thread calling `hipHostMalloc` is bound to the NUMA node physically connected to the target GPU's PCIe root complex. Tools like `numactl` or `rocm-smi --showtopo` help identify the correct CPU-GPU affinity.

### 4. Transfer Batching
SDMA engines incur setup latency for each transfer command. 
- **Optimization:** Batch smaller data chunks into a single large contiguous `hipMemcpyAsync` command whenever possible to amortize the setup cost and maximize sustained PCIe throughput.

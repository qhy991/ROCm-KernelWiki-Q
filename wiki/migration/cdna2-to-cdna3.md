---
id: migration-cdna2-to-cdna3
title: "Migration Guide: CDNA2 to CDNA3 Architecture"
type: wiki-migration
architectures: [cdna2, cdna3]
tags: [migration, mi300x, hardware, compute-unit, memory]
confidence: source-reported
from_architecture: cdna2
to_architecture: cdna3
difficulty: moderate
related: []
sources: []
---

# Migration Guide: CDNA2 to CDNA3 Architecture

The transition from the CDNA2 architecture (MI250X) to the CDNA3 architecture (MI300X/MI300A) represents a significant shift in both hardware packaging and software programming models. The most impactful change for kernel developers is the move from a **dual-GCD (Graphics Compute Die)** approach in MI250X to a **single unified logical device** in MI300X. 

This page details the architectural differences, grid sizing adjustments, and new hardware capabilities to consider when porting and tuning kernels for CDNA3.

## Architectural Shift: GCDs vs. XCDs

### MI250X (CDNA2): The Dual-GCD Model
An MI250X OAM (Open Accelerator Module) is composed of two independent Graphics Compute Dies (GCDs). 
* **OS/HIP View:** Each module appears as **two distinct HIP devices** (e.g., Device 0 and Device 1).
* **Compute Units:** 110 CUs per GCD (220 CUs total per module).
* **Memory/NUMA:** Each GCD has its own local 64GB HBM2e (128GB total). These act as separate NUMA domains.
* **Kernel Launch:** To saturate the full module, developers must split data, use multi-GPU programming models (e.g., NCCL/RCCL, or explicit peer-to-peer copies), and launch kernels independently on both GCDs.

### MI300X (CDNA3): The Unified Accelerator
The MI300X utilizes a highly advanced 3D packaging technology, combining 8 Accelerator Complex Dies (XCDs) and 4 I/O Dies (IODs) connected via a massive silicon interposer.
* **OS/HIP View:** Despite having 8 compute chiplets, the hardware presents itself to the HIP runtime as a **single, unified device**.
* **Compute Units:** 304 CUs total (38 CUs active per XCD).
* **Memory/NUMA:** A single unified 192GB HBM3 memory pool. All CUs have uniform, high-bandwidth access to the entire memory space.
* **Kernel Launch:** A single `hipLaunchKernelGGL` call can span all 304 CUs, drastically simplifying the programming model and eliminating manual cross-GCD synchronization within a module.

## Impact on Kernel Tuning

The consolidation from two 110-CU devices to one 304-CU device necessitates re-tuning grid dimensions and block allocations to avoid severe underutilization.

### 1. Grid Sizing and Saturation
A grid configuration carefully tuned for MI250X will severely under-utilize an MI300X.
* **MI250X Target:** 110 CUs. A common approach is launching 110, 220, or 440 workgroups to achieve 1, 2, or 4 waves per CU.
* **MI300X Target:** 304 CUs. 

> [!WARNING]  
> If you run a kernel tuned with a grid size of `110` or `220` workgroups on an MI300X, the GPU will only operate at **~36% to ~72% occupancy**, leaving dozens of CUs completely idle.

**Code Example: Adaptive Grid Sizing in HIP C++**
```cpp
// ANTI-PATTERN: Hardcoding grid sizes based on MI250X assumptions
int num_blocks = 220; // Assumes 2 blocks per CU on a 110-CU MI250X GCD
hipLaunchKernelGGL(MyKernel, dim3(num_blocks), dim3(256), 0, stream);

// BEST PRACTICE: Query device properties to adaptively size grids
hipDeviceProp_t props;
hipGetDeviceProperties(&props, device_id);
int num_cus = props.multiProcessorCount; // 110 on MI250X, 304 on MI300X

// Scale grid size to target a specific number of waves per CU
int blocks_per_cu = 4;
int optimal_grid_size = num_cus * blocks_per_cu; 
hipLaunchKernelGGL(MyKernel, dim3(optimal_grid_size), dim3(256), 0, stream);
```

### 2. NUMA and Memory Management
On MI250X, managing cross-GCD memory access was critical. Accessing memory residing on GCD0 from GCD1 traversed the Infinity Fabric, providing ~400 GB/s instead of the local ~1.6 TB/s HBM bandwidth. 

On MI300X, developers no longer need to manage memory placement within the same module. The entire 192GB HBM3 pool provides up to 5.3 TB/s of aggregate bandwidth. You can remove `hipMemcpyPeerAsync` calls that were previously used for intra-module communication.

### 3. Workgroup and Wavefront Scheduling
Because the MI300X is physically distributed across 8 XCDs, inter-workgroup communication (e.g., using Global Wave Sync for persistent kernels) now spans across chiplets. While the hardware abstracts this, cross-XCD atomic operations and synchronization have slightly different latency profiles than intra-GCD syncs on MI250X.
* Keep block sizes and shared memory (LDS) usage bounded to allow sufficient active waves to hide this latency.
* **LDS Capacity:** Remains 64KB per CU, but instruction throughput has been optimized. 

## Matrix Core (MFMA) Enhancements

The CDNA3 architecture introduces new Matrix Fused Multiply-Add (MFMA) instructions specifically targeting AI workloads. When migrating, developers should leverage these new formats:

### FP8 and BF8 Native Support
CDNA3 introduces hardware support for 8-bit floating-point math (both E4M3 and E5M2), critical for high-throughput LLM inference and training. 

| ISA Instruction | Datatype (In -> Out) | Block Size | Architecture |
| :--- | :--- | :--- | :--- |
| `v_mfma_f32_32x32x8f16` | FP16/BF16 -> FP32 | 32x32x8 | CDNA2, CDNA3 |
| `v_mfma_f32_32x32x16_fp8_fp8` | FP8 -> FP32 | 32x32x16 | **CDNA3 Only** |
| `v_mfma_f32_16x16x32_fp8_bf8` | FP8/BF8 -> FP32 | 16x16x32 | **CDNA3 Only** |

### TF32 Support
MI300X adds TF32 (Tensor Float 32) support. It behaves similarly to FP32 but truncates the mantissa to 10 bits, allowing the throughput of FP16 with the range of FP32. On MI250X, TF32 operations required software emulation or fallback to true FP32, making MI300X significantly faster for unconverted TF32 PyTorch workloads.

### CDNA3 ISA Additions
* **Dual Compute Matrix Accelerator (CMA):** Each CU in CDNA3 contains optimized data paths feeding the matrix cores, allowing better overlap of vector loads with matrix math compared to CDNA2.
* **VOPD (Vector Dual Issue):** CDNA3 supports issuing certain pairs of vector instructions simultaneously to maximize ALU usage.

## Performance Comparison Profile

| Metric | MI250X (1 Module = 2 GCDs) | MI300X (1 Unified Module) | Migration Implication |
| :--- | :--- | :--- | :--- |
| **Logical Devices** | 2 | 1 | Remove manual multi-GPU IPC logic for single module |
| **Compute Units** | 220 (110 per GCD) | 304 (Unified) | Multiply `grid_size` target by ~2.7x vs a single GCD |
| **Peak Memory BW** | 3.2 TB/s (1.6 per GCD) | 5.3 TB/s (Unified) | Memory-bound kernels see immediate uplift |
| **Max Memory Capacity** | 128 GB (64 per GCD) | 192 GB (Unified) | Can fit larger KV caches/weights without splitting |
| **Peak FP16 TFLOPS** | ~383 TFLOPS | ~1307 TFLOPS | Compute bound kernels scale massively |

## Summary
Porting from CDNA2 to CDNA3 fundamentally reduces programming friction by abstracting away the multi-die physical layout into a single, massive GPU context. To fully utilize the MI300X, kernel developers must dynamically query the `multiProcessorCount` to correctly scale grid sizing, while leveraging new FP8/TF32 MFMA instructions to maximize arithmetic throughput.

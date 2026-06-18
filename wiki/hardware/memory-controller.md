---
id: hw-memory-controller
title: HBM3 Memory Controller
type: wiki-hardware
architectures: [cdna3, cdna4]
tags: [memory, hbm, bandwidth, optimization, mi300x]
confidence: source-reported
hardware_features: []
related: []
sources: []
cuda_equivalent: hbm-memory-controller
---

# HBM3 Memory Controller on AMD CDNA Accelerators

AMD's CDNA3 (MI300 series) and future CDNA4 accelerators rely on High Bandwidth Memory 3 (HBM3) to deliver extreme memory bandwidth for memory-bound AI and HPC workloads. The MI300X, for example, features 192 GB of HBM3 memory distributed across 8 stacks, providing a theoretical peak bandwidth of 5.3 TB/s. To achieve a high fraction of this peak in production kernels, developers must understand the memory controller architecture, channel interleaving, and how to structure memory access patterns to maximize channel utilization.

## Architecture Overview

On the AMD Instinct MI300X (CDNA3), the GPU package contains multiple XCDs (Compute Dies) and IODs (I/O Dies). The memory subsystem interfaces with the 8 HBM3 stacks via the Infinity Fabric.
- **Stacks:** 8 HBM3 stacks per MI300X package.
- **Channels:** Each HBM3 stack is subdivided into multiple independent pseudo-channels (typically 16 pseudo-channels per stack, totaling 128 pseudo-channels across the GPU).
- **Interface:** The interface width per channel is 64 bits (32 bytes per clock on double data rate), and each pseudo-channel is managed by an independent memory controller.

These memory controllers are responsible for scheduling reads, writes, and managing DRAM banks, rows, and columns. Maximum memory bandwidth is only achieved when memory requests originating from the Compute Units (CUs) are evenly distributed across all active memory channels and banks, keeping the memory controllers fully saturated without overflowing their queues.

## Channel Interleaving

AMD GPUs utilize **channel interleaving** to transparently distribute contiguous memory allocations across multiple memory channels. This prevents a single memory channel from becoming a bottleneck when sequential memory addresses are accessed.

- **Interleave Block Size:** Typically, consecutive memory addresses are mapped to the same channel up to a specific interleave block size (often 256 bytes, aligning with a 64-thread wavefront accessing 4-byte elements).
- **Striding:** Once the block size is exceeded, the physical address maps to the next memory channel in a round-robin fashion.
- **Channel Conflicts (Camping):** If a kernel accesses memory in strides that align with the total number of channels multiplied by the interleave size, it can result in channel conflicts. In this scenario, all memory requests hit a small subset of memory channels while the rest remain idle, drastically reducing effective bandwidth.

## Maximizing Channel Utilization

To approach the 5.3 TB/s peak bandwidth of the MI300X, developers must write kernels that issue highly parallel, wide memory operations that map evenly across all HBM3 channels.

### 1. Vectorized Loads and Stores

Always use vectorized load/store instructions (`global_load_dwordx4`, `buffer_load_dwordx4`) to maximize the bytes transferred per instruction. A single 128-bit (16-byte) load per thread is the most efficient way to access global memory.

```cpp
// Suboptimal: Scalar loads per thread
// Generates global_load_dword (4 bytes per thread)
float val = ptr[idx]; 

// Optimal: Vectorized 128-bit loads using float4
// Generates global_load_dwordx4 (16 bytes per thread)
float4 val = reinterpret_cast<const float4*>(ptr)[idx]; 
```

In CDNA assembly, an optimal load looks like:
```nasm
global_load_dwordx4 v[0:3], v[4:5], off
```

### 2. Coalesced Memory Access

Ensure that threads within a wavefront (64 threads on CDNA) access contiguous memory addresses. When a wavefront issues a memory instruction, the memory controller coalesces requests that fall into the same cache line.
- A fully coalesced read of `float4` by a wavefront (64 threads × 16 bytes = 1024 bytes) is broken down into four 256-byte cache line requests. This perfectly matches the Infinity Fabric data movement granularity and optimally saturates the memory controllers.

### 3. Avoiding Power-of-Two Strides

Matrix transpose operations, strided reductions, or column-wise accesses often use power-of-two strides. Because channel interleaving is power-of-two based, striding by exact multiples of the total interleave width causes requests to route to a single HBM3 channel.

**Solution: Memory Padding**
Introduce padding to the leading dimension of your matrices or tensors to break the power-of-two alignment.

```cpp
// Inefficient: Exact power-of-2 stride causes channel camping
int lda = 4096; // 4096 floats = 16KB
float val = matrix[row * lda + col];

// Optimized: Pad the stride by an offset (e.g., +16 elements)
// Forces successive rows to map to different memory channels
int lda_padded = 4096 + 16; 
float val = matrix[row * lda_padded + col];
```

### 4. Maximizing Memory Level Parallelism (MLP)

HBM3 memory controllers have deep instruction queues. It is critical to keep multiple memory requests in-flight per wavefront to hide latency.

- **Unrolling:** Unroll loops to issue multiple `global_load_dwordx4` instructions before consuming the data.
- **Double Buffering:** Use asynchronous data movement and instruction-level parallelism. Issue loads for the next tile of data, then perform mathematical operations (`v_mfma_f32_32x32x8f16`) on the current tile while the memory controllers fetch the next tile.

```cpp
// Pseudo-code for double-buffered memory fetching
float4 regs_A[2], regs_B[2];

// Prologue: Initiate first loads
regs_A[0] = load_global(ptr_A);
regs_B[0] = load_global(ptr_B);

for (int i = 0; i < K_tiles - 1; ++i) {
    // Issue memory requests for the next tile
    regs_A[(i + 1) % 2] = load_global(ptr_A + offset);
    regs_B[(i + 1) % 2] = load_global(ptr_B + offset);
    
    // Wait ONLY for the current tile's memory loads to complete
    // In HIP inline assembly: asm volatile("s_waitcnt vmcnt(...)");
    wait_for_memory(regs_A[i % 2]); 

    // Compute on current tile using MFMA
    compute_mfma(regs_A[i % 2], regs_B[i % 2]);
}
```

### Performance Data: MI250X vs MI300X

| Metric | MI250X (CDNA2) | MI300X (CDNA3) |
| :--- | :--- | :--- |
| **Memory Type** | HBM2e | HBM3 |
| **Peak Bandwidth** | 3.2 TB/s | 5.3 TB/s |
| **Memory Capacity** | 128 GB | 192 GB |
| **Stacks** | 8 | 8 |
| **Effective Bandwidth (STREAM Triad)** | ~2.5 TB/s | ~4.2 - 4.5 TB/s |

By understanding and optimizing for the HBM3 memory controllers, developers can push their memory-bound kernels (like Flash Attention, RMSNorm, and bespoke reductions) to achieve over 80% of the MI300X's theoretical peak bandwidth.

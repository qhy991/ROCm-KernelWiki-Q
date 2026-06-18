---
id: pattern-compute-bound
title: Compute-Bound Optimization Patterns (算力密集优化模式)
type: wiki-pattern
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, compute, mfma, pipeline, tiling, vgpr]
confidence: source-reported
techniques: [mfma-scheduling, register-tiling, double-buffering]
kernel_types: [gemm, conv]
related: []
sources: []
---

# Compute-Bound Optimization Patterns (算力密集优化模式)

Compute-bound kernels, such as dense Matrix-Matrix Multiplication (GEMM) and Convolutions (Conv), represent the backbone of deep learning workloads. On AMD CDNA architectures (MI250X, MI300X, MI350X), these workloads are characterized by high arithmetic intensity, where the primary bottleneck is the throughput of Matrix Fused Multiply-Add (MFMA) instructions rather than memory bandwidth.

This page discusses the core patterns for optimizing compute-bound kernels on ROCm/HIP, focusing on maximizing MFMA utilization, register tiling strategies, and software pipelining techniques.

## Maximizing MFMA Utilization

The overarching goal for compute-bound kernels is to keep the MFMA units fed with data at all times. AMD CDNA architectures employ specialized matrix cores that execute `v_mfma` instructions.

### MFMA Instruction Characteristics
A typical instruction like `v_mfma_f32_32x32x8f16` computes a 32x32 tile of FP32 results from FP16 inputs. 
- **Throughput**: On CDNA3 (MI300X), each Compute Unit (CU) features a Dual Compute Matrix Accelerator (Dual CMA), significantly increasing the peak FLOP/s per CU compared to CDNA2.
- **Latency**: MFMA instructions have a relatively long execution latency. To achieve peak throughput, the pipeline must be kept full. This means interleaving independent MFMA instructions or overlapping them with memory operations to hide latency.

### Instruction Scheduling Strategies
- **Instruction Level Parallelism (ILP)**: Issue multiple independent MFMA instructions sequentially. This requires accumulating into distinct sets of Vector General Purpose Registers (VGPRs).
- **Interleaving with Vector Memory**: Avoid back-to-back memory instructions if possible. Interleave `v_mfma` instructions with `buffer_load_dwordx4` (Global Memory) and `ds_read_b128` (LDS) instructions.

## Register Tiling (寄存器分块)

Register tiling is the practice of keeping heavily reused data (such as the accumulator matrix $C$ in $C = A \times B$) entirely within VGPRs. Since LDS bandwidth can become a bottleneck if accessed continuously for every compute instruction, register tiling minimizes LDS traffic.

### 2D Register Block Design
For a block of matrix $C$ computed by a single wavefront:
- Data from matrices $A$ and $B$ are loaded from LDS into VGPRs.
- The outer product is accumulated in a large block of VGPRs allocated for $C$.

On CDNA3, you have up to 512 VGPRs available per wavefront (in unified VGPR mode), but maximizing occupancy might require limiting VGPR usage to 256. 
A common pattern for FP16/BF16 GEMM is using a 128x128 or 256x128 threadblock size, where a single wavefront might compute a 64x64 or 32x64 sub-tile.

### Example: MFMA Register Accumulation
```cpp
// Example of register tiling utilizing MFMA in HIP C++ (pseudo-code)
__global__ void gemm_compute_bound(...) {
    // Allocate VGPRs for Accumulators
    float c_regs[32]; // Accommodates the output of a 32x32 MFMA tile (depending on lane mapping)
    
    // Initialize accumulators
    for(int i = 0; i < 32; ++i) c_regs[i] = 0.0f;

    // Loop over the K dimension
    for(int k = 0; k < K; k += K_BLOCK) {
        // Load chunks of A and B from LDS into registers
        // A_regs: fp16 registers, B_regs: fp16 registers
        
        // Execute MFMA
        // v_mfma_f32_32x32x8f16
        c_regs = __builtin_amdgcn_mfma_f32_32x32x8f16(A_regs, B_regs, c_regs, ...);
    }
    
    // Write out c_regs to Global Memory
}
```

## Software Pipelining (软件流水线)

To hide the high latency of Global-to-LDS loads, software pipelining (Double Buffering or Multi-Buffering) is essential.

### Double Buffering with Async Copy
Software pipelining overlaps the fetching of the *next* block of data (from Global Memory to LDS) with the computation of the *current* block (from LDS to VGPRs to MFMA units).

#### Pipeline Stages
1. **Prologue**: 
   - Issue Global loads for `Tile 0`.
   - Wait for loads (`s_waitcnt vmcnt(0)`).
   - Write to LDS Buffer A.
   - Issue Global loads for `Tile 1`.
2. **Main Loop**:
   - For `i` from `0` to `num_tiles - 2`:
       - Wait for Global loads of `Tile i+1`.
       - Write `Tile i+1` to LDS Buffer B (or alternate).
       - Ensure LDS write completion (`s_waitcnt lgkmcnt(0)`).
       - Synchronize wavefronts (`s_barrier`).
       - Compute `Tile i` (read from LDS Buffer A, MFMA).
       - Issue Global loads for `Tile i+2` (if any).
       - Swap LDS Buffer pointers.
3. **Epilogue**:
   - Compute the final tile.
   
### Composable Kernel (CK) Tile API
Writing optimal assembly-level scheduling by hand is notoriously difficult. The Composable Kernel (CK) Tile API provides primitives for these patterns out of the box.

```cpp
// Concept using CK tile API software pipelining
using Shape = ck_tile::BlockShape<128, 128, 32>;
using Pipeline = ck_tile::BlockSoftwarePipeline<
    Shape,
    Stages=2, // Double buffering
    ck_tile::AsyncCopy // Hardware async copy where available
>;

// The pipeline automatically handles the waitcnt and s_barrier synchronization
Pipeline::Run(
    [&](auto tile_a, auto tile_b) {
        // Local compute function (MFMA)
        ck_tile::tile_gemm_mfma<ck_tile::MFMA_32x32x8>(acc_tile, tile_a, tile_b);
    },
    num_k_tiles
);
```

## Performance Tuning on MI250X / MI300X

| Architecture | Peak FP16 TFLOPs | MFMA Size | Recommended Wavefront Count | Typical Shared Memory (LDS) limit |
|--------------|------------------|-----------|-----------------------------|-----------------------------------|
| MI250X       | ~383             | 32x32x8   | 4 - 8 per CU                | 64KB per CU                       |
| MI300X       | ~1300            | 32x32x8   | 8 - 10 per CU               | 64KB per dual CU                  |

**Key Tuning Variables**:
1. **Occupancy**: High VGPR usage limits the number of active wavefronts. Sometimes spilling a few variables to scratch space is acceptable if it allows one more wave per CU, hiding more memory latency.
2. **LDS Bank Conflicts**: Ensure padding is applied when storing data to LDS, particularly for Matrix A and B dimensions, to allow conflict-free 128-bit (`ds_read_b128`) reads.
3. **Wavefront Count**: For pure compute-bound tasks, having at least 4 active wavefronts is required to hide the latency of back-to-back MFMA instructions.

By combining Register Tiling (to keep data locally available) and Software Pipelining (to fetch future data continuously), compute-bound kernels on ROCm can approach 80-90% of the theoretical peak FLOPs hardware limit.

import os

wiki_kernels_path = "/Users/haiyan-mini/Agent4Kernel/rocm-kernelwiki-q/wiki/kernels"
wiki_techniques_path = "/Users/haiyan-mini/Agent4Kernel/rocm-kernelwiki-q/wiki/techniques"

gemm_rocm_content = """---
id: wiki-kernel-gemm-rocm
title: "GEMM Implementation on AMD CDNA"
type: wiki-kernel
architectures: [cdna1, cdna2, cdna3]
tags: [gemm, mfma, lds, register-tiling]
confidence: verified
kernel_types: [gemm]
languages: [hip-cpp, ck-dsl]
reproducibility: snippet
---

# GEMM Implementation on AMD CDNA

General Matrix Multiplication (GEMM) on AMD CDNA architectures fundamentally relies on the Matrix Fused Multiply-Add (MFMA) instructions to achieve peak throughput.

## Hierarchical Tiling Strategy

A high-performance GEMM in HIP follows a strict tiling hierarchy:
1. **Grid Level**: The output matrix $C$ is divided into Thread Block Tiles. Each Compute Unit (CU) computes one or more of these tiles.
2. **Block Level**: The thread block loads tiles of $A$ and $B$ from Global Memory into the Local Data Share (LDS).
3. **Wavefront Level**: The 64-thread wavefront loads sub-tiles from LDS into Vector General-Purpose Registers (VGPRs).
4. **Instruction Level**: The `v_mfma` instructions multiply the registers and accumulate the result into a separate set of VGPRs.

## The MFMA Instruction (Matrix Core)

Unlike NVIDIA's `mma.sync` which requires a full warp, AMD's MFMA operates on a single Wavefront (64 threads). 

Example intrinsic in HIP:
```cpp
// 32x32x8 FP16 -> FP32 MFMA
D_reg = __builtin_amdgcn_mfma_f32_32x32x8f16(A_reg, B_reg, C_reg, cbsz, abid, blgp);
```

### Register Pressure

MFMA outputs (the C matrix accumulators) require a large number of VGPRs. For a `32x32` block, a single wave requires 32 VGPRs just for the C accumulators (FP32). Balancing the tile size to maximize MFMA throughput while avoiding register spilling is the primary challenge in ROCm GEMM tuning.

## Double Buffering & Prefetching

To hide the latency of Global Memory and LDS reads, ROCm GEMM implementations use double buffering:
- Load $A_{k+1}$ and $B_{k+1}$ into LDS while computing MFMA on $A_k$ and $B_k$.
- Use `__builtin_amdgcn_s_waitcnt` to control synchronization explicitly, avoiding over-synchronization.
"""

reduction_rocm_content = """---
id: wiki-kernel-reduction-rocm
title: "Reduction Kernels on ROCm"
type: wiki-kernel
architectures: [cdna1, cdna2, cdna3]
tags: [reduction, wave-reduction, lds]
confidence: verified
kernel_types: [reduction]
languages: [hip-cpp]
reproducibility: concept
---

# Reduction Kernels on ROCm

Reduction is a foundational primitive used heavily in operations like Softmax, LayerNorm, and RMSNorm. In ROCm/HIP, reductions are typically performed in two stages: Wavefront reduction and Thread Block reduction.

## 1. Wavefront-Level Reduction

Because AMD architectures utilize a 64-thread Wavefront, a single wave can reduce 64 elements very efficiently without touching shared memory (LDS).

This is done using Data Parallel Primitives (DPP) or standard warp shuffle operations (`__shfl_down`).

```cpp
template<typename T>
__device__ T wave_reduce_sum(T val) {
    // 64-thread wavefront reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, warpSize);
    }
    return val;
}
```

## 2. Block-Level Reduction

To reduce values across multiple wavefronts within a Thread Block, the LDS (Local Data Share) is required.

**Steps:**
1. Each Wavefront performs a `wave_reduce_sum`.
2. The first thread of each wavefront (`lane_id == 0`) writes its result to LDS.
3. A synchronization barrier (`__syncthreads()`) ensures all waves have written.
4. The first wavefront reads the values from LDS and performs a final `wave_reduce_sum`.

### Performance Considerations
- **Vectorized Loads**: When a single thread is reducing multiple elements from Global Memory, always use vectorized loads (e.g., `float4`) to maximize memory bandwidth before starting the cross-thread reduction.
- **Avoid Atomics**: Unless absolutely necessary, avoid global memory atomics (`atomicAdd`) for the final grid reduction. Instead, write partial results to a temporary buffer and launch a secondary kernel, or use a single-pass atomic strategy with a global counter.
"""

wave_reduction_content = """---
id: wiki-technique-wave-reduction
title: "Wavefront Reduction using DPP"
type: wiki-technique
architectures: [cdna1, cdna2, cdna3]
tags: [wave-reduction, dpp, wavefront]
confidence: verified
---

# Wavefront Reduction using DPP

Data Parallel Primitives (DPP) are AMD-specific ISA features that allow threads within a Wavefront to share data directly via the ALUs, bypassing the VGPR and LDS completely.

## Mechanism

A DPP instruction takes a source register, applies a permutation/shift to the thread lanes, and feeds the result directly into an ALU operation in another lane.

In HIP, standard CUDA-like shuffles (`__shfl_down`, `__shfl_xor`) compile down to these DPP instructions or `ds_bpermute` instructions.

## Example: Butterfly Reduction

The most efficient way to sum 64 elements across a Wavefront is the butterfly reduction pattern.

```cpp
int val = thread_data;
val += __shfl_xor(val, 32);
val += __shfl_xor(val, 16);
val += __shfl_xor(val, 8);
val += __shfl_xor(val, 4);
val += __shfl_xor(val, 2);
val += __shfl_xor(val, 1);
```

At the end of this sequence, *every* thread in the Wavefront holds the total sum. 

## Advantages over LDS
- **Zero Memory Traffic**: No LDS read/write ports are consumed.
- **Lower Latency**: DPP operations execute in the ALU pipeline, which is significantly faster than going through the LDS memory controller.
- **No Synchronization**: Because the instruction is issued to the entire Wavefront simultaneously in SIMD fashion, no explicit barrier is needed between steps.
"""

swizzling_content = """---
id: wiki-technique-swizzling
title: "LDS Address Swizzling"
type: wiki-technique
architectures: [cdna1, cdna2, cdna3]
tags: [swizzling, lds, bank-conflict-padding]
confidence: verified
---

# LDS Address Swizzling

Local Data Share (LDS) on AMD CDNA architectures consists of 32 memory banks. If multiple threads in a Wavefront attempt to access different addresses that map to the same bank simultaneously, a **Bank Conflict** occurs, serializing the access and severely degrading performance.

## The Problem with Padding

The traditional solution to bank conflicts is padding: adding a dummy element at the end of each row.
```cpp
// LDS Array with padding
__shared__ float tile[32][32 + 1]; 
```
While padding works, it wastes valuable LDS capacity. In highly optimized kernels (like GEMM or Flash Attention), LDS capacity dictates the maximum block size and occupancy.

## The XOR Swizzling Solution

Swizzling resolves bank conflicts without wasting memory by using a bitwise XOR operation to scramble the column index based on the row index.

### How it Works

Instead of accessing `tile[row][col]`, we access `tile[row][col ^ row]`.

```cpp
template <int ROW, int COL>
__device__ inline int swizzle_idx(int r, int c) {
    // Basic swizzle pattern
    int swizzled_c = c ^ (r % 32);
    return r * COL + swizzled_c;
}

// Writing to LDS
lds_memory[swizzle_idx<32, 32>(thread_row, thread_col)] = val;
```

### Why it Works
When a Wavefront reads a column (e.g., in a GEMM where one operand is read column-wise), all threads have the same `col` but different `row` values. 
- Without swizzling: `col` is constant, so `address % 32` is constant -> 32-way Bank Conflict!
- With swizzling: `col ^ row` generates a unique value for every row from 0 to 31 -> 0 Bank Conflicts!

## Hardware Support
AMD's `ds_read` and `ds_write` instructions support hardware-level swizzle modifiers, meaning the XOR computation often incurs zero ALU overhead if written correctly using compiler intrinsics or if the compiler pattern-matches the XOR arithmetic.
"""

os.makedirs(wiki_kernels_path, exist_ok=True)
os.makedirs(wiki_techniques_path, exist_ok=True)

with open(os.path.join(wiki_kernels_path, "gemm-rocm.md"), "w") as f:
    f.write(gemm_rocm_content)

with open(os.path.join(wiki_kernels_path, "reduction-rocm.md"), "w") as f:
    f.write(reduction_rocm_content)

with open(os.path.join(wiki_techniques_path, "wave-reduction.md"), "w") as f:
    f.write(wave_reduction_content)

with open(os.path.join(wiki_techniques_path, "swizzling.md"), "w") as f:
    f.write(swizzling_content)

print("ROCm wiki generation script complete.")

---
id: technique-xdlops-programming
title: XDLOPS 底层编程 (XDLOPS Low-level Programming)
type: wiki-technique
architectures: [cdna1, cdna2, cdna3]
tags: [mfma, isa, hip, programming, optimization, vgpr, ck-tile]
confidence: source-reported
techniques: [mfma-scheduling, register-tiling]
hardware_features: [mfma, wavefront]
kernel_types: [gemm]
related: []
sources: []
reproducibility: snippet
---

# XDLOPS 底层编程 (XDLOPS Low-level Programming)

## Overview
AMD's XDLOPS (eXtreme Data-parallel Long OPerationS) is the hardware engine behind the Matrix Cores introduced in the CDNA architecture. XDLOPS provides native matrix-multiply-accumulate operations in a single instruction, primarily through the `v_mfma` (Matrix Fused Multiply-Add) instruction family.

Unlike NVIDIA's Tensor Cores which operate at a Warp (32 threads) level, XDLOPS operates at the Wavefront (64 threads) level. A single `v_mfma` instruction issued by a wavefront executes a block matrix multiplication $C = A \times B + C$.

## `v_mfma` Instruction Family

The `v_mfma` instructions take the general form:
`v_mfma_<type>_<M>x<N>x<K><input_type> v[c], v[a], v[b], v[c], imm, imm, imm`

Where:
- `<type>`: The output and accumulation data type (e.g., `f32`, `i32`).
- `<M>x<N>x<K>`: The block size dimension for the matrix multiplication.
- `<input_type>`: The input data type for A and B matrices (e.g., `f16`, `bf16`, `i8`, `f8`).

### Common Block Sizes

1. **32x32x8 (e.g., `v_mfma_f32_32x32x8f16`)**
   - Accumulator matrix $C$: 32x32 `f32`
   - Input matrix $A$: 32x8 `f16`
   - Input matrix $B$: 8x32 `f16`
   - Typically used in GEMM kernels for optimal VGPR utilization and throughput on CDNA1/CDNA2.

2. **16x16x16 (e.g., `v_mfma_f32_16x16x16f16`)**
   - Accumulator matrix $C$: 16x16 `f32`
   - Input matrix $A$: 16x16 `f16`
   - Input matrix $B$: 16x16 `f16`
   - Preferred on CDNA3 for dual-CMA usage.

3. **4x4x4 (e.g., `v_mfma_f32_4x4x4f16`)**
   - Smaller grain size, rarely used directly in high-performance GEMM, mostly for specific alignments.

In CDNA3 (MI300), 16x16x16 is natively supported with a dual CMA (Compute Matrix Accelerator) execution, making 16x16x16 blocks faster than using two 32x32x8 blocks for the equivalent FLOPS.

## Register Layout for A, B, and C Matrices

A critical aspect of XDLOPS programming is understanding how the elements of the matrices are distributed across the 64 threads of a wavefront (the VGPR layout). This dictates how data must be loaded from LDS or Global Memory into the VGPRs before calling the `v_mfma` instruction.

### The 32x32x8 Layout (`v_mfma_f32_32x32x8f16`)

#### Matrix C (Accumulator)
A 32x32 matrix of `f32` contains 1024 elements. Distributed across 64 threads, each thread holds 16 `f32` elements (16 VGPRs).
The layout is blocked and scattered. Thread $i$ (where $i \in [0, 63]$) holds 4 blocks of 4 contiguous elements, spread across the matrix dimensions.

#### Matrix A
A 32x8 matrix of `f16` contains 256 elements. However, since each thread provides a portion of the input, the thread mapping involves specific duplication or grouping.
For `v_mfma_f32_32x32x8f16`:
- Each thread provides 4 `f16` elements (2 VGPRs, effectively a `b16x2` vector).
- The 64 threads together provide elements that span the rows of A.
- Threads are grouped: Threads 0-31 provide the first block of elements, and Threads 32-63 replicate or provide the next sub-block depending on the specific microarchitecture rules for `32x32` broadcast.

#### Matrix B
An 8x32 matrix of `f16` contains 256 elements.
- Similar to Matrix A, each thread provides 4 `f16` elements.
- The layout is transposed relative to A. Threads 0-31 hold the columns of B.

## HIP C++ Inline Assembly Example

To squeeze maximum performance and avoid compiler register spilling, developers sometimes use inline assembly:

```cpp
// Example: 32x32x8 f16 -> f32 MFMA
__device__ void mfma_32x32x8f16(float* c, const float2* a, const float2* b) {
    asm volatile(
        "v_mfma_f32_32x32x8f16 %0, %1, %2, %0\n"
        : "+v"(c[0]), "+v"(c[1]), "+v"(c[2]), "+v"(c[3]),
          "+v"(c[4]), "+v"(c[5]), "+v"(c[6]), "+v"(c[7]),
          "+v"(c[8]), "+v"(c[9]), "+v"(c[10]), "+v"(c[11]),
          "+v"(c[12]), "+v"(c[13]), "+v"(c[14]), "+v"(c[15])
        : "v"(*a), "v"(*b)
    );
}
```

*Note: In modern ROCm, using `__builtin_amdgcn_mfma_f32_32x32x8f16` is standard, but manual register allocation using `asm volatile` guarantees specific VGPR layouts.*

## Register Blocking and LDS Usage

To feed the MFMA units efficiently, a typical GEMM kernel uses **Register Blocking**:
1. Load a large tile of A and B from Global Memory to LDS (e.g., 256x128).
2. Load a sub-tile from LDS into VGPRs (e.g., 32x32x8) for A and B.
3. Execute `v_mfma`.
4. Iterate over the K dimension.

Because of the specific VGPR layout of A and B required by `v_mfma`, the data in LDS must be written and read in a swizzled pattern to prevent LDS bank conflicts. Tools like Composable Kernel (CK) tile API abstract this away:

```cpp
// CK Tile API abstraction over XDLOPS
using Shape = ck_tile::Sequence<32, 32, 8>;
using MFMA = ck_tile::WarpMultF16F16F32<Shape>;
auto c_mac = MFMA::Execute(a_vgpr, b_vgpr, c_mac);
```

## Performance Considerations (MI250X vs MI300X)

| Architecture | MFMA Instruction | Max Throughput / CU (OPS/cycle) | Best Practice |
|--------------|------------------|---------------------------------|---------------|
| CDNA2 (MI250X) | 32x32x8 f16 | 1024 | Use 32x32x8 for high occupancy |
| CDNA3 (MI300X) | 16x16x16 f16 | 2048 | Use 16x16x16 to leverage dual CMA |
| CDNA3 (MI300X) | 32x32x8 fp8  | 4096 | Use for 8-bit precision limits |

When migrating from CDNA2 to CDNA3, it is highly recommended to switch the basic MFMA block from `32x32x8` to `16x16x16` to fully utilize the MI300X's dual CMA (Compute Matrix Accelerator) architecture, avoiding pipeline stalls and achieving ~2x performance.

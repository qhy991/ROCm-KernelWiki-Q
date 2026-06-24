---
id: technique-pr-triton-621
title: "Explicit Multiply-Reduce GEMM for Small Block Sizes in Triton"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, memory-bound, compute, rocm, vgpr]
confidence: inferred
kernel_types: [gemm]
languages: [triton-rocm]
techniques: [register-tiling]
hardware_features: [wavefront, mfma]
sources: [pr-triton-621]
---

# Explicit Multiply-Reduce GEMM for Small Block Sizes in Triton

## Context and Motivation
In standard Matrix Multiplication (GEMM) lowered by the Triton compiler on AMD CDNA architectures, the default behavior typically targets Matrix Fused Multiply-Add (MFMA) instructions. These `v_mfma_*` instructions map naturally to AMD's matrix core units and offer peak throughput for block sizes like `32x32`, `16x16`, and `64x64`. 

However, for **small block sizes** (e.g., highly memory-bound tasks where tile dimensions like $M$, $N$, or $K$ are very small, or irregular reductions), relying on MFMA can be suboptimal:
1. **Underutilization**: The fixed dimensions of MFMA hardware require padding if matrix sizes don't perfectly align. This wastes vector ALUs in the wavefront.
2. **Data Layout Overhead**: MFMA expects data to be staged in a very specific memory layout in LDS or registers before the computation. The overhead to arrange data into this layout outweighs the computational benefits when the tile is tiny.
3. **Register Pressure**: Using MFMA requires allocating registers for accumulator matrices, sometimes leading to inefficient VGPR usage if the block is too small to saturate the units.

## The Technique: Explicit Multiply-Reduce
To overcome the limitations of MFMA on small blocks, PR #621 implements an explicit **multiply-reduce** fallback in the Triton kernel. Instead of mapping `tl.dot()` to the matrix cores (MFMA), the compiler explicitly generates vector multiply and reduction instructions via the standard vector ALUs.

### How it works
1. **Bypassing Matrix Cores:** Instead of invoking matrix operations, the operation is explicitly implemented as a sequence of Element-wise Multiplications (e.g., `v_mul` or `v_mac`/`v_fma`) followed by intra-wave or cross-wave reductions.
2. **Register Tiling / Accumulation:** Matrix operands $A$ and $B$ are kept in VGPRs without being forced into MFMA-compatible data formats. Dot products are computed per thread and accumulated in registers.
3. **Reduction:** Standard tree-reductions (using Data Parallel Primitives `dpp` or `ds_bpermute` where necessary) resolve the final sums for matrix $C$.

### Memory Bounds and Performance Implications
- **Memory-Bound Kernels:** Small block sizes naturally imply low arithmetic intensity (low FLOP-to-byte ratio). For these memory-bound regimes, compute throughput is not the bottleneck. By avoiding the rigid data layout and synchronization overhead required by MFMA, explicit multiply-reduce saves latency and improves overall kernel performance.
- **Wavefront Occupancy:** Removing the constraints of LDS staging (which MFMA often needs to properly format data from global memory) reduces LDS usage. Lower LDS usage allows the scheduler to increase wavefront occupancy per Compute Unit (CU), further hiding global memory latency.
- **Flexibility:** Explicit multiply-reduce logic gracefully handles non-power-of-two sizes and very asymmetric matrices without massive padding overhead.

## Implementation Details (Triton)
When authoring or using Triton for small kernels:
- You may use a custom template or direct explicit operations (like `tl.sum(a * b, axis=...)`) over standard dot operations (`tl.dot`) if you know the block size falls significantly below standard MFMA thresholds.
- Alternatively, Triton's ROCm backend (through PRs like #621) integrates lowering paths that automatically detect small block dimensions and emit explicit multiply-reduce MLIR/LLVM dialects rather than forcing matrix core ops. 

## Summary
By using explicit multiply-reduce instructions (standard ALUs) for small block sizes, Triton on ROCm optimizes latency, VGPR utilization, and wavefront occupancy in memory-bound execution regimes, avoiding the layout and padding overheads inherent to matrix core (MFMA) execution.

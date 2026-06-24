---
id: technique-pr-triton-635
title: "Utility Tools: Layout Plotting for Triton MLIR on ROCm"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
hardware_features:
  - lds
  - mfma
languages:
  - triton-rocm
tags:
  - optimization
  - rocm
  - lds
  - mfma
confidence: inferred
sources:
  - pr-triton-635
---

# Analysis of PR #635 in Triton: Plot Layout Script

## Summary
PR #635 introduces the migration of utility tools—specifically a plot layout script—from the `triton-mlir` branch to the `main_perf` branch in the ROCm Triton repository. Visualizing MLIR layouts is critical for understanding memory distributions and optimizing layout transitions, particularly when mapping tensors to AMD's matrix cores (MFMA) and Local Data Share (LDS).

## Intent and Context
Triton compiles high-level Python code to MLIR, which uses advanced layout systems to map logical multi-dimensional grids onto physical hardware threads and memory banks. On ROCm architectures (such as CDNA2 and CDNA3), optimizing these layouts manually or understanding the compiler's automatic choices requires introspection tools.

The **Plot layout script** gives developers a visual representation of how threads own data elements within a tensor. It is vital for:
1. **LDS Bank Conflict Analysis**: Ensuring that threads within a wavefront do not hit the same LDS bank simultaneously when loading/storing data.
2. **MFMA Layout Mapping**: Aligning the dot operand layouts with what the `v_mfma` instructions expect. The layout script helps verify if a block layout correctly matches the required MFMA shape (e.g., `16x16x16` or `32x32x8`).
3. **Coalesced Global Memory Access**: Visualizing the thread mapping to verify contiguous memory access across the wavefront.

## Deep Technical Analysis

### Understanding Triton Layouts on ROCm
In Triton MLIR, layouts dictate how a tensor is distributed among threads. Key layout types include:
- **Blocked Layout**: Distributes data elements among threads in a structured, often multi-dimensional, blocked manner. Critical for global memory loads/stores to ensure vectorization and coalescing.
- **Shared Layout**: Represents how data is placed into LDS. On ROCm, `SharedLayout` takes into account swizzling to prevent LDS bank conflicts.
- **DotOperand / MFMA Layout**: Specific layouts required for feeding matrix operations. In AMD ROCm, MFMA instructions demand very particular data ownership across the wavefront. Misalignment leads to expensive layout conversion (shuffle/LDS fallback) penalties.

### The Role of Layout Visualization
When tuning a kernel for peak performance on CDNA hardware, developers often struggle to perceive whether a chosen tile shape or memory pipeline introduces implicit layout conversions. The migrated plotting utility extracts the layout attributes from MLIR (such as `#triton_gpu.blocked` or `#triton_gpu.mfma`) and plots them visually. 

By plotting the layout:
- **X and Y dimensions**: Map to logical tensor dimensions.
- **Coloring/Annotations**: Indicate thread IDs within the wavefront (0-63) or across wavefronts within the threadblock.

This enables immediate diagnosis of:
- **Inefficient Swizzling**: Identifying patterns where memory access strides lead to LDS bank conflicts.
- **Register Pressure**: Highlighting cases where layout shapes force large allocations of VGPRs for intermediate conversions.
- **Memory Bounds Optimization**: By aligning the blocked layout sizes and strides, developers ensure the memory requests can be mapped to 128-bit vectorized loads, heavily saturating the HBM bandwidth.

## Optimization Technique: Layout Tuning
1. **Trace MLIR**: Output the MLIR at the `ttg` (Triton GPU) dialect level.
2. **Visualize**: Use the plot layout script on the dumped MLIR layouts.
3. **Identify Conversions**: Look for `ttg.convert_layout` operations in the MLIR. If a layout conversion happens in an inner loop, inspect the source and destination layouts.
4. **Tune Tile Sizes & Orders**: Adjust the block sizes, `order` attributes, and `warps_per_cta` in the Python frontend to match the target layout, minimizing or eliminating conversions.

## Conclusion
While ostensibly a utility script, the migration of layout plotting to the main performance branch acknowledges its necessity for deep kernel optimization on AMD GPUs. It equips kernel engineers with the visibility needed to tailor their Triton operations for optimal MFMA mapping and LDS access patterns, pushing kernels closer to the theoretical HBM bandwidth and matrix core compute bounds.

---
id: technique-pr-triton-639
title: "Online Softmax Implementation in Triton"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - optimization
  - memory-bound
  - tiling
  - vgpr
  - occupancy
kernel_types:
  - softmax
languages:
  - triton-rocm
confidence: inferred
sources:
  - pr-triton-639
---

# Online Softmax Implementation in Triton

## Overview
PR #639 in `ROCm/triton` updates the performance kernel for softmax (`python/perf-kernels/softmax.py`) to use an **online normalizer calculation** (often referred to as online softmax or FlashAttention-style safe softmax). This technique computes the necessary reduction metrics (maximum value and exponential sum) in a block-wise fashion instead of loading the entire sequence/row into on-chip memory at once. 

## Context and Intent
The previous implementation of the softmax kernel in Triton instantiated a block size strictly as `triton.next_power_of_2(n_cols)` and loaded the entire row into memory for processing:
```python
# Old approach (pseudo-code)
row = tl.load(input_ptrs, ...)
row_minus_max = row - tl.max(row, axis=0)
numerator = tl.exp(row_minus_max)
...
```
For very large sequence lengths (e.g., $N = 131072$), loading the entire row significantly exceeds register limits and SRAM capacity. It attempts to load `131072` elements per row directly into VGPRs, leading to compilation failure, register spilling, or out-of-memory errors. The intent of this PR is to make the kernel scalable and memory-bounded by processing elements in smaller, fixed-size chunks.

## Architectural and Optimization Details

### Online Softmax Algorithm
The PR refactors the kernel to iterate over the row twice, maintaining running variables for the block maximum and exponential sum.

1. **Loop 1 - Online Reduction**:
   Iterates over the row in chunks of `BLOCK_SIZE`.
   For each chunk:
   - Loads the chunk from global memory (`cache_modifier=".cg"`).
   - Finds the local max `m_p`.
   - Computes the new global max: $m_{new} = \max(m, m_p)$.
   - Rescales the running sum using the difference between the old and new max: $sum_{new} = sum_{old} \times \exp(m - m_{new})$.
   - Adds the new chunk's exponentiated values to the sum: $sum_{new} += \sum \exp(row\_block - m_{new})$.
   - Updates the global max $m = m_{new}$.

2. **Loop 2 - Materialize Softmax**:
   Re-reads the row in chunks of `BLOCK_SIZE` from global memory.
   For each chunk:
   - Loads the chunk from memory.
   - Computes the final softmax output using the global max and global sum: $\text{output} = \exp(row\_block - m) / sum$.
   - Stores the result back to global memory.

### Bounded Memory and Register Usage
By slicing the columns into blocks, the block size is capped by the available `MAX_FUSED_SIZE`:
```python
MAX_FUSED_SIZE = 65536 // x.element_size()
BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
```
This forces the kernel to process elements at most `MAX_FUSED_SIZE` at a time. It dramatically decreases peak VGPR (Vector General Purpose Register) pressure and avoids catastrophic register spilling or LDS over-allocation for extremely large sequence sizes (e.g., $N=131072$). Register usage is now constrained entirely by `BLOCK_SIZE`.

### Removing Persistent Kernel Scheduling
Interestingly, the PR also reverts a persistent kernel scheduling approach.
- **Before:** `num_programs = min(NUM_SM, n_rows)` with an inner persistent loop: `for row_idx in tl.range(row_start, n_rows, row_step):`
- **After:** `num_programs = n_rows` with a standard 1:1 mapping: `row_idx = tl.program_id(0)`.

This relies on the hardware scheduler to dispatch row programs rather than statically keeping a small set of persistent thread blocks alive. The removal of the persistent loop simplifies the kernel, and the reduction in register footprint per workgroup means that more workgroups can run concurrently, organically improving occupancy.

## Performance Profile
- **Algorithmic Bounds:** The kernel remains heavily memory-bound. Executing two passes over the data means it reads the input tensor twice and writes the output tensor once.
- **Cache Strategy:** The `tl.load` uses `cache_modifier=".cg"` (Cache Global) in both loops, bypassing the L1 cache. This avoids cache thrashing when processing massive sequences (where the row vastly exceeds L1 size), and prevents the eviction of other useful data since the memory access pattern is perfectly linear.

## Reproduction and Testing
The PR successfully adds validation for very large inputs (e.g., `(1, 131072)`, `(1, 89999)`). This confirms that the scaling limitations and compilation limits triggered by the old single-block approach have been fully resolved.

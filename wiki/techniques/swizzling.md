---
id: technique-swizzling
title: "LDS Address Swizzling"
type: wiki-technique
architectures: [cdna1, cdna2, cdna3]
tags: [swizzling, lds, bank-conflict-padding]
confidence: verified
sources: []
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

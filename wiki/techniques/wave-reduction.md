---
id: technique-wave-reduction
title: "Wavefront Reduction using DPP"
type: wiki-technique
architectures: [cdna1, cdna2, cdna3]
tags: [wave-reduction, dpp, wavefront]
confidence: verified
sources: []
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

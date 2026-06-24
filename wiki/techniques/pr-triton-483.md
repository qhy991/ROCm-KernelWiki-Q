---
id: technique-pr-triton-483
title: "AMD-Specific TTGIR Scheduling Pass for Flash Attention and Dot Slicing"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - mfma-scheduling
  - scheduling
  - optimization
  - pipeline
  - rocm-kernel
  - flash-attention
kernel_types:
  - flash-attention
languages:
  - triton-rocm
confidence: inferred
sources:
  - pr-triton-483
---

# AMD-Specific TTGIR Scheduling Pass for Flash Attention and Dot Slicing

## Overview

PR #483 in the ROCm/triton repository introduces an AMD-specific scheduling pass at the Triton GPU Intermediate Representation (TTGIR) level. This pass tackles two critical performance bottlenecks in Triton-generated code for AMD CDNA architectures:

1. **Loop-Invariant Code Motion (LICM) for the Q Tensor in Flash Attention:** Hoisting query tensor operations out of the critical path of the inner loop.
2. **Dot Slicing Instruction Scheduling:** Optimizing the execution order of `mfma` (Matrix Fused Multiply-Add) instructions generated when large block matrix multiplications (dots) are decomposed into hardware-native shapes.

## Architectural Context

Triton lowers its high-level MLIR dialect to TTGIR before finally lowering to LLVM IR. At the TTGIR level, operations still retain tensor-level semantics, but hardware-specific layout conversions and matrix operations have been introduced. By adding a dedicated scheduling pass here, the AMD backend can reorder instructions to better hide latency and minimize register pressure before LLVM's general-purpose instruction scheduler takes over.

### 1. Hoisting the Q Tensor in Flash Attention

In the forward pass of Flash Attention, the algorithm computes attention scores block-by-block. For a given block of the Query ($Q$), the algorithm iterates over blocks of Key ($K$) and Value ($V$):

```python
# Conceptual Flash Attention inner loop
Q_block = load(Q)
for K_block, V_block in KV_sequence:
    S = dot(Q_block, K_block.T)  # Q is invariant
    # ... computation continues
```

Without explicit scheduling or loop-invariant code motion (LICM) at the TTGIR level, address calculations, layout conversions, or redundant loads associated with the `Q` tensor might leak into the inner loop. 

**Technique Implemented:**
The AMD scheduling pass analyzes the TTGIR to identify instructions related to the `Q` tensor that do not depend on the loop's induction variable. By **hoisting** these instructions outside the loop, the compiler achieves:
* **Reduced Loop Overhead:** Fewer instructions executed per iteration.
* **Lower VGPR Pressure:** Temporary registers used for invariant pointer math or layout conversions can be freed or reused.
* **Higher Instruction Bandwidth:** The instruction cache and fetch units can focus entirely on the critical inner loop instructions (mostly `mfma` and global/LDS memory loads for $K$ and $V$).

### 2. Scheduling Dot Slicing Instructions (`mfma-scheduling`)

Triton's `dot` operations define large tile-level matrix multiplications (e.g., $128 \times 128 \times 32$). AMD CDNA hardware Matrix Cores operate on specific, fixed shapes (e.g., $32 \times 32 \times 8$ for `v_mfma_f32_32x32x8f16`). 

The "Dot Slicing" pass decomposes a large Triton `dot` into a sequence of smaller hardware-native `mfma` instructions.

**The Scheduling Challenge:**
Executing a straight-line sequence of `mfma` instructions without considering memory dependencies leads to sub-optimal utilization of the Matrix Cores. There are inherent data dependencies between the LDS loads that feed the `A` and `B` operands and the `mfma` operations themselves. 

**Technique Implemented:**
The scheduling pass interleaves the sliced `mfma` instructions with other operations (like memory loads and layout conversions). This is a form of software pipelining and **MFMA scheduling** applied at the TTGIR level:
* **Latency Hiding:** By placing independent instructions between an LDS load and its corresponding `mfma`, the compiler hides the memory access latency.
* **Pipeline Utilization:** Ensuring the Matrix Cores are continuously fed with independent math instructions, maximizing TFLOPS and overall occupancy.

## Impact on ROCm Developers

For developers writing Triton kernels targeting MI200 and MI300 series GPUs:
* **Flash Attention:** Expect immediate performance improvements in the forward pass of Flash Attention (and similar algorithms where one operand is stationary) due to tighter inner loops.
* **Large Matrix Multiplications:** Any kernel utilizing large `tl.dot` operations will benefit from the optimized instruction scheduling, translating to higher sustained arithmetic intensity and closer-to-peak hardware utilization.

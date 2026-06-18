---
id: technique-pr-triton-455
title: "Decoupling AMDGPU Backend from NVIDIA TMA Dialects in Triton"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - triton-rocm
  - rocm-kernel
  - optimization
  - hardware
confidence: verified
sources:
  - pr-triton-455
---

# Decoupling AMDGPU Backend from NVIDIA TMA Dialects in Triton

## Overview

In Triton PR #455, a significant architectural cleanup was performed in the AMD backend to remove legacy dependencies on the `nvidia_gpu` dialect. Specifically, this PR targets the removal of Tensor Memory Accelerator (TMA) operations, which are hardware features exclusive to NVIDIA Hopper (sm90) and newer architectures. Because AMD CDNA architectures do not have a direct equivalent to TMA, tracking and processing these operations within the `TritonAMDGPUToLLVM` conversion passes was unnecessary and created a tight coupling between backends.

## Architectural Context

### Tensor Memory Accelerator (TMA)
TMA is a specialized hardware unit introduced in NVIDIA's Hopper architecture that accelerates asynchronous data movement between global memory and shared memory. In Triton's intermediate representation (IR), TMA interactions are modeled using the `nvidia_gpu` dialect with operations such as:
- `triton::nvidia_gpu::InsertSliceTMAOp`
- `triton::nvidia_gpu::StoreAsyncOp`

### AMD's Architecture
AMD's CDNA architectures (e.g., MI250, MI300) rely on different asynchronous copy mechanisms (such as explicit `async-copy` using `buffer_load` or specialized equivalents for global-to-LDS transfers). Since TMA is physically non-existent on AMD hardware, the presence of TMA lowering and metadata tracking logic within the AMD backend was technical debt inherited from codebase forks or early multi-backend abstractions.

## Key Changes in PR #455

### 1. Removal of TMA Operation Conversion
The PR removed the registration of `InsertSliceTMAOpConversion` and `StoreAsyncOpConversion` from the `populateLoadStoreOpToLLVMPatterns` function in `lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp`. This ensures the AMD backend no longer attempts to convert these NVIDIA-specific operations to LLVM IR.

### 2. Elimination of TMA Metadata Tracking
Previously, the `TritonGPUToLLVMPass.cpp` for AMD contained logic to collect TMA information:
- It counted the number of TMA loads (`numTMALoad`) and stores (`numTMAStore`) by traversing the function for `InsertSliceTMAOp` and `StoreAsyncOp`.
- It appended arguments to the generated LLVM function to pass TMA descriptors.
- It attached `kAttrNumTMALoadDescsName` and `kAttrNumTMAStoreDescsName` attributes to the generated LLVM functions.

All of this logic was entirely purged from the AMD `FuncOpConversion` and `ConvertTritonGPUToLLVM` passes.

### 3. Cleanup of `TensorPtrMapT`
The `tensorPtrMap`, which mapped TMA operations to their corresponding `MakeTensorPtrOp` descriptors, was removed from the AMD backend. Since `MakeTensorPtrOp` is primarily used to create the descriptor required by the TMA hardware, excising this mapping strictly decouples the AMD backend from NVIDIA's memory pointer abstraction.

### 4. Continuous Integration Improvements
In addition to the dialect cleanup, this PR establishes a new GitHub Actions workflow (`amd-offline-tests.yml`) to perform offline tests specifically for the AMD backend. This helps ensure that future commits do not break AMD-specific compilation paths or reintroduce backend-coupled logic.

## Impact

1. **Backend Isolation**: Ensures that the `TritonAMDGPUToLLVM` compiler pipeline does not crash or exhibit undefined behavior when interacting with an IR that incorrectly mixes AMD and NVIDIA-specific dialects.
2. **Compiler Optimization**: Reduces compile-time overhead by preventing the compiler from walking the IR multiple times to discover operations that should never be present in a valid AMD-targeted IR.
3. **Clean Codebase Boundary**: Sets a clear precedent for future hardware-specific features, emphasizing that `nvidia_gpu` dialect operations must be handled prior to or strictly outside of AMD backend compiler passes.

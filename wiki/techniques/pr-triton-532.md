---
id: pr-triton-532
title: "Triton DotSlicing: Support for Multiple Load Operands"
type: source-pr
repo: ROCm/triton
pr: 532
author: htyu
date: '2024-03-24'
url: https://github.com/ROCm/triton/pull/532
source_category: upstream-code
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, triton, mfma, memory, optimization]
confidence: source-reported
---

# Triton DotSlicing: Support for Multiple Load Operands

## Background

In Triton, large matrix multiplications (`tt.dot`) are often mapped to AMD's Matrix Fused Multiply-Add (`MFMA`) instructions. When a dot operation is larger than what the target hardware can efficiently execute in a single instruction or macro-block, the compiler may decompose it into a sequence of smaller dot operations. In the ROCm fork of Triton, this transformation is handled by the **DotSlicing** pass (`lib/Dialect/TritonGPU/Transforms/DotSlicing.cpp`).

When a `tt.dot` is sliced, its corresponding input operations, such as memory loads (`tt.load`), must also be sliced into matching chunk sizes (e.g., taking a `128x128` load and breaking it into four `128x32` loads).

## The Problem

A standard `tt.load` operation can take multiple operands:
1. **Pointers (`ptr`)**: The memory addresses to read from.
2. **Mask (`mask`)**: An optional boolean tensor used for bounds checking or padding.
3. **Other (`other`)**: An optional tensor providing default values for masked-out elements.

Prior to PR #532, the `DotSlicing` pass only sliced the *first* operand (the pointers) of the operation being sliced. If a load operation included a `mask` (which shares the same shape and layout as the pointer tensor), the mask was passed to the sliced load without being sliced. 

For instance, slicing a `128x128` masked load down to a `128x32` slice resulted in a new `tt.load` that attempted to use a `128x32` pointer tensor alongside the original `128x128` mask tensor. This shape mismatch caused compilation to fail.

## The Solution

PR #532 fixes this issue by iterating through all operands of the operation being sliced. If an operand has a tensor type that matches the layout of the sliced operation, the pass now generates a corresponding `triton_gpu.view_slice` for it.

### Code Changes

```cpp
// Before
auto slice = builder.create<triton::gpu::ViewSliceOp>(
    firstOpToSlice->getLoc(), slicedType, viewPtr,
    ValueRange({}), ValueRange({}), ValueRange({}), sliceOffsets,
    sliceSizes, sliceStrides);
mapping.map(firstOpToSlice->getResult(0), slice);

// After
for (auto arg : firstOpToSlice->getOperands()) {
  auto argType = dyn_cast<RankedTensorType>(arg.getType());
  if (argType && argType.getEncoding() == layout) {
    auto slicedArgType = RankedTensorType::get(
        slicedType.getShape(), argType.getElementType(), layout);
    auto slice = builder.create<triton::gpu::ViewSliceOp>(
        firstOpToSlice->getLoc(), slicedArgType, arg,
        ValueRange({}), ValueRange({}), ValueRange({}), sliceOffsets,
        sliceSizes, sliceStrides);
    mapping.map(arg, slice);
    if (i == 0)
      viewPtr = slice;
  }
}
```

### MLIR Transformation Example

With this fix, a masked load for a `128x128` dot operation is correctly decomposed into sliced pointers **and** sliced masks. In the following example, a `128x128` `tt.load` is sliced along the K dimension into chunks of `32`:

```mlir
// Slicing the pointer tensor
%Q_VIEW_SLICE_1 = triton_gpu.view_slice %Q_PTR[0, 0] [128, 32] [1, 1] 
    : tensor<128x128x!tt.ptr<f16, 1>, #LAYOUT> to tensor<128x32x!tt.ptr<f16, 1>, #LAYOUT>

// Slicing the mask tensor (Added by this PR)
%Q_MASK_SLICE_1 = triton_gpu.view_slice %Q_MASK[0, 0] [128, 32] [1, 1] 
    : tensor<128x128xi1, #LAYOUT> to tensor<128x32xi1, #LAYOUT>

// The sliced load now uses both sliced operands
%LOAD_Q_1 = tt.load %Q_VIEW_SLICE_1, %Q_MASK_SLICE_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} 
    : tensor<128x32xf16, #LAYOUT>
```

## Architectural Impact

This compiler infrastructure fix is crucial for kernels that rely on **masked loads** alongside large blocked layouts that trigger dot slicing. Examples include:
- **Causal Attention**: Where sequence masks are applied to block loads.
- **Variable Sequence Lengths**: Padding elements out of bounds in Flash Attention.
- Kernels running on AMD architectures (CDNA2, CDNA3, CDNA4) where `MFMA` instruction widths necessitate slicing of large dot operations.

By properly mapping the `mask` operand, ROCm Triton successfully compiles masked GEMMs and attention kernels that span multiple MFMA instructions.

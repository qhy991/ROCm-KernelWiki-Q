---
id: migration-cutlass-to-ck
title: CUTLASS → CK Migration Guide
type: wiki-migration
architectures: [cdna2, cdna3, cdna4]
tags: [migration, composable_kernel, ck-tile, gemm]
confidence: source-reported
from_architecture: cuda
to_architecture: cdna3
difficulty: moderate
related: [hw-mfma, technique-ck-tile-programming]
sources: []
---

# CUTLASS → CK Migration Guide

Porting high-performance matrix multiplication kernels from NVIDIA's CUTLASS to AMD's Composable Kernel (CK) requires understanding the structural mapping between the two template libraries. While both rely on C++ template metaprogramming to abstract the GPU hierarchy, CK introduces specific concepts tailored to AMD's CDNA architectures, such as Matrix Fused Multiply-Add (MFMA) instructions and unique tensor coordinate transformations.

## 1. Hierarchy Mapping

Both CUTLASS and CK decompose GEMM operations into a hierarchical execution model. However, terminology and mapping to hardware differ:

| Concept | CUTLASS | Composable Kernel (CK) | AMD Hardware Equivalent |
|---------|---------|------------------------|-------------------------|
| **Grid** | `Threadblock` | `Blockwise` | Workgroup (Compute Unit) |
| **Warp** | `Warp` | `Wavewise` / `Wavefront` | Wavefront (64 threads) |
| **Thread**| `Instruction` / `Thread` | `Threadwise` | Single Thread (VGPRs) |
| **Math** | `TensorOp` | `MFMA` | Matrix Core (v_mfma) |

### Key Differences in Wavefront Size
CUTLASS assumes a warp size of 32 threads. CK is designed for AMD CDNA architectures where the wavefront size is 64 threads. This fundamentally changes tile sizing and register allocation. An MFMA instruction typically uses 64 threads to compute a 32x32x8 block (e.g., `v_mfma_f32_32x32x8f16`), requiring tile dimensions to be scaled accordingly.

## 2. Tile Iterators and Data Movement

In CUTLASS, data movement is handled by Tile Iterators (e.g., `PredicatedTileIterator`). In CK, this is managed by **TensorSliceTransfers** combined with **TensorDescriptors**.

### CUTLASS: Tile Iterator
```cpp
using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
    cutlass::MatrixShape<M, K>,
    ElementA, LayoutA, 1, ThreadMapA>;
```

### CK: Tensor Descriptor and Slice Transfer
CK separates the logical layout (`TensorDescriptor`) from the data movement (`TensorSliceTransfer`).

```cpp
// 1. Define the logical layout using transformations
auto desc_a = make_naive_tensor_descriptor(
    make_tuple(M, K),
    make_tuple(StrideM, StrideK)
);

// 2. Create the data transfer operation (Global -> LDS)
using BlockwiseTransferA = ThreadGroupTensorSliceTransfer_v4r1<
    ThreadClusterLength,  // Threadblock dimensions
    SliceLength,          // Tile size (e.g., MBlock x KBlock)
    ElementA,             // Data type
    AddressSpace::Global, // Source
    AddressSpace::Lds,    // Destination
    ...>;
```

**Mental Model Shift**: Instead of configuring an "Iterator" with a layout and thread map, CK defines a `TensorDescriptor` (which supports complex affine and non-affine transformations) and applies a `SliceTransfer` that optimally maps threads to the descriptor to load a specific "slice" of the tensor.

## 3. Threadblock Layouts (Swizzling)

Bank conflicts in Shared Memory (LDS) are a major bottleneck. CUTLASS mitigates this using layout tags like `layout::RowMajorTensorOpMultiplicandCrosswise`. CK handles this via `Transform` operations on the `TensorDescriptor`.

### CUTLASS: Crosswise Layout
```cpp
using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>;
```

### CK: XOR Swizzling in Descriptors
CK uses coordinate transformations (like `RightShift` and `XOR`) directly in the descriptor to achieve bank-conflict-free layouts.

```cpp
auto lds_desc_a = make_naive_tensor_descriptor(make_tuple(MBlock, KBlock), make_tuple(KBlock, 1));

// Apply XOR swizzle to avoid LDS bank conflicts
auto swizzled_lds_desc_a = transform_tensor_descriptor(
    lds_desc_a,
    make_tuple(
        make_pass_through_transform(MBlock),
        make_xor_transform(KBlock, SwizzleParam) // XOR transformation for swizzling
    ),
    make_tuple(Sequence<0>{}, Sequence<1>{}),
    make_tuple(Sequence<0>{}, Sequence<1>{})
);
```

CK's approach is more explicit and composable, allowing developers to build custom swizzling patterns by composing basic transformations (`pass_through`, `pad`, `slice`, `xor`, etc.).

## 4. Epilogue Mapping

The epilogue handles applying activation functions, scaling, and writing the final results back to global memory.

### CUTLASS: Epilogue Functor
CUTLASS uses an `Epilogue` class that takes an `OutputOp` (e.g., `LinearCombination` or `LinearCombinationRelu`).

```cpp
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput, ElementsPerAccess, ElementAccumulator, ElementCompute>;
```

### CK: Elementwise Operations
CK achieves this by passing C++ functors (`ElementwiseOperation`) directly to the `GridwiseGemm` or `DeviceGemm` launch.

```cpp
// 1. Define the functor
struct ReluAdd {
    template <typename T>
    __device__ void operator()(T& out, const T& in_c, const T& in_d) const {
        out = (in_c + in_d) > 0 ? (in_c + in_d) : 0; // ReLU(C + D)
    }
};

// 2. Pass to the Device GEMM instance
auto gemm = DeviceGemmInstance<...>{};
auto invoker = gemm.MakeInvoker();
auto argument = gemm.MakeArgument(..., ReluAdd{}); // Epilogue passed here
invoker.Run(argument);
```
CK defines `GridwiseEpilogue` underneath to handle the block-to-global write, and it applies the provided functor to each element in VGPRs before the global store.

## 5. Main Loop Mapping

### CUTLASS: MmaPipelined
CUTLASS implements software pipelining (e.g., double buffering) inside its `MmaPipelined` or `MmaMultistage` classes.

### CK: BlockwiseGemm and Software Pipelining
CK exposes pipeline control via `BlockwiseGemm` and `LoopScheduler`.

```cpp
using BlockwiseGemm = BlockwiseGemmXdlops_v2<
    BlockSize,
    FloatA, FloatB, FloatC,
    InMemoryDataOperationEnum::Set,
    ...>;

// CK handles the software pipeline internally based on the selected GEMM trait:
// e.g., GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 enables multi-stage prefetching
```

## CK Tile API (Modern CK)
AMD recently introduced the **CK Tile API**, which is even closer to CUTLASS 3.x's CuTe syntax. It provides a more intuitive, layout-algebra-based approach.

If migrating from CUTLASS 3.x (CuTe), consider using the CK Tile API instead of traditional CK.

```cpp
// CK Tile API example
auto tile_a = ck_tile::make_tile_window(
    tensor_a,
    ck_tile::make_tuple(MBlock, KBlock), // Shape
    ck_tile::make_tuple(m_idx, k_idx)    // Origin
);
ck_tile::load_tile(lds_tile_a, tile_a);
```

## Migration Checklist
1. **Identify Tile Sizes:** Map CUTLASS `<ThreadblockShape, WarpShape, InstructionShape>` to CK `BlockSize, MPerBlock, NPerBlock, KPerBlock, MPerXDL, NPerXDL`.
2. **Translate Layouts:** Convert `Layout::RowMajor` / `ColumnMajor` to CK `TensorDescriptor` and affine transforms.
3. **Convert Epilogue:** Rewrite CUTLASS `OutputOp` to a CK `ElementwiseOperation` functor.
4. **Update Host Code:** Replace `cutlass::gemm::device::Gemm` with `ck::tensor_operation::device::DeviceGemm`.
5. **Tune for 64-Thread Wavefronts:** Adjust block sizes and MFMA configurations to match AMD CDNA specifications.

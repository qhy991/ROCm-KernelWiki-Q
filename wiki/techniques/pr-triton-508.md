---
id: technique-pr-triton-508
title: "Triton Flash Attention: Head Padding for Non-Power-of-2 Head Sizes"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - triton-rocm
  - rocm-kernel
  - flash-attention
  - optimization
  - tiling
  - memory
kernel_types:
  - flash-attention
  - attention
languages:
  - triton-rocm
confidence: source-reported
sources:
  - pr-triton-508
---

# Triton Flash Attention: Head Padding for Non-Power-of-2 Head Sizes

## Summary

This optimization introduces conditional padding logic to support non-power-of-2 head dimensions (e.g., 33, 65, 90) in the ROCm Triton Flash Attention implementation. By padding the head size up to the next hardware-friendly power of 2 (32, 64, or 128) and enabling conditional boundary checks in Triton `tl.load` and `tl.store` operations, the kernel avoids out-of-bounds memory accesses while still utilizing optimal block sizes for MFMA matrix cores.

## Architectural & Performance Context

Flash Attention divides Query, Key, and Value tensors into blocks and relies on dense matrix multiplications. On AMD CDNA hardware, matrix operations (MFMA) prefer tiles with power-of-2 dimensions (typically 32, 64, 128, or 256) to ensure optimal vector register (VGPR) usage and efficient mapping to matrix cores. 

When user models specify non-standard head dimensions (e.g., $d = 90$), forcing the underlying matrix multiplication to operate on irregular shapes would result in inefficient loop bounds and poor utilization of the compute units. To resolve this:
- The host-side code identifies the next optimal power of 2 ($d_{padded} \ge d_{actual}$).
- The Triton kernel allocates internal static buffers (SRAM/LDS) mapped to $d_{padded}$ (via `BLOCK_DMODEL`).
- Reads from and writes to global memory (HBM) are performed with precise 2D boundary checking to avoid out-of-bounds accesses.

## Implementation Details

### Host-Side Padding Calculation

In `python/perf-kernels/flash-attention.py`, the driver logic calculates `padded_d_model` based on an explicit set of optimal `BLOCK_DMODEL` sizes (`{32, 64, 128}`):

```python
# Get closest power of 2 over or equal to 32.
unpadded_head_dims = {32, 64, 128}
if head_size not in unpadded_head_dims:
    padded_d_model = None
    for i in unpadded_head_dims:
        if i > head_size:
            padded_d_model = i
            break
    assert padded_d_model is not None
else:
    padded_d_model = head_size
```

Both sizes are passed into the Triton kernel arguments:
- `actual_block_dmodel = head_size`
- `BLOCK_DMODEL = padded_d_model`

### Kernel-Side Conditional Bounds Checking

Inside the Triton JIT-compiled kernel, `actual_block_dmodel` limits the shape defined by `tl.make_block_ptr`, meaning the hardware loads and stores exactly the requested tensor slice, while internally padding the remaining elements with zeros (using `padding_option="zero"`):

```python
Q_block_ptr = tl.make_block_ptr(
    base=Q + q_offset,
    shape=(seqlen_q, actual_block_dmodel), # Shape constrained to actual dimension
    strides=(stride_qm, stride_qk),
    offsets=(start_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_DMODEL),   # Internal tile size remains power of 2
    order=(1, 0)
)
```

To avoid boundary checking overhead where not needed, the kernel determines statically if it needs 2D boundary checks by checking if `actual_block_dmodel == BLOCK_DMODEL`. 

```python
# padded_head controls whether to add column boundary checks
if PADDED_BLOCK:
    if padded_head:
        k = tl.load(K_block_ptr, boundary_check=(1,0), padding_option="zero")
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
else:
    if padded_head:
        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    else:
        k = tl.load(K_block_ptr)
```

### Store Phase

When accumulating the result into `Out`, the kernel also applies a strict boundary check `(0, 1)` to make sure the zero padding introduced in the `BLOCK_DMODEL` registers doesn't spill over into the output buffer in global memory:

```python
# Don't exceed shape, makes sure padding isn't put in output.
tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,1))
```

## Key Takeaways

1. **Power-of-2 Tiling**: Standardizing `BLOCK_DMODEL` to 32, 64, or 128 provides predictable and highly optimized inner loop unrolling, reducing instruction overhead.
2. **2D Boundary Checking**: Using `boundary_check=(0,1)` allows `tl.make_block_ptr` to cleanly handle both the sequence length dimension (which is frequently non-divisible by `BLOCK_M`/`BLOCK_N`) and the head dimension.
3. **Zero Padding**: By adding `padding_option="zero"`, the padded regions don't corrupt accumulation sums or matrix multiplication products, as anything multiplied by the zero-padded region resolves to zero.

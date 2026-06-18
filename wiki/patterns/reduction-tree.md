---
id: pattern-reduction-tree
title: Reduction Tree
type: wiki-pattern
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [reduction, cross-lane, dpp, bpermute, lds, wave-reduction, optimization]
confidence: source-reported
techniques: [wave-reduction]
kernel_types: [reduction, softmax, layernorm, rmsnorm]
related: []
sources: []
---

# Reduction Tree Pattern

The **Reduction Tree** is a fundamental parallel programming pattern used to combine multiple values across a workgroup (thread block) into a single result (e.g., sum, max, min). On AMD CDNA architectures, optimizing this pattern requires mapping the logical tree to the physical execution hierarchy: reducing first within a wavefront using register shuffles, and then across wavefronts using Local Data Share (LDS).

## Architecture Mapping

On AMD GPUs, wavefronts consist of 64 threads (lanes). The most efficient reduction tree follows a two-tier approach:
1. **Wave-Level Reduction:** Reduces 64 values into a single value using register-to-register communication without accessing memory.
2. **Block-Level Reduction:** Reduces the partial results from each wavefront using LDS memory.

## 1. Wave-Level Reduction (Register Shuffles)

Modern AMD architectures provide two primary mechanisms for cross-lane communication within a wavefront:
* **DPP (Data Parallel Primitives):** Allows ALU instructions to source operands directly from neighboring lanes. DPP provides low-latency reductions but has restricted routing patterns (e.g., row shifts, quad permutations).
* **ds_bpermute:** Uses the LDS crossbar (without allocating LDS memory) to allow arbitrary all-to-all data routing within a wavefront.

In HIP C++, these hardware features are exposed via the `__shfl_down` and `__shfl_xor` intrinsics. 

### Wave Reduction Implementation

A typical wave-level reduction iteratively folds the wavefront in half:

```cpp
template <typename T>
__device__ __forceinline__ T wave_reduce_sum(T val) {
    // 64-lane wavefront reduction
    val += __shfl_down(val, 32);
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
    return val; // Lane 0 holds the total sum
}
```

Under the hood on CDNA architectures, the compiler translates these `__shfl_down` operations into `ds_bpermute_b32` instructions (or DPP shifts), enabling high-bandwidth, low-latency data movement.

## 2. Block-Level Reduction (LDS)

Once the wave-level reduction is complete, lane 0 of each wavefront holds the partial sum for that wave. To complete the block-level reduction, these partial sums must be communicated across wavefronts using LDS.

### Block Reduction Implementation

```cpp
template <typename T>
__device__ T block_reduce_sum(T val) {
    // Shared memory for partial sums (1 element per wave)
    // Max 1024 threads per block -> Max 16 waves per block
    __shared__ T shared_sums[16]; 

    int lane_id = threadIdx.x % 64;
    int wave_id = threadIdx.x / 64;

    // 1. Perform wave-level reduction
    val = wave_reduce_sum(val);

    // 2. Lane 0 of each wave writes its partial sum to LDS
    if (lane_id == 0) {
        shared_sums[wave_id] = val;
    }

    __syncthreads(); // Ensure all waves have written to LDS

    // 3. Final reduction by Wave 0
    // We only need the first wave to perform the final reduction
    // Assuming block size is <= 1024 (so max 16 waves)
    T final_val = (threadIdx.x < (blockDim.x + 63) / 64) ? shared_sums[lane_id] : 0;
    
    if (wave_id == 0) {
        final_val = wave_reduce_sum(final_val);
    }

    return final_val; // Lane 0 of Wave 0 holds the final block sum
}
```

## Performance Considerations for CDNA

* **LDS Bank Conflicts:** In the block-level reduction, writing partial sums to `shared_sums[wave_id]` may cause bank conflicts if multiple waves write simultaneously. However, since only lane 0 of each wave writes, these writes are staggered or broadcast efficiently via the LDS crossbar.
* **Vectorized Memory Access:** Before initiating the reduction tree, ensure that data is loaded from global memory using vectorized loads (`buffer_load_dwordx4`) to maximize memory bandwidth utilization. The threads can perform a thread-local sequential reduction on the vectorized elements before entering the wave-reduction phase.
* **Instruction Choice:** While `__shfl` intrinsics are portable, hand-optimized assembly using native `v_add_f32_dpp` can sometimes eliminate intermediate `v_mov` instructions by fusing the addition and the shuffle into a single cycle. However, compiler heuristics for `__shfl` are highly optimized in modern ROCm releases.

## Applications

The two-tier reduction tree is ubiquitous in kernel development. It is the core primitive in:
* **Softmax / LayerNorm / RMSNorm:** Requires computing the maximum and sum of exponentials (or sums of squares) across the feature dimension.
* **Flash Attention:** The online softmax computation relies heavily on maintaining running maximums and sums via block reductions.
* **GEMM Epilogues:** When accumulating results across split-K tiles.

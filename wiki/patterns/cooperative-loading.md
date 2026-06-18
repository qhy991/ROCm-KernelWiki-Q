---
id: pattern-cooperative-loading
title: Cooperative Loading
type: wiki-pattern
architectures: [cdna2, cdna3, cdna4]
tags: [memory, bandwidth, optimization, lds, vectorized-load]
confidence: source-reported
techniques: [vectorized-load, swizzling, async-copy]
kernel_types: [gemm, attention, flash-attention, conv]
related: []
sources: []
---

# Cooperative Loading (协作加载)

Cooperative Loading is a fundamental memory access pattern used in GPU kernel design to maximize global memory bandwidth utilization. In this pattern, threads within a workgroup (thread block) collaborate to fetch a large block (tile) of data from global memory (HBM) into the Local Data Share (LDS). Once the data is cached in LDS, threads can independently and efficiently fetch the specific subsets of data they need for computation (such as feeding matrix core `v_mfma` instructions).

This pattern transforms scattered, non-contiguous, or narrow memory accesses into wide, perfectly coalesced memory transactions, saturating the High Bandwidth Memory (HBM) limits.

## 1. Motivation

- **Global Memory Bandwidth**: Accessing HBM is latency-bound and highly bandwidth-sensitive. Utilizing 128-bit wide vectorized instructions (e.g., `buffer_load_dwordx4` or `global_load_dwordx4`) is critical for hitting the peak bandwidth on MI250X and MI300X.
- **Data Reuse**: In workloads like GEMM and Flash Attention, the same memory tile is used by multiple threads multiple times. Loading it once into LDS prevents redundant, expensive HBM reads.
- **Uncoalesced Accesses Mitigation**: Compute patterns often require reading memory in ways that cause uncoalesced accesses if done directly from global memory. By cooperatively loading data into LDS first, the data can be written using a layout (e.g., swizzling) that ensures completely conflict-free subsequent reads.

## 2. Implementation Details

In ROCm/HIP, a standard thread block might be 256 threads (4 wavefronts of 64 threads). If the block needs to load a `128 x 128` tile of `fp16` elements (32 KB total), cooperative loading divides this workload evenly across all 256 threads.

Each thread is responsible for `128 * 128 / 256 = 64` elements. By loading chunks of 8 `fp16` elements (using a 128-bit `float4` equivalent type), each thread issues exactly 8 vectorized memory loads.

### 2.1 HIP C++ Example

```cpp
template <typename T, int TILE_M, int TILE_N, int BLOCK_SIZE>
__global__ void cooperative_load_kernel(const T* __restrict__ global_in, T* __restrict__ global_out) {
    // Allocate LDS (Local Data Share)
    __shared__ T lds_tile[TILE_M][TILE_N];

    int tid = threadIdx.x;
    
    // Calculate elements per thread
    constexpr int TOTAL_ELEMENTS = TILE_M * TILE_N;
    constexpr int ELEMENTS_PER_THREAD = TOTAL_ELEMENTS / BLOCK_SIZE;
    
    // Force 128-bit vectorized load, e.g., float4 for 16-bit types
    using LoadType = float4; 
    constexpr int ELEMENTS_PER_LOAD = sizeof(LoadType) / sizeof(T);
    constexpr int LOADS_PER_THREAD = ELEMENTS_PER_THREAD / ELEMENTS_PER_LOAD;

    const LoadType* global_ptr = reinterpret_cast<const LoadType*>(global_in);
    LoadType* lds_ptr = reinterpret_cast<LoadType*>(&lds_tile[0][0]);

    // Phase 1: Cooperative Load from HBM to LDS
    #pragma unroll
    for (int i = 0; i < LOADS_PER_THREAD; ++i) {
        // Coalesced linear index mapped across the thread block
        int linear_idx = tid + i * BLOCK_SIZE;
        
        // HBM -> VGPR -> LDS
        lds_ptr[linear_idx] = global_ptr[linear_idx];
    }

    // Synchronize to ensure all data is in LDS
    __syncthreads();

    // Phase 2: Independent compute or LDS read
    // Now threads can independently read data from lds_tile
    // (e.g., using ds_read_b128 to feed v_mfma_f32_32x32x8f16)
}
```

### 2.2 Key Hardware Considerations on CDNA

1. **Instruction Choice**: To maximize L2 cache to VGPR bandwidth, the compiler should emit `global_load_dwordx4` or `buffer_load_dwordx4` instructions. Requesting 128 bits per thread is the maximum supported width, ensuring optimal transaction granularity.
2. **Register Pressure (VGPRs)**: Since cooperative loads traditionally land in VGPRs before being written to LDS via `ds_write_b128`, large tiles will consume significant register files. An MI300X Compute Unit (CU) has 4 SIMD32 units, each with 1024 VGPRs. Excessive loads without consumption will lead to immediate occupancy drops.
3. **LDS Bank Conflicts**: The LDS on CDNA architectures comprises 32 memory banks (each 4 bytes wide). Consecutive `ds_write_b128` instructions from a wavefront must be carefully patterned. Applying XOR-based address swizzling or padding during the cooperative LDS write is often strictly required to maintain full throughput and prevent serialization.
4. **Asynchronous Copies**: On CDNA3 (MI300X), developers leverage asynchronous pipeline copies using `llvm.amdgcn.make.buffer.rsrc` and async memory instructions (equivalent to CUDA's `cp.async`) to push data directly from HBM into LDS. This significantly reduces VGPR pressure and hides memory latency behind active MFMA compute cycles.

## 3. Performance Characteristics

For a typical `256 x 128` FP16 matrix block load on MI300X:

| Strategy | Vectorization Width | HBM Utilization | LDS Write Bank Conflicts | Throughput (MI300X) |
| :--- | :--- | :--- | :--- | :--- |
| Naive HBM direct access | None (16-bit) | Low (Uncoalesced) | N/A | ~800 GB/s |
| Per-thread strided load | Partial (32-bit) | Medium | N/A | ~2.5 TB/s |
| **Cooperative Load** | **128-bit (float4)** | **Maximum** | **High (if unpadded)** | **>4.8 TB/s** |
| **Cooperative + Swizzled** | **128-bit (float4)** | **Maximum** | **None** | **>5.1 TB/s** |

*Note: Peak memory bandwidth on MI300X is ~5.3 TB/s.*

## 4. Typical Use Cases
- **Matrix Multiplication (GEMM)**: Both `A` and `B` operands are cooperatively loaded into LDS to feed the `v_mfma` core pipeline. 
- **Flash Attention**: Query, Key, and Value blocks are cooperatively loaded, which is essential to avoid redundant reads of Keys and Values across the attention window.
- **Convolution (Implicit GEMM)**: Image tiles and filter weights are packed and loaded collaboratively to execute convolution ops as standard GEMMs efficiently.

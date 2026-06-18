---
id: pattern-grid-stride-loop
title: Grid-Stride Loop
type: wiki-pattern
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, programming-model, memory, memory-bound]
confidence: source-reported
techniques: [persistent-kernel, vectorized-load]
kernel_types: [activation, layernorm, rmsnorm]
related: []
sources: []
---

# Grid-Stride Loop Pattern

The Grid-Stride Loop is a fundamental kernel programming pattern in HIP/ROCm and CUDA that decouples the physical execution grid dimensions from the logical size of the dataset being processed. 

## Traditional vs. Grid-Stride Loop

### Monolithic Thread Mapping
Traditionally, a kernel assigns exactly one thread per data element. The launch configuration must be tailored precisely (or padded) to match the total number of elements $N$.

```cpp
__global__ void add_traditional(int n, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] + y[i];
    }
}
```
This fails if $N$ is too large (exceeding maximum grid dimensions) or forces suboptimal block counts that might not divide evenly into the dataset size without branching on the boundary.

### Grid-Stride Approach
A grid-stride loop iterates over the data using the total number of threads in the grid as the stride. This ensures every element is processed exactly once, regardless of how many threads are physically launched.

```cpp
__global__ void add_grid_stride(int n, const float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}
```

## Advantages

1. **Flexibility & Scalability:** The kernel can gracefully handle arrays of any size, even those far exceeding the maximum number of threads allowed by the hardware.
2. **Memory Coalescing:** Consecutive threads within a wavefront process consecutive elements during each loop iteration. This leads to perfectly coalesced global memory accesses, maximizing memory bandwidth efficiency.
3. **Hardware Utilization (Occupancy):** You can decouple the grid launch parameters from the data size. This allows you to launch exactly the right number of wavefronts to fill the target GPU (e.g., 304 CUs on MI300X $\times$ optimal wavefronts/CU) to maximize occupancy without launching an excessive number of blocks that incur scheduling overhead.
4. **Persistent Kernels:** This pattern is the foundation for persistent kernels. By launching just enough thread blocks to fill the GPU CUs, the threads can continuously fetch work from a queue or process large datasets without the overhead of retiring and dispatching new thread blocks.
5. **Debugging & Testing:** You can easily test a kernel's logic sequentially by launching it with `<<<1, 1>>>`. The grid-stride loop will execute sequentially across the entire dataset.

## Vectorized Grid-Stride Loops on CDNA
To maximize throughput on CDNA architectures (MI250X, MI300X), the grid-stride pattern is frequently combined with vectorized loads/stores (e.g., `float4`, `double2`). On MI300X, saturating the massive HBM bandwidth requires issuing wide memory requests.

```cpp
__global__ void add_grid_stride_vectorized(int n, const float4 *x, float4 *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 floats (128 bits) at a time
    int n_vec = n / 4;
    for (int i = index; i < n_vec; i += stride) {
        float4 x_val = x[i];
        float4 y_val = y[i];
        y_val.x += x_val.x;
        y_val.y += x_val.y;
        y_val.z += x_val.z;
        y_val.w += x_val.w;
        y[i] = y_val;
    }
    
    // Handle remaining elements (n % 4) using a scalar grid-stride loop ...
}
```
This pattern translates beautifully into 128-bit memory instructions (`global_load_dwordx4` / `global_store_dwordx4`), maximizing memory throughput.

## Performance Considerations on AMD CDNA

- **Wavefront Size**: ROCm/HIP uses a wavefront size of 64 (`warpSize == 64`) for CDNA architectures. The `blockDim.x` should always be a multiple of 64 to avoid idle threads within a wavefront.
- **L1/L2 Cache Efficiency**: The grid-stride loop naturally streams data sequentially, which plays nicely with the cache hierarchy and prefetchers on CDNA.
- **Loop Unrolling**: For compute-light operations, the instruction overhead of the loop itself (`i += stride`, `i < n`) can become a bottleneck. The compiler can automatically unroll the grid-stride loop, or developers can manually unroll it using `#pragma unroll` to hide instruction latency and increase instruction-level parallelism (ILP).

```cpp
__global__ void add_grid_stride_unrolled(int n, const float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}
```

By ensuring that the total number of threads (`blockDim.x * gridDim.x`) is large enough to hide memory latency (sufficient occupancy), the grid-stride pattern becomes the standard way to write high-performance element-wise and reduction kernels on AMD hardware.

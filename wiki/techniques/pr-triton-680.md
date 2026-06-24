---
id: wiki-technique-pr-triton-680
title: "Dynamic LDS Bounds Querying for GEMM and Stream-K Autotuning"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, triton-rocm, lds, optimization, scheduling, memory-bound, hardware]
confidence: inferred
---

# Dynamic LDS Bounds Querying for GEMM and Stream-K Autotuning

## 1. Architectural Intent and Context
Triton's autotuning framework systematically explores various configurations (tile sizes, warps, pipeline stages) to find the most performant kernel implementations for specific hardware. A critical constraint in this exploration is the Local Data Share (LDS) capacity. 

In AMD ROCm architecture, LDS (equivalent to Shared Memory in CUDA) is a scarce, high-bandwidth on-chip memory. Exceeding the maximum available LDS per workgroup (or per Compute Unit) leads to compilation failures or severely limits occupancy (the number of active wavefronts). 

The intent of PR 680 is to introduce hardware-awareness into the `tune_gemm` and `tune_streamk` scripts by explicitly querying the device's maximum LDS size. This enables the tuning infrastructure to dynamically filter out search space candidates that would violate the hardware's LDS constraints.

## 2. Optimization Technique: Search Space Pruning
By computing the expected LDS allocation for a given configuration before compilation, the tuning scripts can perform **Search Space Pruning**. The technique operates as follows:

- **Query Hardware Limit**: Interrogate the ROCm backend or device properties to determine the max `lds_size` per block for the target architecture (e.g., CDNA2, CDNA3, CDNA4).
- **Calculate Allocation**: For GEMM and Stream-K kernels, LDS usage is primarily dictated by the input tiles and software pipelining. The theoretical usage can be modeled as `(BLOCK_SIZE_M * BLOCK_SIZE_K + BLOCK_SIZE_K * BLOCK_SIZE_N) * bytes_per_element * num_stages`.
- **Early Rejection**: If the calculated LDS allocation exceeds the queried hardware limit, the configuration is immediately pruned.

This drastically reduces autotuning time and avoids runtime/compilation exceptions caused by out-of-memory errors on the device.

## 3. Memory Bounds and Scheduling Implications

### LDS (Local Data Share) Bounds
For heavy compute kernels like GEMM and Stream-K, LDS is the primary bottleneck for data reuse. 
- On modern CDNA hardware, the LDS limit dictates how large the matrix tiles (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`) can be, and how deep the software pipeline (`num_stages`) can go. 
- While larger tiles and more stages allow better latency hiding (overlapping global memory loads with MFMA compute), they directly increase the LDS footprint. Querying the exact bounds allows Triton to max out the pipeline stages without breaching the physical LDS limits, yielding optimal performance.

### Stream-K and Load Balancing
Stream-K is a dynamic load-balancing algorithm where the matrix multiplication operations are distributed across all available CUs, regardless of the output tile grid. It inherently relies on efficient state sharing and work distribution. Enforcing LDS bounds is even more critical here to maintain uniform execution across Compute Units. If Stream-K blocks were allowed to exceed LDS bounds, it would result in uneven occupancy or outright scheduling failures, negating the benefits of the Stream-K algorithm.

## 4. Conclusion
Integrating an LDS size query into Triton's tuning scripts (`tune_gemm` and `tune_streamk`) is a fundamental hardware-aware optimization. It safely bounds the autotuning search space, ensuring that the Triton compiler only evaluates mathematically valid and hardware-compatible configurations. This results in more robust autotuning phases and guaranteed deployment compatibility across CDNA architectures.

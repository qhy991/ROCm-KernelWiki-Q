---
id: technique-triton-gemm-tuning
title: "Advanced GEMM Tuning in Triton: Rotating Tensors, ICache Flushes, and Bias"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, memory, bandwidth, memory-bound, rocm]
confidence: inferred
techniques: []
hardware_features: []
kernel_types: [gemm]
languages: [triton-rocm]
sources: [pr-triton-588]
reproducibility: concept
---

# Advanced GEMM Tuning in Triton: Rotating Tensors, ICache Flushes, and Bias

## Context
Accurately benchmarking and tuning General Matrix Multiply (GEMM) kernels in Triton is challenging. A naive tuning script that invokes the same kernel repeatedly over the same input tensors often falls victim to artificial caching effects (both for data in L2/L3 caches and instructions in the ICache). These caching effects inflate performance metrics, producing artificially high TFLOPS numbers that fail to materialize in production workloads.

ROCm/triton PR [#588](https://github.com/ROCm/triton/pull/588) addresses these profiling inaccuracies by introducing three critical features to the `tune_gemm.py` script: **rotating tensors**, **icache flushing**, and **bias** support.

## Architectural & Intent Analysis

### 1. Rotating Tensors (Data Cache Eviction)
**Intent:** Prevent artificial data cache (L2/L3) hits during iterative benchmarking.
**Mechanism:** Instead of reusing the identical `A` and `B` tensors for every iteration of the kernel, the tuning script allocates a larger pool of memory and cycles through independent tensors on each run.
**Impact:** 
- **Forces HBM Traffic:** Guarantees that the GPU must fetch input matrices directly from High Bandwidth Memory (HBM) instead of hitting the L2 cache, accurately modeling real-world streaming workloads.
- **True Memory-Bound Profiling:** Exposes the true memory bandwidth limitations of the kernel. This is especially crucial for CDNA3 (MI300X), which features massive amounts of cache. Without rotating tensors, a memory-bound tall-and-skinny GEMM can incorrectly appear compute-bound.

### 2. Instruction Cache (ICache) Flushing
**Intent:** Prevent instruction cache staleness and measure cold-start instruction fetching penalties.
**Mechanism:** The ICache is explicitly flushed between kernel executions. In ROCm, this can be handled via built-in ISA cache control instructions or by executing a dummy kernel designed to evict the instruction cache footprint.
**Impact:** 
- **Cold-Start Consistency:** Ensures each kernel launch starts with a cold instruction cache. 
- **Complex Kernels:** Modern fused kernels (e.g., Flash Attention, heavy Epilogues) and heavily unrolled GEMMs have large instruction footprints that may exceed ICache capacity. Flushing prevents these kernels from artificially benefiting from staying resident in the ICache during a tight benchmark loop.

### 3. Bias Support (Epilogue Fusion)
**Intent:** Enable accurate autotuning for end-to-end neural network layers (e.g., Linear + Bias).
**Mechanism:** Incorporates an optional bias vector addition into the matrix multiplication (`D = A @ B + C`).
**Impact:**
- **Register Pressure:** Fused epilogues increase Vector General Purpose Register (VGPR) pressure and alter occupancy limits. Tuning directly with bias ensures that the autotuner evaluates the exact register allocation profile of the fused kernel.
- **Holistic Tuning:** Prevents situations where an optimal block size for a pure GEMM falls off an occupancy cliff when bias operations are fused into the epilogue.

## Performance and Memory Bounds
For operations heavily reliant on HBM bandwidth, introducing **rotating tensors** provides a much harsher, more realistic environment to test memory-latency hiding techniques. A well-optimized kernel should successfully hide memory fetches behind Matrix Fused Multiply-Add (`MFMA`) instructions through:
- `double-buffering` (or multi-stage pipelining).
- `async-copy` global-to-LDS transfers.

If the performance difference between single-tensor loops and rotating-tensor loops is high, it indicates that the kernel is largely **memory-bound** and relies too heavily on L2 caching, failing to pipeline HBM loads properly. Conversely, for large square GEMMs that are heavily **compute-bound**, the performance degradation from rotating tensors will be marginal, verifying that MFMA execution dominates the timeline.

## Best Practices for Triton Tuning on AMD CDNA
1. **Always Rotate Tensors:** When searching for the best tile sizes (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`), ensure `tune_gemm.py` uses rotating tensors. The optimal tile size for cached data is often sub-optimal for HBM-streamed data.
2. **Tune with Final Fusions:** If your production kernel requires bias, activation, or scaling, include those in the tuning phase. Fusions change the VGPR footprint, significantly altering the occupancy landscape.
3. **Account for ICache:** For large, aggressively unrolled kernels, ensure ICache flushes are enabled during benchmarking to avoid selecting a configuration that thrashes the ICache in production.

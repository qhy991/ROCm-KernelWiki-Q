---
id: technique-wavefront-scheduling
title: Multi-Wavefront Scheduling Strategies
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [scheduling, pipeline, occupancy, wavefront]
confidence: source-reported
techniques: [mfma-scheduling, occupancy-tuning, double-buffering]
hardware_features: [wavefront, compute-unit]
kernel_types: [gemm, flash-attention]
related: [technique-double-buffering, technique-mfma-scheduling]
sources: []
reproducibility: snippet
---

# Multi-Wavefront Scheduling Strategies (多 Wavefront 调度策略)

In AMD CDNA architectures, achieving peak performance requires a deep understanding of how the hardware schedules **wavefronts** (AMD's equivalent of CUDA warps) to SIMD execution pipelines. The hardware relies on high occupancy and zero-overhead context switching between wavefronts to hide memory latency and keep the Matrix Core (MFMA) pipelines saturated. 

This guide details how hardware schedulers dispatch wavefronts, the performance impact of `s_barrier` placement, the handling of memory latencies, and techniques to maintain maximum pipeline utilization.

## 1. Hardware Scheduler and Wavefront Dispatch

Each AMD Compute Unit (CU) contains multiple execution units, including Vector ALUs (VALU), Scalar ALUs (SALU), Memory pipelines (LSU for VMEM and LDS), and Matrix Cores (MFMA). In CDNA2 and CDNA3, a CU typically has 4 SIMD16 units. A single wavefront consists of 64 threads. 

### Zero-Overhead Context Switching
When a wavefront encounters a stall—most commonly due to an unfulfilled memory request handled by an `s_waitcnt` instruction—the hardware scheduler instantly context-switches to another eligible wavefront residing on the same SIMD unit.
- The state of each wavefront (VGPRs, SGPRs, Program Counter) is physically partitioned in the register files, meaning no data needs to be saved or restored.
- The scheduler checks the instruction buffer of active wavefronts and issues instructions to the available execution pipelines (e.g., dispatching an MFMA instruction from Wave 0, while dispatching a `ds_read_b128` from Wave 1).

To effectively utilize this capability, kernels require **sufficient occupancy**: enough active wavefronts per CU to always have eligible instructions when others stall.

## 2. Managing Memory Latencies

Memory operations exhibit significant latencies that must be hidden by independent compute instructions from the same wavefront, or by switching to other wavefronts. Approximate latencies on CDNA2/CDNA3:
- **LDS (Shared Memory)**: ~30-40 cycles
- **L1 Cache Hit**: ~30-50 cycles
- **L2 Cache Hit**: ~100-150 cycles
- **HBM (Global Memory)**: 300+ cycles

### Interleaving Vector Memory and LDS
To hide these latencies, kernels should group memory loads (`buffer_load_dwordx4` or `global_load_dwordx4`) and interleave them with long-running compute instructions (like `v_mfma_f32_32x32x8f16`). Since MFMA instructions occupy the matrix pipeline for multiple cycles (e.g., 32 cycles on CDNA2, 16 on CDNA3), the LSU is free to process `ds_read` or `buffer_load` operations concurrently without stealing throughput from the math units.

## 3. Impact of Barrier Placement (`s_barrier`)

The `s_barrier` instruction synchronizes all wavefronts within a workgroup. It is typically required when data loaded into LDS by one wavefront must be consumed by another. 

**The Performance Cliff:**
When a wavefront hits an `s_barrier`, it halts execution until all other wavefronts in the workgroup reach the same barrier. If barrier placement is suboptimal:
1. **Wavefront Draining**: Early-arriving wavefronts sit idle. The CU's active wavefront pool shrinks, removing the scheduler's ability to hide latencies.
2. **Execution Pipeline Starvation**: With fewer eligible wavefronts, VALU, MFMA, and memory pipelines sit idle, drastically dropping IPC (Instructions Per Clock).

### Optimal Barrier Strategies
- **Defer Barriers**: Push `s_barrier` as late as possible. After loading data into LDS, have the wavefront perform other independent work (e.g., address calculation for the next tile, or incrementing pointers) before hitting the barrier.
- **Software Pipelining**: In a double-buffered or multi-staged pipeline, `s_barrier` is placed after global-to-LDS loads are issued, but ideally *after* some independent compute of the current stage is dispatched.

```cpp
// Anti-Pattern: Early Barrier
for(int i=0; i<iters; ++i) {
    // 1. Load from Global to LDS
    lds[idx] = global_data[idx];
    
    // 2. Immediate Barrier (Causes Pipeline Starvation)
    __syncthreads(); // Compiles to s_barrier
    
    // 3. Compute
    compute(lds);
    __syncthreads();
}

// Optimized Pattern: Deferred Barrier with Double Buffering
// 1. Prologue Load
lds[0][idx] = global_data[idx];
__syncthreads();

for(int i=0; i<iters-1; ++i) {
    // 2. Issue next Load asynchronously
    lds[(i+1)%2][idx] = global_data[next_idx];
    
    // 3. Independent Work: Compute on current buffer
    compute(lds[i%2]);
    
    // 4. Deferred Barrier: Wait for the next load to complete
    __syncthreads();
}
// Epilogue compute
compute(lds[(iters-1)%2]);
```

## 4. Techniques for High Pipeline Utilization

To maximize hardware scheduler efficiency on CDNA GPUs:

### A. Instruction-Level Parallelism (ILP) and Wait states
Use `s_waitcnt` effectively. Do not wait for all memory operations to complete if you only need a subset. 
- `vmcnt(N)`: Wait until at most N vector memory operations are pending.
- `lgkmcnt(N)`: Wait until at most N LDS/GDS/Constant operations are pending.

By fine-tuning `s_waitcnt`, you allow the wavefront to proceed and issue MFMA instructions as soon as the specific `ds_read` data is ready, while other global loads may still be in flight.

### B. Register Pressures vs. Occupancy Balancing
Wavefront scheduling relies on having enough active wavefronts. However, utilizing many VGPRs (to hold large matrix tiles) reduces occupancy. 
- **CDNA2 (MI250X)**: Max 512 VGPRs per thread. Using 256 VGPRs allows 2 wavefronts per SIMD (8 per CU). 
- **CDNA3 (MI300X)**: Increased register capacity allows larger allocations, but the trade-off remains.

Often, allocating *more* VGPRs to increase ILP per wavefront (e.g., larger block sizes and register tiling) yields better performance than maximizing occupancy, provided that the memory latency can be hidden by the larger math instructions.

### C. Leveraging Composable Kernel (CK) tile APIs
Instead of manually tuning barriers and wavefront scheduling, using abstractions like the CK tile API enforces optimal software pipelining. The framework automatically unrolls loops, interleaves `v_mfma` and `ds_read`/`buffer_load`, and optimally places `s_waitcnt` and `s_barrier`.

## Performance Data on MI300X

Optimizing wavefront scheduling, particularly through double-buffering and deferred barriers, shows massive speedups in memory-bound and compute-bound kernels:

| Kernel Variant | Scheduling Strategy | Pipeline Utilization | Peak TFLOPS (FP16) |
| :--- | :--- | :--- | :--- |
| Naive GEMM | Synchronous loads, early barriers | ~45% | ~180 TFLOPS |
| Tuned GEMM | Double-buffered, deferred barriers | ~75% | ~290 TFLOPS |
| CK / Triton | Asynchronous pipeline, fine `s_waitcnt`| ~90%+ | ~340+ TFLOPS |

> [!TIP]
> **Profiler Validation**: Use `rocprof` to measure `VALU_UTILIZATION` and `MFMA_UTILIZATION`. If these are low but memory bandwidth is also not saturated, it strongly indicates wavefront stalling—likely due to poor `s_barrier` placement or missing software pipelining.

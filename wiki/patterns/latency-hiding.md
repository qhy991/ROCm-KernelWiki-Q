---
id: pattern-latency-hiding
title: Latency Hiding (延迟隐藏)
type: wiki-pattern
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, memory, occupancy, scheduling]
confidence: source-reported
techniques: [double-buffering, occupancy-tuning, mfma-scheduling]
kernel_types: [gemm, flash-attention, conv]
related: [hw-wavefront, hw-compute-unit]
sources: []
---

# Latency Hiding Pattern (延迟隐藏模式)

Latency hiding is a fundamental programming and scheduling pattern in GPU compute, essential for maximizing throughput on AMD ROCm/CDNA architectures. Memory access operations (such as global memory loads from HBM or local data share reads) have significant latency—often ranging from tens to hundreds of clock cycles. The latency hiding pattern involves keeping the execution pipelines (especially the Matrix Cores / MFMA units) busy with independent compute instructions while waiting for memory requests to complete.

On CDNA architectures (MI250X, MI300X), latency is primarily hidden through a combination of **Instruction-Level Parallelism (ILP)**, **Thread-Level Parallelism (TLP) / Occupancy**, and explicit **Software Pipelining (Double Buffering)**.

## Mechanisms for Latency Hiding

### 1. Thread-Level Parallelism (TLP) and Occupancy

When a wavefront executes a memory load, it eventually reaches an instruction that consumes the loaded data. At this point, the hardware scheduler encounters a stall (enforced via `s_waitcnt` instructions). To hide this stall, the hardware can perform a zero-overhead context switch to another active wavefront on the same Compute Unit (CU) that has independent instructions ready to execute.

*   **Occupancy**: The ability to hide latency via TLP is directly proportional to occupancy—the number of active wavefronts resident on a CU.
*   **VGPR Constraints**: Complex kernels (e.g., FlashAttention, large GEMMs) often consume a high number of Vector General-Purpose Registers (VGPRs). High VGPR allocation per wavefront reduces the maximum number of concurrent wavefronts (lowering occupancy). When occupancy is low, TLP alone cannot hide all memory latencies, forcing developers to rely more on ILP.

### 2. Instruction-Level Parallelism (ILP)

If TLP is insufficient due to register pressure, latency must be hidden within a single wavefront by issuing independent instructions. The CDNA ISA executes different types of instructions on distinct hardware pipelines:
*   **VMEM**: Vector Memory operations (global loads/stores).
*   **LDS/LGKM**: Local Data Share operations (`ds_read`, `ds_write`).
*   **VALU**: Vector ALU operations (standard arithmetic).
*   **MFMA**: Matrix Fused Multiply-Add operations.

Because these pipelines operate asynchronously and independently, a wavefront can issue a VMEM instruction, immediately followed by several independent MFMA or VALU instructions, before eventually waiting on the memory operation.

### 3. Software Pipelining and Double Buffering

To maximize ILP, kernels typically implement software pipelining (e.g., Double Buffering). Instead of executing "Load -> Compute -> Load -> Compute", the kernel overlaps the memory load of the *next* iteration with the compute of the *current* iteration.

## Implementation & Code Example (HIP C++ / ISA concept)

Proper latency hiding relies heavily on the correct interleaving of instructions and strategic placement of memory wait states (`s_waitcnt`).

### Conceptual Pipelining

```cpp
// Pseudocode for Double Buffered Latency Hiding

// Prologue: Load iteration 0
global_load_dwordx4(vA_next, ...);
global_load_dwordx4(vB_next, ...);

// Wait for iteration 0 loads to complete
__builtin_amdgcn_s_waitcnt(0); 

// Main Loop
for (int k = 0; k < K_BLOCKS - 1; k++) {
    // Swap registers (next becomes current)
    swap(vA_curr, vA_next);
    swap(vB_curr, vB_next);

    // Issue VMEM loads for iteration K+1 early
    // These execute asynchronously on the VMEM pipeline
    global_load_dwordx4(vA_next, ...);
    global_load_dwordx4(vB_next, ...);

    // Issue MFMA compute for iteration K using vA_curr / vB_curr
    // MFMAs take multiple cycles and execute on the Matrix pipeline
    for (int i = 0; i < M_TILES; i++) {
        for (int j = 0; j < N_TILES; j++) {
            v_c[i][j] = __builtin_amdgcn_mfma_f32_32x32x8f16(vA_curr[i], vB_curr[j], v_c[i][j], 0, 0, 0);
        }
    }

    // Wait for the iteration K+1 loads to finish before the next loop
    // vmcnt(0) means wait until 0 VMEM operations are outstanding
    __builtin_amdgcn_s_waitcnt(0); 
}

// Epilogue: Compute final iteration
swap(vA_curr, vA_next);
swap(vB_curr, vB_next);
for (int i = 0; i < M_TILES; i++) {
    for (int j = 0; j < N_TILES; j++) {
        v_c[i][j] = __builtin_amdgcn_mfma_f32_32x32x8f16(vA_curr[i], vB_curr[j], v_c[i][j], 0, 0, 0);
    }
}
```

### Advanced Waitcnt Tuning

In optimized AMD assembly or compiler-generated code, you will rarely see a generic `s_waitcnt 0` (which blocks until *all* memory operations are finished). Instead, specific counters are used:

*   **`vmcnt(N)`**: Wait until at most $N$ vector memory operations are outstanding.
*   **`lgkmcnt(M)`**: Wait until at most $M$ LDS/GDS/constant memory operations are outstanding.

By precisely tuning $N$ and $M$, the scheduler allows memory loads to remain in-flight as long as possible without stalling the compute pipelines. For instance, you can wait for `lgkmcnt(0)` (LDS reads finished) to feed an MFMA instruction, while leaving `vmcnt(1)` (Global memory reads for the next tile still pending).

## Performance Considerations on CDNA3 (MI300X)

*   **MFMA Latency**: On MI300X, a single `v_mfma_f32_32x32x8f16` operation is pipelined over multiple cycles. Issuing back-to-back independent MFMAs effectively hides the arithmetic latency of the matrix core itself.
*   **VGPR Limits vs Latency**: MI300X allows a very high maximum of VGPRs per SIMD (up to 512 registers in extreme configurations, though typical heavy kernels use 256). Allocating massive register files for multi-stage software pipelining (e.g., using 3 or 4 stages instead of 2) might reduce occupancy so severely that the ILP gained does not outweigh the TLP lost. Balancing the number of pipeline stages against VGPR usage is the core challenge of occupancy-tuning.

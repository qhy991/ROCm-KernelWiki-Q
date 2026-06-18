---
id: technique-sgpr-scalar-unit
title: SGPR and Scalar Unit Optimization
type: wiki-technique
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [optimization, vgpr, isa, compute, memory]
confidence: source-reported
techniques: [occupancy-tuning]
hardware_features: [wavefront, compute-unit]
kernel_types: [gemm, attention, flash-attention, reduction]
related: [technique-register-tiling]
sources: []
reproducibility: snippet
---

# SGPR and Scalar Unit Optimization

In AMD GPUs (CDNA and RDNA architectures), the execution engine is split into a **Scalar** and a **Vector** path. Optimizing the usage of Scalar General-Purpose Registers (SGPRs) and the Scalar ALU (SALU) is a critical technique to reduce VGPR pressure and free up the Vector ALU (VALU) for heavy computations, maximizing overall kernel performance.

## The Role of Scalar ALU and SGPRs

Each Compute Unit (CU) has independent scalar and vector resources:
- **Vector ALU (VALU) & VGPRs**: Handle divergent or thread-specific data. A single VGPR stores 64 distinct 32-bit values (one per thread in a wavefront).
- **Scalar ALU (SALU) & SGPRs**: Handle uniform data shared across the entire wavefront. An SGPR stores a single 32-bit value shared by all 64 threads. 

By offloading uniform calculations to the SALU, you can achieve two major performance benefits:
1. **Save VGPRs**: Replacing VGPRs with SGPRs lowers VGPR pressure. Lower VGPR usage can directly increase wavefront occupancy, allowing the GPU to hide memory latency more effectively.
2. **Save VALU Cycles**: SALU instructions execute asynchronously and in parallel with VALU instructions. Moving work to the SALU frees the VALU for core math operations (like `v_fma` or `v_mfma` Matrix Core instructions).

## Common Offloading Patterns

### 1. Loop Counters and Control Flow
Loop counters are inherently uniform across a wavefront unless there's divergent control flow. Allowing the compiler to map these to the SALU prevents wasting valuable vector resources.

**Inefficient (VGPR-bound):**
```cpp
// If index is mistakenly calculated in a thread-divergent way
int k_idx = threadIdx.x * 0; // Forces index to be a VGPR
for (int i = k_idx; i < K; ++i) {
    // Math operations
}
```

**Optimized (SGPR-bound):**
Using uniform variables for loop counters ensures they map to scalar add and compare instructions.
```cpp
// Handled by Scalar ALU
for (int i = 0; i < K; ++i) {
    // Math operations
}
```
At the ISA level, this optimal loop control complies to pure scalar instructions:
```asm
s_add_i32 s0, s0, 1     ; SALU: increment counter
s_cmp_lt_i32 s0, s1     ; SALU: compare
s_cbranch_scc1 loop_top ; SALU: branch
```

### 2. Memory Base Pointers
When loading tiles from global memory, the base pointer of the tile is often uniform across the wavefront, while only the element offset is thread-specific. 

Instead of computing the full 64-bit address in VGPRs for every thread (which consumes 2 VGPRs per thread), compute the 64-bit base address in SGPRs and use a scalar load (`s_load_dwordx4`) or an offset-based vector load (like `buffer_load` or `global_load` with an SGPR base and VGPR offset).

**Inefficient (Full VGPR Address):**
```cpp
// Computes full 64-bit address per thread
float* my_ptr = base_ptr + blockIdx.x * BLOCK_STRIDE + threadIdx.x * THREAD_STRIDE;
float val = *my_ptr;
```

**Optimized (SGPR Base + VGPR Offset):**
```cpp
// Base calculated uniformly (Stays in SGPR)
float* tile_base = base_ptr + blockIdx.x * BLOCK_STRIDE; 
// Thread offset (Stays in VGPR)
int thread_offset = threadIdx.x * THREAD_STRIDE;        
// Compiles to buffer_load using SGPR base and VGPR offset
float val = tile_base[thread_offset];
```
In ISA, this generates highly efficient buffer instructions:
```asm
; s[2:5] is the 128-bit buffer descriptor (SGPR)
; v0 is the thread offset (VGPR)
buffer_load_dword v1, v0, s[2:5], 0 offen
```

### 3. Uniform Constants and Scaling Factors
In operations like Softmax, RMSNorm, or scaled GEMM, scaling factors (like $1 / \sqrt{d}$) are often identical for the entire wavefront. 

Load these factors into SGPRs using `s_load_dword` and use the vector multiply instruction `v_mul` which is capable of taking one SGPR operand directly without needing to broadcast it to a VGPR:
```asm
s_load_dword s2, s[0:1], 0x0      ; SALU: Load scalar alpha from memory
v_mul_f32 v1, s2, v0              ; VALU: Multiply SGPR x VGPR -> VGPR
```
This avoids wasting 64 registers to store redundant copies of the same constant.

## Compiler Intrinsics and Hints

When writing HIP C++ or Triton code, the LLVM compiler generally attempts to extract scalar operations automatically. However, in complex kernels with excessive register pressure, you can forcefully guide it:

- **`__builtin_amdgcn_readfirstlane`**: Explicitly broadcasts a value from lane 0 of a VGPR to an SGPR if you as the programmer can guarantee the value is uniform across the active wavefront.
```cpp
// Force extraction to SGPR if compiler fails to prove uniformity
float uniform_val = __builtin_amdgcn_readfirstlane(val);
```

## Performance Impact on CDNA Architecture

| Architecture | SGPRs per Wave | VGPRs per Wave | SALU Pipeline | Target Workloads |
|--------------|----------------|----------------|---------------|------------------|
| MI250X (CDNA2)| 800 available  | 512 max        | 1 per SIMD    | Dense/Sparse GEMM |
| MI300X (CDNA3)| 800 available  | 512 max        | 1 per SIMD    | Flash Attention, MoE |

Modern CDNA architectures feature massive Matrix Core (MFMA) throughput. The VALU can easily become a bottleneck if clogged with address generation and loop logic rather than data transformations. 

Effective SGPR utilization can improve instructions-per-clock (IPC) by 10-20% in memory-addressing heavy kernels (like Flash Attention or custom fused operations) and permit higher occupancy by keeping VGPR usage under critical allocation thresholds (e.g., dropping from 130 VGPRs to <128 VGPRs to achieve 4 waves/SIMD).

## Key Takeaways
- **Keep it uniform**: Variables that are the same for all 64 threads should be handled as scalars.
- **Base + Offset**: Structure memory accesses using uniform base pointers (SGPR) and divergent offsets (VGPR).
- **Check the ISA**: Use tools like `rocobj` or `llvm-objdump` to inspect assembly and verify that loops and base addresses use `s_*` instructions instead of `v_*`.

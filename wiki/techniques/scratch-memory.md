---
id: technique-scratch-memory
title: Scratch Memory Spill Management
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [memory, scratch, register-spilling, performance]
confidence: source-reported
techniques: []
hardware_features: [vgpr, scratch-memory]
kernel_types: [compute-bound, memory-bound]
related: []
sources: []
reproducibility: snippet
---

# Scratch Memory Spill Management

In AMD ROCm/CDNA architectures, **Scratch Memory** is a private, per-thread memory space backed by the global memory system. While the compiler uses it automatically to handle register spills and certain local data structures, relying on scratch memory can severely degrade kernel performance due to its high latency and bandwidth consumption.

## Causes of Scratch Memory Usage

Scratch memory is primarily allocated by the LLVM AMDGPU compiler backend under the following conditions:

### 1. Register Spilling
When a kernel requires more Vector General-Purpose Registers (VGPRs) or Scalar General-Purpose Registers (SGPRs) than are available or allowed per thread, the compiler "spills" the excess variables into scratch memory.
* **Hardware Limits**: CDNA2 (MI250X) and CDNA3 (MI300X) support up to 512 VGPRs (with AccVGPRs) and 128 SGPRs per thread, but maximizing occupancy often requires limiting register usage (e.g., 64 VGPRs for 8 waves/SIMD).
* **Launch Bounds**: If `__launch_bounds__(max_threads_per_block, min_blocks_per_cu)` forces a register limit to guarantee a certain occupancy, the compiler will spill variables rather than fail compilation.

### 2. Dynamic Array Indexing
If a kernel allocates a local array (e.g., `float local_arr[16];`) and accesses it using a dynamic (variable) index that cannot be resolved at compile time, the compiler cannot map the array elements to discrete VGPRs. Instead, it places the entire array in scratch memory.

### 3. Large Local Structures
Passing large structures by value or instantiating large thread-local objects that exceed register capacity will force the compiler to allocate them in scratch memory.

## Performance Impact

Scratch memory operations map to `scratch_load_*` and `scratch_store_*` instructions (or `buffer_load_*`/`buffer_store_*` using the scratch resource descriptor) in the AMD ISA.

* **Latency**: Although scratch memory is backed by the L1 vector cache (vL1D) and L2 cache, cache misses result in full global memory latency (~300-500 cycles).
* **Bandwidth Contention**: Scratch traffic competes with standard global memory accesses, potentially starving the computation and bottlenecking memory-bound kernels.
* **Instruction Overhead**: Spilling introduces extra address calculation and memory instructions, increasing the overall instruction count and lowering the ALU pipeline utilization.

## Identifying Scratch Memory Usage

### 1. Compiler Flags
The easiest way to detect scratch usage at compile time is by inspecting the compiler's resource usage report.

Add `-Rpass-analyze=kernel-resource-usage` to your `hipcc` command:
```bash
hipcc -O3 --offload-arch=gfx90a -Rpass-analyze=kernel-resource-usage kernel.cpp
```
**Output Example:**
```text
remark: kernel 'my_kernel' uses 128 VGPRs, 32 SGPRs, 4096 bytes LDS, 256 bytes ScratchMemory
```
Any non-zero value for `ScratchMemory` indicates spilling or local array allocation.

### 2. ISA Inspection
By compiling with `--save-temps` or dumping the assembly, you can look for the kernel descriptor metadata at the end of the `.s` file:
```yaml
.amdhsa_private_segment_fixed_size 256 # Scratch size per thread in bytes
```
You can also directly search for ISA instructions like `scratch_store_dword` or `scratch_load_dwordx4`.

### 3. Profiling with Omniperf / rocprof
Using profiling tools, you can check if scratch usage is a runtime bottleneck:
* `TCP_TCP_TA_DATA_REQ_SCRATCH`: Number of scratch data requests to the L1 cache.
* `TCP_TCP_TA_DATA_REQ_SPILL`: Number of spill/restore requests.
In Omniperf, the **Memory Hierarchy** tab highlights high L1/L2 bandwidth consumption originating from scratch memory.

## Strategies to Eliminate Scratch Memory

### 1. Loop Unrolling for Constant Indexing
If a local array is placed in scratch due to dynamic indexing within a loop, unrolling the loop can make the indices constant. The compiler can then map the array elements directly to VGPRs (scalarization).

**Before (Causes Scratch Usage):**
```cpp
float local_vals[4];
// i is dynamic, forces local_vals into scratch
for (int i = 0; i < 4; i++) {
    local_vals[i] = compute(i);
}
```

**After (VGPR Allocation):**
```cpp
float local_vals[4];
#pragma unroll
for (int i = 0; i < 4; i++) {
    local_vals[i] = compute(i); // Loop unrolled, indices are constant
}
```

### 2. Managing Register Pressure
If scratch usage is caused by register spilling, you must reduce the number of live variables.
* **Scope Variables Tightly**: Declare variables only in the inner-most scope where they are needed.
* **Recompute vs. Store**: If calculating a value requires fewer instructions than loading a spilled value from memory, recompute it instead of keeping it in a register.
* **Vector Types**: Use types like `float4` or `int4` (e.g., loading via `global_load_dwordx4`) to maximize memory transaction efficiency and minimize the number of separate register addresses the compiler needs to track.

### 3. Adjusting `__launch_bounds__`
If your kernel uses `__launch_bounds__` to force high occupancy (e.g., forcing 8 waves/SIMD by restricting registers to 64), the compiler might be forced to spill extensively.
* Evaluate if higher register usage (and lower occupancy) actually yields better performance. Removing or relaxing `__launch_bounds__` might allow the compiler to use more VGPRs (up to 256 normal VGPRs on MI250X) and eliminate spilling entirely. 
* A compute-bound kernel with no spilling at 4 waves/SIMD often outperforms a kernel with heavy spilling at 8 waves/SIMD.

### 4. Moving Data to LDS (Local Data Share)
If a thread requires a large local array that cannot be scalarized, consider allocating it in LDS instead of thread-private memory.
* LDS is much faster, has lower latency than scratch memory, and does not pollute the L1/L2 data caches.
* Note that LDS is shared per thread block. You must index it using `threadIdx.x` and ensure sufficient LDS size (up to 64KB per CU) to accommodate the block.

```cpp
// Move from scratch to LDS
__shared__ float lds_vals[BLOCK_SIZE][16];
lds_vals[threadIdx.x][dynamic_index] = val;
```

## Hardware Specifics: CDNA Architecture

On MI250X (gfx90a) and MI300X (gfx942), the unified register file is massive (up to 512 VGPRs per thread when combining normal VGPRs and AccVGPRs). Therefore, spilling usually implies either extreme loop unrolling, massive matrix tile sizes (e.g., large MFMA `m` and `n` dimensions), or suboptimal compiler heuristics. 

If using Matrix Core operations like `v_mfma_f32_32x32x8f16`, the resulting accumulator tiles reside in ArchVGPRs or AccVGPRs. Managing the life cycle of these tiles is critical; holding too many tiles simultaneously will immediately trigger spilling to scratch.

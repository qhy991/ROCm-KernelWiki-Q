---
id: technique-pr-triton-634
title: "Persistent Softmax Optimization in Triton"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
hardware_features:
  - compute-unit
techniques:
  - persistent-kernel
  - vectorized-load
  - occupancy-tuning
kernel_types:
  - softmax
languages:
  - triton-rocm
tags:
  - rocm-kernel
  - optimization
  - memory-bound
  - bandwidth
confidence: inferred
sources:
  - pr-triton-634
---

# Persistent Softmax Optimization in Triton

## Summary
Triton PR #634 introduces a highly optimized `softmax` implementation for ROCm, leveraging a **persistent kernel** model alongside architecture-specific autotuning for AMD's CDNA GPUs. Softmax is fundamentally a memory-bound operation, and this implementation maximizes HBM memory bandwidth utilization and GPU occupancy through controlled program dispatch and optimal hardware configuration mapping.

## Intent and Architectural Focus
The primary intent of this PR is to provide an efficient baseline softmax implementation within the `perf-kernels` directory of `ROCm/triton`. Because softmax typically operates over the row dimension of a 2D tensor, the dominant performance bottleneck is fetching and storing tensor elements to/from High Bandwidth Memory (HBM).

The kernel is explicitly tuned for CDNA architectures (`cdna2`, `cdna3`, `cdna4`). It utilizes optimizations that ensure maximal streaming multiprocessor (Compute Unit) occupancy without incurring the overhead of launching extraneous thread blocks.

## Key Optimization Techniques

### 1. Persistent Kernel Execution
Instead of launching a grid size equal to the number of rows (`n_rows`), the implementation bounds the grid size to the number of physical Compute Units (`NUM_SM`):

```python
# Persistent kernel: Set num of programs equal to number of streaming multi-processors
num_programs = min(NUM_SM, n_rows)
grid = lambda meta: (num_programs, )
```

Inside the kernel, a grid-stride loop is used to process the rows sequentially. Each persistent program iteratively fetches the next unprocessed row:

```python
row_start = tl.program_id(0)
row_step = tl.num_programs(0)
for row_idx in tl.range(row_start, n_rows, row_step):
    # Process row
```

**Why it matters:** 
- **Reduced Scheduling Overhead**: Eliminates the hardware overhead of launching and scheduling large numbers of individual workgroups.
- **Cache Locality**: Allows waves already resident on the Compute Units to continuously stream through memory, maintaining steady state execution and minimizing pipeline bubbles.

### 2. ROCm-Specific Occupancy Tuning (`waves_per_eu`)
The PR introduces a custom autotuning schedule tailored specifically for HIP architectures:

```python
def get_hip_autotune_config():
    return [
        triton.Config({'waves_per_eu': 1}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=8, num_stages=1),
        # ... variations covering waves_per_eu of 1, 2, and 4
    ]
```

**Why it matters:** 
While NVIDIA architectures tune thread concurrency primarily via `num_warps` and registers, AMD GPUs provide explicit compiler control over `waves_per_eu` (Waves per Execution Unit) via Triton's HIP backend. By searching over `waves_per_eu = 1, 2, 4`, the autotuner maps out the Pareto frontier between hiding latency (higher wave limits) and preventing VGPR pressure/register spilling (lower wave limits).

### 3. Vectorized Memory Access
To saturate the HBM bus, the kernel utilizes 16-byte (128-bit) vectorized loads and stores by hinting to the compiler that the memory pointers are perfectly aligned:

```python
input_ptrs = tl.multiple_of(input_ptrs, (16, ))
row = tl.load(input_ptrs, mask=mask, other=-float('inf'), cache_modifier=".cg")

...

output_ptrs = tl.multiple_of(output_ptrs, (16, ))
tl.store(output_ptrs, softmax_output, mask=mask)
```

**Why it matters:** 
Asserting that pointers are multiples of 16 enables the Triton compiler to emit `global_load_dwordx4` and equivalent 128-bit wide memory instructions. This is essential for achieving peak memory throughput in a bandwidth-constrained kernel.

### 4. Cache Modifiers (`.cg`)
The kernel employs `cache_modifier=".cg"` (Cache Global) for its load instructions.

**Why it matters:** 
Softmax processes each row exactly once in a pure streaming read pattern, meaning there is zero temporal locality for the input data in the L1 cache. The `.cg` modifier instructs the hardware to bypass L1 caching entirely. This averts cache thrashing and preserves L1 capacity for other active wavefronts or operations, improving global bandwidth efficiency.

## Performance Characteristics and Bounds

- **Memory Bound**: Softmax computes $O(N)$ arithmetic operations for $O(N)$ data loaded, resulting in exceptionally low arithmetic intensity. Performance is strictly bound by HBM memory bandwidth.
- **Maximized Throughput**: Through persistent scheduling and 128-bit memory instructions, the kernel closely approaches theoretical memory bandwidth limits without being bottlenecked by instruction latency or launch overheads.
- **Power of Two Block Sizes**: The column reduction is mapped to `triton.next_power_of_2(n_cols)`, utilizing enough shared memory (LDS) to retain a full row during the reduction phases (calculating the max and the sum).

## Conclusion
This implementation establishes an excellent reference for authoring high-performance, memory-bound operations in Triton targeting AMD GPUs. It seamlessly synthesizes generalized access optimizations (vectorized loads, cache bypassing, persistent execution) with architecture-specific tuning (`waves_per_eu`) to fully leverage CDNA compute structures.

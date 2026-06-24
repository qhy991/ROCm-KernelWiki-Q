---
id: pr-triton-670-technique
title: "Persistent Kernel Optimization for Flash Attention Forward in ROCm Triton"
type: technique
tags:
  - rocm
  - flash-attention
  - triton
  - attention
  - optimization
  - performance
  - hardware
  - synchronization
  - scheduling
architectures:
  - cdna2
  - cdna3
  - cdna4
hardware_features:
  - compute-unit
techniques:
  - persistent-kernel
kernel_types:
  - flash-attention
  - attention
languages:
  - triton-rocm
sources:
  - pr-triton-670
---

# Persistent Kernel Optimization for Flash Attention Forward in ROCm Triton

## Overview
This technique analyzes the implementation of a **persistent kernel** variant for the Flash Attention forward pass in AMD's ROCm Triton backend, as introduced in PR #670. In standard execution models, Triton spawns a separate block (workgroup) for each query tile. However, for moderate sequence lengths—and particularly when executing causal attention—this strict 1:1 scheduling approach can lead to suboptimal utilization of Compute Units (CUs) and unbalanced workloads.

A persistent kernel mitigates this by launching a fixed or dynamically scaled number of workgroups across available CUs, which then actively pull and loop over the required workload tiles. This implementation also includes a crucial fix for correctly calculating floating-point operations (FLOPs) when query and key sequences are of unequal lengths in causal setups.

## Intent and Architecture
The primary intent behind this optimization is to **maximize hardware occupancy and minimize grid launch overhead** by keeping the CUs active and aggressively reusing them for multiple output tiles. 

### 1. The Persistent Kernel Execution Model
Instead of relying entirely on the default hardware scheduler to map a massive 3D grid `(num_tiles_m, num_heads, batch_size)` onto CUs, the persistent version calculates the total number of work units globally and explicitly manages scheduling within the kernel code.

Two modes of persistent execution are introduced:
- **Fixed Persistent Scheduling:** Workgroups stride over the problem size evenly. The loop increments the `tile_id` by `NUM_WG` (Number of Workgroups), which is determined by `NUM_CU * GRID_CU_MULTIP`.
- **Dynamic Persistent Scheduling:** CUs dynamically claim the next available tile from a global counter utilizing an atomic add operation (`atomic_counter.atomic_add(1)`). This behaves like a work-stealing queue and provides perfect load-balancing: if one workgroup finishes its tile faster, it immediately pulls the next chunk of work without waiting for slower workgroups.

### 2. Implementation Mechanics
In the persistent kernel loop (`while tile_id < num_tiles_total`), the global 1D index `tile_id` is unpacked to deduce the corresponding batch index, head index, and the target query tile block (`start_m`):
```python
if PERSISTENT:
    # Decode the flat tile_id back to grid coordinates
    off_z = tile_id // num_tiles_per_sample
    off_h_q = tile_id % num_tiles_per_sample // num_tiles_per_head
    start_m = tile_id % num_tiles_per_sample % num_tiles_per_head
```
This enables a unified loop that iterates efficiently, drastically improving GPU saturation. In the Triton configurations, this behavior is supported by passing a tunable parameter `'GRID_CU_MULTIP': 2`, explicitly directing the persistent configuration to launch double the amount of workgroups as there are SMs/CUs. This effectively overlaps computation and memory latencies by utilizing wave-mixing on AMD hardware.

## Causal FLOP Calculation Fix
When sequences have mismatched query (`seqlen_q`) and key (`seqlen_k`) lengths, the traditional `total_flops * 0.5` approximation for causal masking misrepresents the actual compute workload.

The corrected calculation properly accounts for the non-square, trapezoidal nature of the attention matrix:
```python
if causal:
    if seqlen_q > seqlen_k:
        # Tapered masking starts lower down the query sequence
        total_flops *= seqlen_k / (2 * seqlen_q)
    else:
        # Key sequence is larger, reducing the overall masked fraction
        total_flops *= 1 - seqlen_q / (2 * seqlen_k)
```
This guarantees accurate profiling and exact TFLOPs/sec benchmarking numbers for autoregressive decoding or asymmetric prompt processing.

## Memory Bounds & Performance Profiling
By persistently keeping the kernel executing on the CUs, this optimization minimizes the implicit register allocation and deallocation phases, and LDS (Local Data Share) memory resets.

1. **Moderate Sequence Lengths**: This technique shines for moderate sequence lengths where the total tile count might leave CUs momentarily starved as the grid tails off.
2. **Causal Attention Load Balancing**: Because causal attention leads to highly uneven workloads (early rows compute far fewer key tokens than later rows), **dynamic persistent scheduling** allows workgroups handling the "short" early query blocks to quickly move on to the next set of tiles. This naturally resolves the heavy load imbalance inherent to lower triangular masking.
3. **Atomic Overhead**: Dynamic scheduling utilizes an L2-resident atomic counter (`atomic_counter.atomic_add(1)`). The tiny latency cost of this global atomic synchronization is easily amortized and hidden by the large algorithmic complexity (GEMMs) processed inside each tile step.

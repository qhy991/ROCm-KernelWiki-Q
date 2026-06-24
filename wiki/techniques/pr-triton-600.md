---
id: technique-pr-triton-600
title: "Pruning LDS Usage in Triton's New Pipeliner"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
hardware_features: [lds]
techniques: [occupancy-tuning]
tags: [rocm, lds, triton, optimization, memory-bound, occupancy, pipeline]
confidence: inferred
sources: [pr-triton-600]
---

# Pruning LDS Usage in Triton's New Pipeliner

## Context and Intent
In Triton, software pipelining is critical for overlapping global memory loads (HBM to LDS) with computation (LDS to VGPRs or Matrix Cores). Pipelining typically requires allocating multiple buffers in Local Data Share (LDS) memory—one for each stage of the pipeline—to prevent data races between concurrent load and compute instructions.

PR #600 in `ROCm/triton` addresses an inefficiency in how the "new pipeliner" calculated LDS requirements. The primary intent is to **prune unnecessary LDS usage** by strictly bounding the memory allocation to the exact number of active pipeline stages (`num_stage`).

## Technical Details

### The Problem
Previously, the LDS allocation logic in Triton's compiler pipeline passes failed to tightly bound the buffer sizes based on dynamically configured pipeline depths, leading to over-allocated shared memory. Over-allocating LDS has severe consequences on AMD CDNA architectures:
- **Reduced Occupancy:** Compute Units (CUs) have a fixed amount of LDS (typically 64KB or 128KB, depending on the architecture and configuration). If a workgroup reserves more LDS than it strictly needs, fewer workgroups can be scheduled concurrently on the same CU, reducing the number of active wavefronts.
- **Resource Exhaustion:** Overestimating LDS footprint can lead to compilation failures if the calculated footprint artificially exceeds the physical maximum LDS per workgroup.

### The Optimization Technique
The PR modifies the compiler's shared memory allocation logic to dynamically and strictly multiply the base buffer size by the configured `num_stage`.
- **Exact Sizing:** `LDS_usage = base_buffer_size * num_stage`. This ensures that a kernel configured with, for example, 3 stages only pays the exact LDS cost for those 3 stages, removing any unused padding or overestimated overhead.
- **Legacy Fallback:** The implementation specifically checks for `num_stages == 0`. This preserves backward compatibility with Triton's "stream pipeline" (an older or alternative software pipelining strategy where stages might not map directly to linearly scaled multi-buffered LDS allocations).

## Memory Bounds and Performance Implications

By pruning LDS usage, this compiler optimization directly expands the tuning space for Triton kernels:

1. **Higher Occupancy:** Workgroups become "lighter" in their LDS footprint. This increases the maximum achievable wavefront occupancy, which is essential for hiding instruction latency and improving arithmetic throughput.
2. **Deeper Pipelines:** With exact sizing, developers and autotuners can push for higher `num_stage` values without prematurely hitting the physical LDS ceiling. Deeper pipelines are critical for hiding long-latency HBM accesses in memory-bound kernels like Flash Attention or large-scale GEMMs.
3. **Improved Bandwidth Utilization:** The synergistic effect of better latency hiding (via deeper pipelines) and higher occupancy directly translates to higher sustained HBM bandwidth utilization across the GPU.

## Related Hardware Features
- **LDS (Local Data Share):** The critical on-chip SRAM resource whose allocation is being tightly optimized.
- **Wavefront Occupancy:** The scheduling metric directly improved by reducing per-workgroup LDS reservations.

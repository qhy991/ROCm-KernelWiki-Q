---
id: technique-pr-triton-661
title: "Triton stream-pipelineV2 LDS Usage Optimization"
type: wiki-technique
architectures:
  - cdna2
  - cdna3
  - cdna4
tags:
  - optimization
  - memory
  - pipeline
  - tiling
  - lds
  - double-buffering
confidence: inferred
sources:
  - pr-triton-661
---

# Triton stream-pipelineV2 LDS Usage Optimization

## Context
In Triton's AMD backend, specifically for GEMM kernels, the tuner (`tune_gemm.py`) iterates through various tiling and pipelining configurations to find the optimal combination for given matrix shapes (`M`, `N`, `K`). A critical step in the auto-tuning process is pruning configurations that would exceed the hardware limits, particularly the shared memory (LDS - Local Data Share) capacity limit, which is typically 64KB (65536 bytes) per workgroup on many CDNA configurations.

Before this optimization, the tuning script assumed a naive allocation of LDS buffers, multiplying the total LDS needed for one stage of operands A and B by the total number of pipeline stages (`num_stages`). This led to overly conservative LDS estimates, incorrectly pruning viable and highly performant pipeline configurations.

## Technique

### stream-pipelineV2 and LDS Allocation

The ROCm backend of Triton uses a specialized scheduling pass known as `stream-pipelineV2` for optimizing software pipelines. This pass optimizes the movement of data between global memory, LDS, and registers. 

The previous calculation of LDS usage for validation was:
```python
LDS = (LDSA + LDSB) * max(1, num_stages)
```

The PR updates the LDS requirement based on how `stream-pipelineV2` actually manages buffers during execution:

1. **No Pipeline (`num_stages <= 1`)**:
   When no software pipeline is active, we do not need simultaneous persistent buffers for all operands to hide global memory latency. Instead, buffer A and buffer B can re-use the same shared memory allocation sequentially or overlap in ways that don't compound their footprints. Thus, the LDS requirement is simply the maximum size between block A and block B:
   ```python
   LDS = max(LDSA, LDSB)
   ```

2. **Pipelined (`num_stages > 1`)**:
   When software pipelining is enabled, data must be prefetched, requiring multi-buffering. However, the `stream-pipelineV2` implementation requires `num_stages - 1` buffers in LDS at the same time, not `num_stages`. Because one pipeline stage's data is actively loaded into registers (VGPRs) for computation, the LDS footprint is reduced by exactly one stage. 
   ```python
   LDS = (LDSA + LDSB) * (num_stages - 1)
   ```

## Performance Impact and Memory Bounds

This logic refinement has a direct impact on the configurations explored during `tune_gemm`, broadening the search space for the auto-tuner:

- **Enhanced Occupancy**: By accurately modeling the reduced LDS footprint (`num_stages - 1` buffers instead of `num_stages`), more aggressive tile sizes (e.g., larger `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`) become valid and fit within the 64KB threshold.
- **Deeper Pipelines**: A configuration that previously required 3 pipeline stages might have been pruned if `3 * (LDSA + LDSB) > 64KB`. With the new formula, it requires only `2 * (LDSA + LDSB)`, which could safely pass the filter. Deep pipelines are essential for hiding memory latency in memory-bound GEMM kernels.
- **Tuning Precision**: By preventing early rejection of mathematically viable shapes, the autotuner can discover configurations that are natively supported by `stream-pipelineV2`, pushing peak TFLOPS closer to the hardware limit.

## Implementation Details

The core fix corrects the heuristic filter logic within `python/perf-kernels/tools/tune_gemm/tune_gemm.py`.

```python
LDSA = BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a
LDSB = BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b

if num_stages <= 1:
    # No pipeline, buffer A and buffer B can re-use each other
    LDS = max(LDSA, LDSB)
else:
    # Pipeline, we need (num_stages - 1) buffers for both A and B at the same time
    LDS = (LDSA + LDSB) * (num_stages - 1)

if LDS > 65536:
    continue
```

This precise modeling ensures that configurations are only rejected if they truly exceed physical hardware limitations.

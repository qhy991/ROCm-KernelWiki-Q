---
id: technique-persistent-kernel
title: Persistent Kernel Pattern
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [persistent-kernel, gws, scheduling, optimization]
confidence: source-reported
techniques: [persistent-kernel]
hardware_features: [gws]
kernel_types: [gemm, attention, moe, grouped-gemm]
related: [hw-gws, hw-mfma-matrix-core, technique-ck-tile-programming]
sources: [doc-mi300x-workload, blog-amdgpu-kernel-opt]
reproducibility: snippet
---

# Persistent Kernel Pattern

A persistent kernel is a long-running GPU kernel that stays resident on the device, processing multiple work tiles from a global work queue rather than launching a new kernel for each batch of work.

## When to Use

| Scenario | Persistent? | Reason |
|----------|-------------|--------|
| Many small GEMMs (MoE) | ✓ | Avoid kernel launch overhead |
| Streaming inference | ✓ | Pipeline with data arrival |
| Attention with variable seq lengths | ✓ | Adaptive tile scheduling |
| Single large GEMM | ✗ | Standard tiled kernel is fine |
| Training forward+backward | ✗ | Separate kernels needed |

## Pattern Structure

```c
__global__ void persistent_moe_gemm(
    const float* weights,  // [num_experts, K, N]
    const float* inputs,   // [total_tokens, K]
    float* outputs,        // [total_tokens, N]
    const int* expert_ids, // [total_tokens]
    int total_tokens,
    int K, int N) {

    // Get cooperative grid for global sync
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    // Global work queue (in global memory)
    // Each block atomically claims a tile
    while (true) {
        int tile_id = atomicAdd(&work_queue.head, 1);
        if (tile_id >= total_tokens) break;

        int token = tile_id;
        int expert = expert_ids[token];

        // Compute GEMM for this token-expert pair
        const float* W = weights + expert * K * N;
        const float* X = inputs + token * K;
        float* Y = outputs + token * N;

        // Load W tile and X, compute via MFMA
        gemm_tile(W, X, Y, K, N);
    }

    // Wait for all blocks to finish before next round
    grid.sync();
}
```

## Launch Configuration

```c
// Must use cooperative launch for grid-wide (GWS) sync.
// hipLaunchCooperativeKernel takes a void** array of argument pointers.
int num_blocks = num_CUs * max_blocks_per_CU;
void* args[] = { &weights, &inputs, &outputs, &expert_ids,
                 &total_tokens, &K, &N };
hipLaunchCooperativeKernel(
    reinterpret_cast<void*>(persistent_moe_gemm),
    dim3(num_blocks), dim3(256),
    args,
    0 /* shared mem */, stream);
```

## Work Queue Design

### Lock-Free Atomic Queue

```c
struct WorkQueue {
    int head;        // Next tile to claim
    int total;       // Total tiles
    int completed;   // Completed count
};

// Each block claims one tile atomically
int tile = atomicAdd(&queue->head, 1);
if (tile >= queue->total) break;
```

### Grouped Claim (for tile-level parallelism)

```c
// Each block claims a GROUP of tiles
int group_start = atomicAdd(&queue->head, TILES_PER_GROUP);
for (int t = group_start; t < group_start + TILES_PER_GROUP && t < total; t++) {
    process_tile(t);
}
```

## Performance Characteristics

| Metric | Standard Launch | Persistent Kernel |
|--------|----------------|-------------------|
| Launch overhead | ~5-20μs per kernel | One-time ~20μs |
| Work distribution | Static (grid) | Dynamic (queue) |
| Load balancing | May be uneven | Self-balancing |
| CU utilization | Per-kernel | Continuous |
| Code complexity | Simple | Higher |

## Common Applications on ROCm

1. **MoE GEMM**: Route tokens to experts dynamically via work queue
2. **Flash Attention**: Process variable-length sequences in one kernel
3. **Grouped GEMM**: Variable M sizes across groups
4. **Batch decoding**: Process multiple decode requests concurrently

## CK Support

```c
// CK provides persistent kernel templates
#include "ck/tensor_operation/operator/gridwise_gemm_pipeline.hpp"
// CK's gridwise GEMM can operate in persistent mode with work queues
```

## References

- [MI300X Workload Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [HIP Cooperative Groups](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___c_g.html)
- [AMDGPU Kernel Optimization Guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md)

---
id: hw-gws
title: GWS — Global Wave Sync
type: wiki-hardware
architectures: [cdna2, cdna3, cdna4]
tags: [gws, synchronization, hardware, cross-cu]
confidence: source-reported
hardware_features: [gws]
related: [hw-wavefront, hw-lds]
sources: [doc-cdna4-isa, doc-cdna4-whitepaper]
cuda_equivalent: cooperative_groups_grid
---

# GWS — Global Wave Sync

Global Wave Sync (GWS) is AMD's cross-CU synchronization mechanism, enabling persistent kernels and coordinated execution across multiple compute units.

## Overview

| Property | CDNA2 | CDNA3 | CDNA4 |
|----------|-------|-------|-------|
| GWS scope | All CUs on GCD | All CUs on GCD | All CUs on GCD |
| Barrier type | Global | Global | Global |
| Launch support | `hipLaunchCooperativeKernel` | ✓ | ✓ |

GWS allows all wavefronts across all CUs to synchronize at a global barrier — essential for persistent kernel patterns where a single kernel runs for the entire application lifetime.

## HIP Usage

### Cooperative Kernel Launch

```c
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;

// Launch a cooperative kernel (enables GWS-based grid-wide synchronization).
// NOTE: hipLaunchCooperativeKernel takes a void** ARRAY of argument pointers,
// not varargs, and the kernel function pointer must be cast to void*.
void* args[] = { &d_arg0, &d_arg1 /* , ... */ };
hipLaunchCooperativeKernel(
    reinterpret_cast<void*>(my_persistent_kernel),
    grid, block,
    args,        // array of pointers to kernel arguments
    0,           // dynamic shared memory bytes
    stream);

// Inside the kernel:
__global__ void my_persistent_kernel(...) {
    // Each block processes work items from a global queue
    while (has_work()) {
        process_tile(...);

        // Global barrier across ALL blocks in the grid
        cg::this_grid().sync();
    }
}
```

### Cooperative Groups

```c
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void persistent_gemm_kernel(...) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    while (work_queue.has_work()) {
        // Each block claims a tile
        int tile_id = atomicAdd(&work_queue.counter, 1);
        if (tile_id >= num_tiles) break;

        process_gemm_tile(tile_id, ...);

        // Sync entire grid before next round
        grid.sync();
    }
}
```

## GWS Instructions

The Global Wave Sync hardware is driven by the `ds_gws_*` instruction family (issued through the LDS/DS unit), not by the workgroup-scoped `s_barrier*` scalar instructions. In practice these are emitted by the compiler/runtime for cooperative launches rather than written by hand.

| Instruction | Description |
|-------------|-------------|
| `ds_gws_init` | Initialize a GWS resource (barrier/semaphore) for the dispatch |
| `ds_gws_barrier` | Cross-CU barrier across participating wavefronts |
| `ds_gws_sema_v` / `ds_gws_sema_p` | GWS semaphore signal (V) / wait (P) |
| `ds_gws_sema_release_all` | Release all waiters on the GWS semaphore |

> The `s_barrier` / `s_barrier_signal` / `s_barrier_wait` scalar instructions synchronize wavefronts **within a single workgroup**, which is a different (narrower) scope than GWS cross-CU sync. Confirm exact `ds_gws_*` operand encodings against the ISA PDF.

## Persistent Kernel Pattern

```
┌─────────────────────────────────────────────┐
│ Host: hipLaunchCooperativeKernel()           │
│   Grid: N blocks × 256 threads              │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│ GPU: Persistent Kernel                      │
│                                              │
│   while (global_queue.has_work()) {          │
│     tile_id = atomicAdd(&queue.head, 1)      │
│     if (tile_id >= total) break              │
│     process(tile_id)                         │
│     grid.sync()  // ← GWS barrier           │
│   }                                          │
└─────────────────────────────────────────────┘
```

## vs CUDA Cooperative Groups

| Feature | CUDA | ROCm |
|---------|------|------|
| Launch | `cudaLaunchCooperativeKernel` | `hipLaunchCooperativeKernel` |
| Grid sync | `cg::this_grid().sync()` | Same API |
| Multi-GPU | `cg::this_multi_grid()` | Limited support |
| Hardware | Cluster Launch Control (Blackwell) | GWS (CDNA2+) |

## Performance Considerations

1. **Launch overhead**: Cooperative kernels have higher launch cost — use for long-running kernels only
2. **Occupancy**: All blocks must fit simultaneously — limit block count to `maxActiveBlocks`
3. **Queue design**: Use lock-free atomic queues to avoid contention between blocks
4. **Memory**: Work queue must be in global memory visible to all CUs

```c
// Query max cooperative blocks
int max_blocks;
hipOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_blocks, my_kernel, block_size, shared_mem);
int total_blocks = max_blocks * num_SM;
```

## References

- [HIP Cooperative Groups](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___c_g.html)
- [CDNA4 ISA Reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/)

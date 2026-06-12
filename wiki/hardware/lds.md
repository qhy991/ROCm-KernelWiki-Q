---
id: hw-lds
title: LDS — Local Data Share
type: wiki-hardware
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [lds, memory, hardware]
confidence: verified
hardware_features: [lds, lds-transpose]
related: [hw-mfma-matrix-core, hw-wavefront]
sources: [doc-cdna4-isa, doc-cdna4-whitepaper]
cuda_equivalent: shared_memory
---

# LDS — Local Data Share

AMD's equivalent to CUDA shared memory. Fast on-chip memory shared by all threads in a workgroup.

## Specifications

| Property | CDNA1-3 | CDNA4 |
|----------|---------|-------|
| Size per CU | 64 KB | 64 KB |
| Number of banks | 32 | 32 |
| Bank width | 4 bytes (dword) | 4 bytes (dword) |
| Read ports | 2 per cycle | 2 per cycle |
| Write ports | 2 per cycle | 2 per cycle |
| Latency | ~4-8 cycles | ~4-8 cycles |
| Addressable by | DS instructions | DS instructions |
| **Transpose read** | ✗ | ✓ (new in CDNA4) |

## Bank Layout

```
Address (bytes)    Bank
0x000 - 0x003     Bank 0
0x004 - 0x007     Bank 1
0x008 - 0x00B     Bank 2
...
0x07C - 0x07F     Bank 31
0x080 - 0x083     Bank 0    ← wraps!
```

- 32 banks, each 4 bytes wide
- Consecutive 4-byte words map to consecutive banks
- Bank index = (address / 4) % 32

## HIP Usage

```c
// Declare LDS (shared) memory
__shared__ float tile[256][64];

// Static allocation
extern __shared__ float dynamic_tile[];  // dynamically sized

// Access from kernel
__global__ void my_kernel(float* output) {
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = ...;

    // Synchronize to ensure all writes are visible
    __syncthreads();

    float val = smem[(threadIdx.x + 1) % BLOCK_SIZE];
}
```

## Common LDS Instructions

| Instruction | Operation | Width |
|-------------|-----------|-------|
| `ds_read_b32` | Read 1 dword | 4 bytes |
| `ds_read_b64` | Read 2 dwords | 8 bytes |
| `ds_read_b128` | Read 4 dwords | 16 bytes |
| `ds_read2_b64` | Read 2×2 dwords (paired) | 16 bytes |
| `ds_read2st64_b64` | Read 2×2 dwords (stride-64) | 16 bytes |
| `ds_write_b32` | Write 1 dword | 4 bytes |
| `ds_write_b64` | Write 2 dwords | 8 bytes |
| `ds_write_b128` | Write 4 dwords | 16 bytes |
| `ds_write2st64_b64` | Write 2×2 dwords (stride-64) | 16 bytes |

## CDNA4 Enhancement: Read-with-Transpose

CDNA4 introduces LDS instructions that can read and transpose data in a single operation:

```asm
; CDNA4 only: read 16x16 tile and transpose
ds_read_transt_b128 v[0:3], v0 offset:0
```

This eliminates the need for separate transpose kernels in GEMM epilogues and attention score computation.

## Performance Optimization

### Bank Conflict Avoidance

```c
// BAD: 64-wide rows cause 2-way conflicts
__shared__ float mat[64][64];

// GOOD: pad to 65 to break conflict
__shared__ float mat[64][65];
```

See [technique-bank-conflict-padding](../techniques/bank-conflict-padding.md) for full details.

### Vectorized Access

```c
// Use 128-bit loads for maximum throughput
float4 data = *((float4*)&smem[row * 65 + col]);
```

### Asynchronous Copy (CDNA3+)

CDNA3+ supports limited async copy from global memory to LDS:

```c
// Using FLAT instructions with non-temporal hints
// Compiler may generate optimized sequences
```

> Note: AMD has no direct equivalent to NVIDIA's TMA (Tensor Memory Accelerator).
> Data movement from global to LDS must go through VGPRs or use flat loads.

## Occupancy Impact

LDS usage directly affects occupancy:

| LDS per workgroup | Max workgroups/CU |
|-------------------|-------------------|
| 0 KB | 32+ |
| 16 KB | 4 |
| 32 KB | 2 |
| 64 KB | 1 |

Use `hipOccupancyMaxPotentialBlockSize` to find optimal block sizes.

## References

- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [AMDGPU Kernel Optimization Guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md)

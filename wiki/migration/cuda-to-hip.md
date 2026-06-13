---
id: migration-cuda-to-hip
title: CUDA → HIP Migration Guide
type: wiki-migration
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [migration, cuda, hip, porting]
confidence: verified
from_concept: CUDA
to_concept: HIP
difficulty: moderate
related: [hw-mfma-matrix-core, hw-lds, hw-dpp-cross-lane]
sources: [doc-hip-programming-guide, doc-hip-porting-guide]
---

# CUDA → HIP Migration Guide

Step-by-step guide for porting CUDA kernels and applications to AMD HIP.

## Automatic Conversion

### hipify — The Starting Point

```bash
# Install hipify
pip install hipify

# Convert a single file
hipify-perl foo.cu > foo.hip

# Convert a project
hipify-perl -p ./project -o ./project-hip
```

**What hipify does**:
- `cudaMalloc` → `hipMalloc`
- `__global__` → `__global__` (same!)
- `__shared__` → `__shared__` (same!)
- `__syncthreads()` → `__syncthreads()` (same!)
- `threadIdx.x` → `threadIdx.x` (same!)

**What hipify does NOT do**:
- Warp-level operations (shuffle semantics differ)
- Tensor Core / Matrix Core code (WMMA → rocWMMA)
- Architecture-specific optimizations (shared → LDS bank layout)
- SASS-level inline assembly

## API Mapping Quick Reference

### Runtime API

| CUDA | HIP | Notes |
|------|-----|-------|
| `cudaMalloc` | `hipMalloc` | 1:1 |
| `cudaFree` | `hipFree` | 1:1 |
| `cudaMemcpy` | `hipMemcpy` | 1:1 |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | 1:1 |
| `cudaStreamCreate` | `hipStreamCreate` | 1:1 |
| `cudaGetDeviceCount` | `hipGetDeviceCount` | 1:1 |
| `cudaSetDevice` | `hipSetDevice` | 1:1 |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | 1:1 |
| `cudaGetLastError` | `hipGetLastError` | 1:1 |
| `cudaGetErrorString` | `hipGetErrorString` | 1:1 |
| `cudaLaunchKernel` | `hipLaunchKernel` | 1:1 |

### Kernel Launch

```c
// CUDA
myKernel<<<grid, block, shared_mem, stream>>>(args);

// HIP (option 1 — same syntax)
myKernel<<<grid, block, shared_mem, stream>>>(args);

// HIP (option 2 — explicit)
hipLaunchKernelGGL(myKernel, grid, block, shared_mem, stream, args);
```

### Device Builtins

| CUDA | HIP | Key Difference |
|------|-----|---------------|
| `threadIdx.x` | `threadIdx.x` | Same |
| `blockIdx.x` | `blockIdx.x` | Same |
| `blockDim.x` | `blockDim.x` | Same |
| `__syncthreads()` | `__syncthreads()` | Same |
| `__shfl_sync()` | `__shfl()` | **No mask param!** |
| `__shfl_up_sync()` | `__shfl_up()` | **No mask!** |
| `__shfl_down_sync()` | `__shfl_down()` | **No mask!** |
| `__shfl_xor_sync()` | `__shfl_xor()` | **No mask!** |
| `__ballot_sync()` | `__ballot()` | **No mask!** |
| `__any_sync()` | `__any()` | **No mask!** |
| `__all_sync()` | `__all()` | **No mask!** |

## Critical Differences

### 1. Wavefront Size (64 vs 32)

```c
// CUDA: warpSize is always 32
int warp = threadIdx.x / warpSize;  // warpSize = 32

// HIP: warpSize is 64 on CDNA!
int wave = threadIdx.x / warpSize;  // warpSize = 64

// Portable code:
int lane = threadIdx.x % warpSize;
int warp_id = threadIdx.x / warpSize;
```

### 2. Warp Shuffle Without Masks

```c
// CUDA (with mask)
float val = __shfl_sync(0xFFFFFFFF, my_val, src_lane, 32);

// HIP (no mask — all lanes participate)
float val = __shfl(my_val, src_lane, warpSize);
```

### 3. Shared Memory → LDS

```c
// Same declaration syntax
__shared__ float smem[256];

// But bank layout differs:
// CUDA: 32 banks, 4-byte width
// AMD:  32 banks, 4-byte width (same!)
// But wavefront = 64 threads → different conflict patterns

// Padding for conflict avoidance may need adjustment
```

### 4. Tensor Core → Matrix Core

```c
// CUDA WMMA
#include <mma.h>
using namespace nvcuda::wmma;
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;

// HIP rocWMMA
#include <rocwmma/rocwmma.h>
using namespace rocwmma;
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;

// API is nearly identical, but tile sizes and scheduling differ
```

### 5. Cooperative Groups

```c
// CUDA
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
auto block = cg::this_thread_block();

// HIP (limited support)
// HIP has basic cooperative group support but not full parity
// Use __syncthreads() and barrier intrinsics instead
```

### 6. Grid Synchronization

```c
// CUDA: cooperative kernels — args is a void** array of argument pointers
void* args[] = { &arg0, &arg1 };
cudaLaunchCooperativeKernel((void*)myKernel, grid, block, args, 0, stream);

// HIP: Global Wave Sync (GWS), available on CDNA2+.
// Direct 1:1 API mapping — same void** args-array signature, no "GGL" variant exists.
hipLaunchCooperativeKernel((void*)myKernel, grid, block, args, 0, stream);
```

## Build System Changes

```cmake
# CUDA
find_package(CUDA REQUIRED)
cuda_add_executable(my_app kernel.cu)

# HIP
find_package(hip REQUIRED)
add_executable(my_app kernel.hip)
target_link_libraries(my_app hip::host hip::device)
```

## Compilation

```bash
# CUDA
nvcc -arch=sm_80 kernel.cu -o kernel

# HIP
hipcc --offload-arch=gfx942 kernel.hip -o kernel
# Or use target ID for CDNA3:
hipcc --offload-arch=gfx942:sramecc+:xnack- kernel.hip -o kernel
```

## Common Pitfalls

| Pitfall | Description | Fix |
|---------|-------------|-----|
| Warp size assumption | Code assumes warpSize=32 | Use `warpSize` variable, don't hardcode |
| Shuffle mask | HIP shuffles have no mask parameter | Remove mask args |
| Register pressure | 64-lane wavefront uses more VGPRs | Reduce block size or tile size |
| Bank conflict pattern | Different conflict pattern with 64 threads | Re-tune LDS padding |
| No TMA | AMD has no Tensor Memory Accelerator | Use flat loads + LDS copies |
| No TMEM | AMD has no Tensor Memory | Use VGPR accumulators |
| cuBLAS API differences | Some cuBLAS extensions not in rocBLAS | Check rocBLAS docs |

## References

- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- [HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- [CUDA-to-HIP API Mapping](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/cuda_to_hip_api_mapping.html)

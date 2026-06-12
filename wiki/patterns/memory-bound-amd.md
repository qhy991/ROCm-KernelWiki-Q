---
id: pattern-memory-bound-amd
title: Memory-Bound Kernel Optimization on AMD CDNA
type: wiki-pattern
architectures: [cdna2, cdna3, cdna4]
tags: [memory-bound, optimization, bandwidth, hbm]
confidence: source-reported
sources: [doc-mi300x-workload, blog-amdgpu-kernel-opt]
related: [hw-lds, technique-vectorized-load, technique-double-buffering]
---

# Memory-Bound Kernel Optimization on AMD CDNA

Diagnosis and solutions for kernels limited by memory bandwidth on AMD MI-series GPUs.

## Symptoms

| Symptom | How to Detect |
|---------|--------------|
| Low compute utilization | `rocprof-compute` shows <50% MFMA utilization |
| High HBM traffic | HBM bandwidth near theoretical peak |
| Low SM/CU occupancy | `rocprof-sys` shows low active waves/CU |
| Kernel time scales with data size | Not with compute complexity |

### Diagnosis Commands

```bash
# Basic profiling
rocprofv3 --stats ./my_app

# Detailed memory analysis
rocprof-compute --hsa-stats ./my_app

# Check bandwidth utilization
rocprof-compute --metric-group SQ,TA,TCP,TCC ./my_app
```

## HBM Specifications

| GPU | HBM | Bandwidth | Theoretical BW |
|-----|-----|-----------|----------------|
| MI100 | 32 GB HBM2e | 1.2 TB/s | — |
| MI250X | 128 GB HBM2e | 3.2 TB/s | — |
| MI300X | 192 GB HBM3 | 5.3 TB/s | — |
| MI350X | 288 GB HBM3e | ~8 TB/s | — |

## Solutions

### 1. Vectorized Loads (128-bit)

```c
// BAD: 32-bit loads (one per thread)
float val = input[idx];

// GOOD: 128-bit loads (4 floats per thread)
float4 vals = *((float4*)&input[idx * 4]);
```

Effect: 4× fewer memory transactions, better bus utilization.

### 2. Memory Access Coalescing

```c
// BAD: Strided access (thread 0 reads [0], thread 1 reads [N])
// Each warp generates scattered memory requests

// GOOD: Consecutive access (thread 0 reads [0], thread 1 reads [1])
// Warp generates contiguous memory request
float val = input[threadIdx.x + blockIdx.x * blockDim.x];
```

### 3. Double Buffering

```c
// Overlap HBM→LDS copy with compute
__shared__ float tile_a[2][TILE_SIZE];
__shared__ float tile_b[2][TILE_SIZE];

// While computing tile[0], load tile[1]
// Ping-pong between buffers
int buf = 0;
for (int k = 0; k < K; k += TILE_K) {
    async_load(tile_a[1-buf], a_ptr + next_offset);  // Start async load
    compute(tile_a[buf], tile_b[buf]);                 // Compute current
    buf = 1 - buf;                                      // Swap
}
```

### 4. Data Layout Transformation

```c
// BAD: AoS (Array of Structures)
struct Point { float x, y, z; };
Point points[N];
// Accessing all x values: stride-3, terrible coalescing

// GOOD: SoA (Structure of Arrays)
float x[N], y[N], z[N];
// Accessing all x values: stride-1, perfect coalescing
```

### 5. Kernel Fusion

```c
// BAD: Two separate kernels → 2× HBM traffic
kernel_add<<<...>>>(a, b, temp, N);     // Write temp to HBM
kernel_mul<<<...>>>(temp, c, out, N);   // Read temp from HBM

// GOOD: Fused kernel → 1× HBM traffic
kernel_fused<<<...>>>(a, b, c, out, N); // temp stays in VGPR/LDS
```

### 6. Quantization (FP8/BF8 on CDNA3+)

```c
// Store weights in FP8, compute with MFMA FP8 instructions
// 2× reduction in memory traffic for same compute
using weight_t = __fp8;  // CDNA3+ FP8 type
v_mfma_f32_16x16x32_fp8_bf8  // 32 FP8 elements per MFMA
```

## Quantitative Impact

| Technique | Typical BW Improvement | Notes |
|-----------|----------------------|-------|
| Vectorized loads (128-bit) | 1.5-2× | Most impactful for simple kernels |
| Coalescing | 2-5× | Critical if access pattern is scattered |
| Double buffering | 1.3-1.8× | Overlaps compute and memory |
| Kernel fusion | 1.5-3× | Depends on intermediate data size |
| FP8 quantization | 2× | CDNA3+ only, reduces traffic by half |
| SoA layout | 2-4× | If original layout is AoS |

## References

- [MI300X Workload Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [AMDGPU Kernel Optimization Guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md)

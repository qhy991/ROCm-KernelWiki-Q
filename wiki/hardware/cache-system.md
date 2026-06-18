---
id: hw-cache-system
title: Cache System and Memory Hierarchy
type: wiki-hardware
architectures: [cdna1, cdna2, cdna3]
tags: [hardware, memory, bandwidth, mi300x, optimization]
confidence: source-reported
hardware_features: []
related: []
sources: []
cuda_equivalent: l1_l2_cache
---

# Cache System and Memory Hierarchy

The AMD Instinct (CDNA) architecture employs a deep, multi-level cache hierarchy to feed high-bandwidth data to the matrix cores (MFMA) and vector ALUs. Understanding and optimizing for the L1, L2, and Infinity Cache (on CDNA 3) is critical for achieving peak memory bandwidth and compute utilization.

## Cache Hierarchy Overview

The CDNA cache structure varies by generation, but it generally features Compute Unit (CU)-local L1 caches, an L2 cache partitioned across the compute dies, and (in CDNA 3) a massive Infinity Cache (MALL).

### L1 Caches (Per CU)

Each Compute Unit contains specialized L1 caches to handle different types of data:
- **Vector L1 Data Cache (vL1D):** The primary data cache for vector instructions.
  - **CDNA 2 (MI250X):** 16 KB, 64-way associative per CU.
  - **CDNA 3 (MI300X):** 32 KB per CU.
  - **Cache Line Size:** 64 bytes.
- **Instruction Cache (L1I):** Caches shader instructions. 
  - **CDNA 2:** 32 KB per CU.
  - **CDNA 3:** 64 KB shared per pair of CUs.
- **Scalar Cache (K Cache):** 16 KB per CU, used for scalar data and constants.

### L2 Cache (Die-Local)

The L2 cache acts as a coalescing point and the main on-die buffer before data requests fall back to the last-level cache or HBM.
- **CDNA 2 (MI250X):** 8 MB of L2 cache per Graphics Compute Die (GCD), partitioned into 32 slices. A full MI250X (2 GCDs) has 16 MB total.
- **CDNA 3 (MI300X):** 4 MB per Accelerator Complex Die (XCD).
- **Cache Line Size:** Typically operates with a **128-byte** full-line size for reads from L2, but supports **64-byte** half-line writes per clock cycle.

### Infinity Cache / MALL (CDNA 3)

Introduced to the data center lineup with CDNA 3, the AMD Infinity Cache (Memory-Attached Last-Level cache, or MALL) fundamentally shifts how memory-bound workloads (like LLMs) perform.
- **Capacity:** 256 MB device-wide on MI300X.
- **Function:** Acts as an L3-style caching layer, bridging the multiple XCDs and minimizing costly reads from the 192 GB HBM3 memory pools. 
- **Impact:** Greatly accelerates KV-cache lookups and weight streaming for generative AI, boosting effective bandwidth well beyond the 5.3 TB/s HBM3 theoretical peak.

## Cache Optimization Strategies

### 1. Maximizing Cache Line Utilization (Coalescing)

Since the L1 cache line size is 64 bytes and the L2 read line size is 128 bytes, contiguous memory access by adjacent threads in a wavefront is essential.

- **Vectorized Loads:** Use 128-bit vector types (e.g., `float4`, `uint4`) to fetch data. A wavefront of 64 threads loading 16 bytes each will request exactly 1024 bytes per instruction, perfectly utilizing multiple 64-byte and 128-byte cache lines without waste.
- **Memory Coalescing:** Ensure thread `i` accesses memory offset `i`. If threads access memory randomly or with large strides, cache lines will be fetched but only partially used, wasting memory bandwidth.

```cpp
// Example: Coalesced vectorized load in HIP
__global__ void vectorized_copy(const float4* in, float4* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread loads 16 bytes. 
    // Wavefront (64 threads) loads 1024 bytes (16 x 64-byte cache lines) contiguously.
    float4 val = in[idx]; 
    out[idx] = val;
}
```

### 2. Cache Hit Rate Optimization via Tiling

For compute-intensive operations like GEMM, data must be tiled so that working sets fit within the L1/L2 caches and LDS.

- **L1/LDS Tiling:** Bring small blocks of data (e.g., 16x16 or 32x32 tiles) from L2/HBM into the CU's LDS (Local Data Share) or keep them in VGPRs. This prevents repeatedly evicting data from the 16KB/32KB vL1D cache.
- **L2 Tiling:** Group workgroups that operate on the same data onto the same XCD/GCD. This maximizes the L2 cache hit rate (8MB per GCD on MI250X, 4MB per XCD on MI300X).

### 3. Non-Temporal Memory Accesses (Cache Bypassing)

If data is read or written only once (e.g., streaming large arrays or writing the final output of a matrix multiplication), caching it in L1 or L2 will evict other useful data. 
Use non-temporal hints to bypass caches:

- **HIP/C++:** Use `__builtin_nontemporal_store()` or `__builtin_nontemporal_load()`.
- **Assembly:** Use cache modifiers like `glc` (globally coherent, bypass L1), `slc` (system level coherent, bypass L2), or `nt` (non-temporal) on buffer load/store instructions.

```cpp
__global__ void bypass_cache_store(float* out, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Bypasses L1/L2 caches, writing directly to HBM to avoid polluting caches
    __builtin_nontemporal_store(val, &out[idx]); 
}
```

## Performance Metrics & Profiling

To analyze cache hit rates and memory bottlenecks, use **rocprof** to collect hardware performance counters:

- **L1 Hit Rate:** `TCP_HIT_RATE` (Texture Cache Hit Rate, where vL1D is sometimes mapped).
- **L2 Hit Rate:** `TCC_HIT` (Total L2 Cache Hits) / `TCC_REQ` (Total L2 Requests).
- **Bandwidth utilization:** Profile `FETCH_SIZE` and `WRITE_SIZE` to see if your kernel is fully utilizing the 64B/128B cache lines.

Achieving high utilization of the cache hierarchy is critical on CDNA architectures to feed the highly performant matrix cores and avoid HBM bottlenecks.

---
id: technique-double-buffering
title: LDS Double Buffering
type: wiki-technique
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [memory, optimization, pipeline]
confidence: source-reported
techniques: [double-buffering]
hardware_features: [lds, mfma]
kernel_types: [gemm, attention]
related: [hw-lds, technique-mfma-scheduling, technique-ck-tile-programming]
sources: [blog-amdgpu-kernel-opt, doc-mi300x-workload]
reproducibility: snippet
---

# LDS Double Buffering

Ping-pong two LDS tile buffers so global→LDS loads for iteration `k+1` overlap with MFMA compute on iteration `k`.

## Pattern

```cpp
__shared__ float tile_a[2][TILE_M][TILE_K];
int buf = 0;
for (int k = 0; k < K; k += TILE_K) {
    load_global_to_lds(tile_a[buf ^ 1], k + TILE_K);  // prefetch
    mfma_compute(tile_a[buf], ...);                   // compute current
    __syncthreads();
    buf ^= 1;
}
```

## Trade-offs

- Doubles LDS footprint — may reduce occupancy
- CK Tile encodes this in pipeline templates; manual kernels need explicit sync discipline

## Related

- [MFMA Scheduling](mfma-scheduling.md)
- [CK Tile Programming](ck-tile-programming.md)

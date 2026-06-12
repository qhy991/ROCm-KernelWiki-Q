---
id: hw-wavefront
title: Wavefront (64-thread execution unit)
type: wiki-hardware
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [wavefront, hardware, compute]
confidence: verified
hardware_features: [wavefront]
related: [hw-dpp-cross-lane, hw-lds, hw-mfma-matrix-core]
sources: [doc-cdna4-isa, doc-cdna4-whitepaper]
cuda_equivalent: warp
---

# Wavefront

AMD's 64-thread SIMD execution unit — equivalent to NVIDIA's 32-thread warp.

## Implications for Kernel Design

| Topic | CUDA (warp=32) | CDNA (wavefront=64) |
|-------|----------------|---------------------|
| Reduction tree depth | 5 steps | 6 steps |
| LDS bank conflicts | 32-lane patterns | 64-lane patterns |
| Occupancy | blocks × warps | workgroups × wavefronts |
| Shuffle APIs | `__shfl_*_sync` | `__shfl_*` / DPP |

## MFMA Register Layout

MFMA instructions distribute matrix fragments across lanes in a wavefront. Tile configs (e.g. 16×16×16 FP16) assume all 64 lanes participate with a fixed register mapping.

## Related Techniques

- Use [DPP](../hardware/dpp-cross-lane.md) for cross-lane communication
- Pad [LDS](../hardware/lds.md) for 64-thread access patterns

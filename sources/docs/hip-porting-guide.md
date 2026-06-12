---
id: doc-hip-porting-guide
title: CUDA to HIP Porting Guide
type: source-doc
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [migration, cuda, hip-cpp]
date: '2026-01-01'
url: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
source_category: official-doc
confidence: verified
---

# CUDA to HIP Porting Guide

AMD's official porting notes covering API mapping, compiler differences, and common pitfalls when moving CUDA code to HIP.

## High-Impact Differences

| Area | CUDA | HIP / CDNA |
|------|------|------------|
| Wave/warp width | 32 | 64 |
| Matrix API | WMMA | rocWMMA / MFMA |
| Shuffle | `__shfl_*_sync` | `__shfl_*` or DPP asm |
| Profiler | Nsight | rocprof / rocprof-compute |

## Related Wiki Pages

- [CUDA → HIP Migration](../../wiki/migration/cuda-to-hip.md)

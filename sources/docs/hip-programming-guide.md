---
id: doc-hip-programming-guide
title: HIP Programming Guide
type: source-doc
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [hip-cpp, migration, runtime-api]
date: '2026-01-01'
url: https://rocm.docs.amd.com/projects/HIP/en/latest/
source_category: official-doc
languages: [hip-cpp]
confidence: verified
---

# HIP Programming Guide

Official ROCm guide for writing portable GPU applications with HIP: compilation, runtime API, memory model, streams, and device management.

## Kernel Author Checklist

- Build with `hipcc` and target `gfx942` / `gfx950` for MI300/MI350
- Prefer `hipMalloc` + `hipMemcpyAsync` staging patterns for host/device transfers
- Use `hipStream_t` for overlap; avoid excessive `hipDeviceSynchronize`
- Validate occupancy with `hipOccupancyMaxActiveBlocksPerMultiprocessor`

## Related Wiki Pages

- [CUDA → HIP Migration](../../wiki/migration/cuda-to-hip.md)

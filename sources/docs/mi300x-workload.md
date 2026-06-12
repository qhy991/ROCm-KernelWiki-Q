---
id: doc-mi300x-workload
title: MI300X Workload Optimization Guide
type: source-doc
architectures: [cdna3]
tags: [mi300x, optimization, inference, training]
date: '2025-01-01'
url: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
source_category: official-doc
confidence: verified
---

# MI300X Workload Optimization

ROCm guidance for tuning AI workloads on MI300X: memory bandwidth limits, MFMA utilization, kernel fusion opportunities, and framework integration.

## Memory-Bound Triage

1. Measure achieved HBM bandwidth vs peak (~5.3 TB/s)
2. Check LDS bank conflicts on tiled loads
3. Increase vectorized global loads (128-bit)
4. Fuse elementwise ops to reduce memory round-trips

## Related Wiki Pages

- [Memory-Bound AMD Pattern](../../wiki/patterns/memory-bound-amd.md)
- [LDS](../../wiki/hardware/lds.md)

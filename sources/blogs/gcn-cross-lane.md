---
id: blog-gcn-cross-lane
title: AMD GCN Assembly Cross-Lane Operations
type: source-blog
author: AMD GPUOpen
date: '2020-01-01'
url: https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [dpp, cross-lane, wavefront]
hardware_features: [dpp, bpermute]
confidence: source-reported
---

# GCN Cross-Lane Operations

GPUOpen tutorial on DPP (`v_mov_dpp`) and LDS bypass permutes for wavefront-wide data movement without shared memory traffic.

## Key Takeaways

- DPP reuses LDS routing hardware with ~4 cycle latency
- `quad_perm`, `row_shl`, `row_shr`, `wave_ror` cover most shuffle patterns
- `ds_bpermute` enables arbitrary lane mapping when DPP modes are insufficient

## Related Wiki Pages

- [DPP Cross-Lane](../../wiki/hardware/dpp-cross-lane.md)

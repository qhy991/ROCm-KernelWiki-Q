---
id: blog-amdgpu-kernel-opt
title: AMDGPU Kernel Optimization Guide
type: source-blog
author: nod-ai
date: '2025-01-01'
url: https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, memory, occupancy]
techniques: [vectorized-loads, double-buffering, bank-conflict-padding]
confidence: source-reported
---

# AMDGPU Kernel Optimization Guide

Community optimization guide covering memory coalescing, LDS usage, occupancy tuning, and MFMA-friendly layouts on AMD GPUs.

## High-Impact Patterns

- Vectorized 128-bit global loads when alignment allows
- Double-buffered LDS tiles to hide memory latency
- Pad LDS rows to eliminate bank conflicts in GEMM/attention kernels

## Related Wiki Pages

- [Memory-Bound AMD](../../wiki/patterns/memory-bound-amd.md)

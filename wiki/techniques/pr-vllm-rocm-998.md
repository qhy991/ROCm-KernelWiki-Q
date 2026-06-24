---
id: technique-gfx11x-4096b-cache-cliff
title: "Evading the RDNA3 4096B Cache Cliff via Stride Padding"
type: wiki-technique
confidence: verified
architectures:
  - rdna3
kernel_types:
  - gemm
tags:
  - cache-cliff
  - stride-padding
sources:
  - pr-vllm-rocm-998
---

# Evading the RDNA3 4096B Cache Cliff via Stride Padding

## The Cache Aliasing Problem
On AMD's RDNA3 architectures (`gfx11x` / `gfx1151`), the L2 Cache and Memory Attached Last-Level (MALL) cache maps physical addresses using specific stride boundaries. When a matrix row stride is a perfect multiple of **4096 Bytes**, consecutive rows mapped into the cache aggressively collide and evict each other, causing a dramatic drop in effective memory bandwidth.

## The Optimization
This collision can be elegantly sidestepped by introducing **Stride Padding**. 
By appending exactly one cache line worth of dummy data (**128 Bytes**) to the end of each matrix row during the initial weight load phase, the memory layout breaks the 4096B divisible alignment.

- **Storage Cost**: ~3% overhead.
- **Compute Cost**: Zero. The kernel simply iterates using the new leading dimension (`ldb`), ignoring the padding columns.
- **Throughput Gain**: E2E prefill latency (TTFT) improved by over 13.6% in OpenVLA-7B due to restored L2 hit rates.

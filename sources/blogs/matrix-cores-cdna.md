---
id: blog-matrix-cores-cdna
title: Matrix Core Programming on CDNA
type: source-blog
author: AMD ROCm Blogs
date: '2024-01-01'
url: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
architectures: [cdna2, cdna3, cdna4]
tags: [mfma, gemm]
hardware_features: [mfma, dual-cma]
techniques: [mfma-scheduling]
confidence: source-reported
---

# Matrix Core Programming on CDNA

Official ROCm blog series on programming AMD matrix cores: MFMA instruction selection, data layout, and performance considerations on MI200/MI300 class hardware.

## Key Takeaways

- Choose MFMA tile shape to match register budget and occupancy targets
- FP16 `16x16x16` and BF16 `16x16x32` are common GEMM building blocks
- Dual CMA on MI300X increases matrix throughput per CU

## Related Wiki Pages

- [MFMA Matrix Core](../../wiki/hardware/mfma-matrix-core.md)
- [MFMA Scheduling](../../wiki/techniques/mfma-scheduling.md)

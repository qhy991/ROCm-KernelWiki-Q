---
id: doc-cdna4-whitepaper
title: AMD Instinct MI350 Series Architecture Overview
type: source-doc
architectures: [cdna4]
tags: [cdna4, hardware, matrix-core, hbm]
date: '2025-01-01'
url: https://www.amd.com/en/products/accelerators/instinct/mi350.html
source_category: official-doc
hardware_features: [mfma, scaled-mfma, dual-cma, lds-transpose]
confidence: verified
---

# MI350 / CDNA4 Architecture Overview

High-level hardware reference for CDNA4 accelerators: compute unit organization, matrix core throughput, memory hierarchy, and FP8/FP4 support.

## Kernel-Relevant Facts

- Dual CMA (Compute Matrix Accelerator) per CU on CDNA3/CDNA4
- Higher HBM bandwidth than MI300X generation
- Block-scaled low-precision paths for inference/training efficiency
- Wavefront remains 64 threads; LDS is 64 KB per CU

## Related Wiki Pages

- [MFMA Matrix Core](../../wiki/hardware/mfma-matrix-core.md)
- [Wavefront](../../wiki/hardware/wavefront.md)

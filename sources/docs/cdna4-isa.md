---
id: doc-cdna4-isa
title: AMD CDNA4 Instruction Set Architecture Reference
type: source-doc
architectures: [cdna4]
tags: [isa, mfma, cdna4, hardware]
date: '2025-01-01'
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
source_category: official-doc
hardware_features: [mfma, scaled-mfma, lds-transpose, dual-cma]
confidence: verified
---

# AMD CDNA4 ISA Reference

Official ISA guide for MI350X/MI355X (gfx950). Primary reference for MFMA variants, scaled FP8/FP6/FP4 matrix instructions, and CDNA4-specific memory/LDS behavior.

## Key Topics for Kernel Authors

- `v_mfma_*` tile shapes and accumulator layouts
- Scaled MFMA (`v_mfma_scale_*`) for block-scaled low-precision GEMM
- LDS read-with-transpose on CDNA4
- FLAT/DS/VMEM instruction semantics and operand encodings

## Related Wiki Pages

- [MFMA Matrix Core](../../wiki/hardware/mfma-matrix-core.md)
- [LDS](../../wiki/hardware/lds.md)
- [DPP Cross-Lane](../../wiki/hardware/dpp-cross-lane.md)

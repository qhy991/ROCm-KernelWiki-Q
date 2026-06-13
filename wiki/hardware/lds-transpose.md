---
id: hw-lds-transpose
title: LDS Read-with-Transpose (CDNA4)
type: wiki-hardware
architectures: [cdna4]
tags: [lds, lds-transpose, memory, hardware, cdna4]
confidence: source-reported
hardware_features: [lds, lds-transpose]
related: [hw-lds, hw-mfma-matrix-core]
sources: [doc-cdna4-isa, doc-cdna4-whitepaper, pr-rocm_libraries-5813]
cuda_equivalent: null
---

# LDS Read-with-Transpose (CDNA4)

CDNA4 introduces LDS instructions that can **read and transpose data in a single operation**, eliminating the need for separate transpose kernels in GEMM epilogues and attention score computation.

## Overview

| Feature | CDNA3 | CDNA4 |
|---------|-------|-------|
| Standard LDS read | ✓ | ✓ |
| Read-with-transpose | ✗ | ✓ |
| Transpose width | — | Up to 32×32 tiles |
| Impact | Separate transpose kernel needed | Fused into load |

## Problem It Solves

In typical GEMM kernels, the output matrix C is computed in a tiled layout that doesn't match the desired row-major or column-major output. On CDNA3 and earlier, you need a separate LDS transpose step:

```
CDNA3 (two-step):
  1. Write C tile to LDS in compute layout
  2. Read from LDS with transposed indices → transpose
  3. Write to global memory

CDNA4 (one-step):
  1. Write C tile to LDS in compute layout
  2. Read-with-transpose from LDS → directly to VGPR in output layout
  3. Write to global memory
```

This saves one LDS round-trip and reduces register pressure.

## Instruction Format

```asm
; CDNA4 (gfx950): read LDS tile and transpose in hardware.
; The mnemonic family is ds_read_<width>_tr_<elem_bits> — the transpose
; granularity is keyed to the element width being fed to MFMA.
ds_read_b64_tr_b16  v[0:1], v2  offset:0   ; 16-bit elements (fp16/bf16)
ds_read_b128_tr_b16 v[0:3], v4  offset:0   ; wider load, 16-bit elements
ds_read_b64_tr_b8   v[0:1], v2  offset:0   ; 8-bit elements (fp8/bf8)
```

> The repo's own source [`pr-rocm_libraries-5813`](../../sources/prs/hipblaslt/PR-5813.md) confirms `ds_read_tr` transposed LDS loads on gfx950 (used for 32×32×64 MX-GEMM warp tiles). Exact width/element variants (`b4`/`b6`/`b8`/`b16`) should be confirmed against the CDNA4 ISA PDF.

## HIP Usage

```c
// The LDS transpose is typically used implicitly via CK library
// CK handles tile layout and automatically uses LDS transpose on CDNA4

// Manual example (inline asm):
__shared__ float tile[16][16];  // Written in column-major by MFMA

// Read with hardware transpose: column-major → row-major
float4 row = *((float4*)&tile[threadIdx.y][threadIdx.x * 4]);
// On CDNA4, compiler may generate ds_read_*_tr_* instructions
```

## Performance Impact

| Operation | CDNA3 (cycles) | CDNA4 (cycles) | Savings |
|-----------|---------------|----------------|---------|
| GEMM epilogue transpose | ~40-80 (LDS read + shuffle) | ~20-40 (fused read) | 40-50% |
| Attention score transpose | ~60-100 | ~30-50 | 40-50% |
| Register pressure | Higher (need temp VGPRs) | Lower | — |

## Affected Kernel Types

1. **GEMM epilogue**: Write C from MFMA accumulator layout to output layout
2. **Flash Attention**: Transpose attention scores from S = Q×K^T layout
3. **Grouped GEMM**: Variable tile sizes benefit more from fused transpose
4. **Conv im2col**: Channel-last to channel-first layout conversion

## References

- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/)
- [CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/)

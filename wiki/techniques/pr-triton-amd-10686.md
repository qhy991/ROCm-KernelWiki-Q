---
id: technique-tdm-descriptor-sharing
title: "TDM Descriptor SGPR Sharing for Gather/Scatter Loops"
type: wiki-technique
confidence: verified
architectures:
  - cdna4
kernel_types:
  - elementwise
tags:
  - tdm
  - sgpr-optimization
  - salu
sources:
  - pr-triton-amd-10686
---

# TDM Descriptor SGPR Sharing for Gather/Scatter Loops

## Overview
On AMD CDNA4 (`gfx1250`), Tensor Descriptor Memory (TDM) provides hardware-accelerated multidimensional tensor loading. However, generating the descriptor for every instruction chunk in a pipelined unrolled loop causes severe Scalar ALU (SALU) bottlenecks.

## Optimization Strategy
Instead of rebuilding the 4-Dword SGPR descriptor tuple for each row-chunk in a gather/scatter loop, the compiler is modified to hoist the chunk-invariant descriptor fields. 
1. **LDS Offset Calculation**: The LDS destination address is computed fully for chunk 0. For chunks 1..N-1, it is statically proven to be a compile-time constant byte delta.
2. **SGPR Sharing**: The backend emits `s_add_co_i32` to bump only the `lds_addr` lane between chunks, utilizing a single shared SGPR tuple for all `tensor_load_to_lds` instructions.

## Results
In-loop SALU drops significantly (e.g. 35 to 27 instructions), clearing instruction cache bottlenecks and freeing up SGPR resources.

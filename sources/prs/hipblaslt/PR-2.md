---
id: pr-hipblaslt-2
type: source-pr
repo: ROCm/hipBLASLt
pr: 2
title: "[Kernel][Nemotron] SM100 FP8 dense GEMM + ReLU\xB2 fusions and Mamba2/RMSNorm\
  \ fusions for Nemotron-3-Super NVFP4 (B200)"
author: jinhuang12
date: '2022-11-22'
url: https://github.com/ROCm/hipBLASLt/pull/2
source_category: upstream-code
architectures:
- cdna2
- cdna3
- cdna4
tags:
- fp8
- fp4
- rocm
- gemm
- hipblaslt
- layernorm
kernel_types:
- gemm
- layernorm
languages: []
captured_at: '2026-06-12'
status: merged
inclusion_reason: kernel-related changes
confidence: source-reported
---

# [Kernel][Nemotron] SM100 FP8 dense GEMM + ReLU² fusions and Mamba2/RMSNorm fusions for Nemotron-3-Super NVFP4 (B200)

Merged PR #2 in [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt/pull/2).

**Author:** jinhuang12
**Merged:** 2022-11-22

## Description

> Add tensilelite. Changed 100 files including `tensilelite/Tensile/Activation.py`, `tensilelite/Tensile/AsmAddressCalculation.py`, `tensilelite/Tensile/AsmMemoryInstruction.py`, `tensilelite/Tensile/AsmStoreState.py`, `tensilelite/Tensile/BenchmarkProblems.py`, `tensilelite/Tensile/BenchmarkSplitter.py` (+94 more files).

## Changed Files

- `.clang-format` (+124/-0)
- `CMakeLists.txt` (+1/-1)
- `README.md` (+1/-2)
- `tensilelite/LICENSE.md` (+7/-0)
- `tensilelite/MANIFEST.in` (+4/-0)
- `tensilelite/Tensile/Activation.py` (+1040/-0)
- `tensilelite/Tensile/AsmAddressCalculation.py` (+501/-0)
- `tensilelite/Tensile/AsmMemoryInstruction.py` (+63/-0)
- `tensilelite/Tensile/AsmStoreState.py` (+445/-0)
- `tensilelite/Tensile/BenchmarkProblems.py` (+420/-0)
- `tensilelite/Tensile/BenchmarkSplitter.py` (+170/-0)
- `tensilelite/Tensile/BenchmarkStructs.py` (+299/-0)
- `tensilelite/Tensile/ClientExecutable.py` (+83/-0)
- `tensilelite/Tensile/ClientWriter.py` (+1574/-0)
- `tensilelite/Tensile/Common.py` (+1662/-0)
- `tensilelite/Tensile/Component.py` (+260/-0)
- `tensilelite/Tensile/Components/ComputeStoreVgprs.py` (+231/-0)
- `tensilelite/Tensile/Components/GlobalWriteBatch.py` (+1143/-0)
- `tensilelite/Tensile/Components/LocalRead.py` (+226/-0)
- `tensilelite/Tensile/Components/LraTileAssignment.py` (+132/-0)
- ... and 80 more files
See the PR for full details including code changes and review discussion.

## References

- [PR #2](https://github.com/ROCm/hipBLASLt/pull/2)

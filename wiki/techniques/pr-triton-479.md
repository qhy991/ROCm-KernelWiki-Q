---
id: technique-pr-triton-479
title: "Triton Compiler: MFMA Pipeline Layout Cleanup"
type: wiki-technique
architectures: [cdna1, cdna2, cdna3]
tags: [rocm, pipeline, optimization]
hardware_features: [mfma]
kernel_types: [gemm]
languages: [triton-rocm]
confidence: verified
sources: [pr-triton-479]
---

# Triton Compiler: MFMA Pipeline Layout Cleanup

## Overview

PR [#479 in ROCm/triton](https://github.com/ROCm/triton/pull/479) implements a structured cleanup of the `triton_gpu.mfma` layout attribute in the Triton compiler's MLIR dialect. By transitioning the MFMA encoding versioning from a single floating-point value to discrete major and minor version integers, the PR provides a more robust and extensible way to represent AMD CDNA architecture generations within the compiler's intermediate representation.

## Architectural Implications

In Triton's MLIR-based compilation pipeline for AMD GPUs, Matrix Fused Multiply-Add (MFMA) instructions are explicitly represented in the tensor layout format via the `#triton_gpu.mfma` attribute. 

Historically, the version of the MFMA layout was stored as a single floating-point number (e.g., `version = 3.0` for CDNA 3). However, floating-point numbers are generally discouraged for exact versioning identifiers in MLIR dialects due to potential precision quirks and parsing ambiguities.

This update modifies the `AMDMfmaEncodingAttr` print/parse methods in the TritonGPU dialect (`lib/Dialect/TritonGPU/IR/Dialect.cpp`) to emit and ingest:
- `versionMajor` (Integer)
- `versionMinor` (Integer)

This explicit versioning aligns with the architectural delineations of AMD's Matrix Core hardware:
- **CDNA 1** (`gfx908`): Maps to `versionMajor = 1, versionMinor = 0`
- **CDNA 2** (`gfx90a`): Maps to `versionMajor = 2, versionMinor = 0`
- **CDNA 3** (`gfx94x`): Maps to `versionMajor = 3, versionMinor = 0`

## Implementation Details

### MLIR Format Changes

The layout printing format was updated across all MFMA-related integration tests (`accelerate-matmul-cdna1.mlir`, `cdna2.mlir`, `cdna3.mlir`):

**Before:**
```mlir
#triton_gpu.mfma<{version = 3.0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
```

**After:**
```mlir
#triton_gpu.mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
```

### Conversion Pipeline Cleanup

The PR also cleans up inline documentation and comments inside `lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM/SharedToDotOperandMFMA.cpp`. This file is crucial for the compiler lowering phase, as it handles the data layout conversion from shared memory (LDS) into the specific register formats required by the dot operand layout of the underlying MFMA hardware instructions.

## Conclusion

This refactoring ensures that Triton's internal representation of AMD MFMA layouts is cleanly versioned. By using integer-based major and minor versioning, the Triton compiler can more reliably generate and parse MLIR targeted at distinct CDNA generations, reducing technical debt and paving the way for future architectures like CDNA 4.

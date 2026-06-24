---
id: technique-ocp-mx-emulation-mi300
title: "OCP MX Microscaling (W-MXFP4-A-FP8) Emulation on MI300"
type: wiki-technique
confidence: verified
architectures:
  - cdna3
kernel_types:
  - moe
tags:
  - ocp-mx
  - w-mxfp4-a-fp8
  - moe-emulation
sources:
  - pr-vllm-rocm-46142
---

# OCP MX Microscaling (W-MXFP4-A-FP8) Emulation on MI300

## Background on OCP MX
The Open Compute Project (OCP) Microscaling Formats define extremely low-precision tensor types (like MXFP4, MXFP6) which utilize shared block exponents to mitigate the accuracy loss of 4-bit mantissas.
When running these next-generation quantized models on AMD MI300 series hardware, the framework must often emulate the math operations if native 4-bit block-scaled tensor cores are not utilized.

## The W-MXFP4-A-FP8 Emulation Pipeline
In a `W-MXFP4-A-FP8` layout:
1. **Weights (W)** are stored in MXFP4.
2. **Activations (A)** are ingested as FP8.

### Emulation Pitfalls
A critical bug in earlier Triton backend dispatches (`UNFUSED_TRITON`) for Mixture of Experts (MoE) skipped the activation quantization step entirely. This meant the matrix multiply was receiving unscaled inputs, breaking the numerical consistency of the MX emulation.

### Correction Strategy
To ensure bit-accurate emulation:
- The system must explicitly force the MoE emulation code path to inject the `moe_kernel_quantize_input` step for activations.
- Care must be taken not to call quantization *twice* in the kernel stack (e.g., once in `triton_moe.py` and again in `ocp_mx_emulation_moe.py`), which would degrade the FP8 activation into garbage through double-rounding.

This correct emulation dispatch ensures models like `Qwen_1.5-moe-a2.7b-mxfp4` and `gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8` execute flawlessly on CDNA3.

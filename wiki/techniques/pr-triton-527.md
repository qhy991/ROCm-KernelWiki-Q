---
id: technique-pr-triton-527
title: "FP8 AMAX Normalization and Safety Margins in Triton Fused Attention"
type: wiki-technique
architectures: [cdna3, cdna4]
tags:
  - flash-attention
  - triton-rocm
  - quantization
  - fp8
confidence: verified
sources:
  - pr-triton-527
---

# FP8 AMAX Normalization and Safety Margins in Triton Fused Attention

## Overview

When implementing FP8 Fused Attention (Flash Attention) in Triton for ROCm, handling quantization scales and dynamic ranges is critical for both numerical stability and correctness. This page explores an optimization and correctness fix originally identified in **PR #527** in `ROCm/triton`, which rectifies the output Absolute Max (`amax_o`) calculation and introduces an explicit safety margin for FP8 quantization factors.

These considerations are specifically relevant for CDNA3 and CDNA4 architectures where native FP8 (`float8_e4m3fnuz` and `float8_e5m2fnuz`) instruction pipelines are frequently exercised.

## 1. Correct Output AMAX (`amax_o`) Calculation

In FP8 Flash Attention, the kernel must typically output not only the computed attention tensor but also the maximum absolute value (`amax`) of the output tensor so that subsequent layers can dynamically adjust their quantization scales. 

In a typical block-level implementation, `acc` accumulates the attention products. However, due to the online softmax logic in Flash Attention, the accumulated values are unnormalized until the very end.

**The Bug:**
Calculating the max over the accumulator *before* normalizing with the denominator `l_i` captures artificially inflated values.

```python
# INCORRECT: Calculating amax before softmax normalization
if use_fp8:
    tl.atomic_max(amax_s, amax_s_local)
    acc *= acc_descale
    tl.atomic_max(amax_o, tl.max(acc)) # <-- Captures raw unnormalized accumulators

acc = acc / l_i[:, None]
```

**The Fix:**
The `amax_o` atomic calculation is deferred until after `acc` is properly normalized.

```python
# CORRECT: Calculating amax after softmax normalization
if use_fp8:
    tl.atomic_max(amax_s, amax_s_local)
    acc *= acc_descale

acc = acc / l_i[:, None]

if use_fp8:
    tl.atomic_max(amax_o, tl.max(acc)) # <-- Captures the true normalized max
```

> [!WARNING]
> While moving `tl.atomic_max` below the division introduces a data dependency that slightly delays the atomic instruction issue, correct quantization scales are mandatory for FP8. Using pre-normalized scales leads to underutilization of the dynamic range in subsequent layers, essentially zeroing out signals.

## 2. Power-of-Two FP8 Quantization Margins

Determining optimal FP8 scaling factors (`s_scale`, `o_scale`, etc.) requires mapping the tensor's dynamic range into the representable bounds of the format. For instance, AMD's `float8_e4m3fnuz` has a maximum representable value of `240.0`. 

Normally, the scaling factor is calculated as:
$Scale = 2^{\lfloor \log_2(\frac{MaxReprVal}{TensorAMax}) \rfloor}$

However, because floating-point precision naturally experiences small variance during accumulations (especially in block-wise algorithms), scaling precisely to the edge of the maximum representable value often causes unintended overflow to `NaN` (or saturation). 

To prevent this, a 1-bit safety **margin** is introduced to the exponent:

```python
def get_fp8_quantization_factor(
    fp8_max_repr_val: float, 
    tensor: torch.Tensor, 
    fp8_type: torch.dtype=torch.float8_e4m3fnuz,
    margin: int=1
) -> float:
    if fp8_type not in fp8_type_list:
        return 1.0
    ret = fp8_max_repr_val / _tensor_amax(tensor)
    
    # Floor to nearest power of 2, minus a safety margin
    ret = math.pow(2, math.floor(math.log2(ret)) - margin)
    return ret
```

### Why a `margin=1`?
By subtracting `1` from the base-2 exponent, the maximum values of the scaled tensor will only reach $\approx \frac{MaxReprVal}{2}$. 
- **Pros**: It provides a 1-bit safety buffer (a factor of `0.5x`) ensuring that minor deviations from matrix multiplications and reductions do not cross into unrepresentable territories.
- **Cons**: You sacrifice one bit of precision, but for FP8 e4m3 formats, ensuring avoiding catastrophic overflow strongly outperforms the minimal precision loss in attention kernels.

## Takeaways for Kernel Developers
1. **Always Normalize Before Quantizing:** When fusing operations like softmax, ensure your statistical aggregations (`amax`, `amin`) execute after division.
2. **Buffer Your FP8 Scales:** If calculating static or dynamic scaling factors externally in PyTorch before launching a Triton kernel, deliberately pad the quantization factor (by backing off the exponent). Do not quantize right up to the theoretical maximum.
3. **Format Nuances:** Keep in mind that ROCm uses `e4m3fnuz` by default, which lacks dedicated `Infinity` representations. Overflows usually wrap or saturate depending on the instruction flag, making avoiding them at the scaling stage paramount.

---
id: pr-triton-467
title: "Revert Autotuning for Flash Attention due to pre_hook Limitations"
type: source-pr
repo: ROCm/triton
pr: 467
author: vgokhale
date: '2024-01-17'
url: https://github.com/ROCm/triton/pull/467
source_category: upstream-code
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, triton, programming-model]
kernel_types: [flash-attention]
languages: [triton-rocm, python]
captured_at: '2026-06-18'
status: merged
inclusion_reason: "Provides insight into Triton's autotuner pre_hook limitations and parameter dependency handling in AMD flash attention"
confidence: source-reported
---

# Revert Autotuning for Flash Attention due to `pre_hook` Limitations

Merged PR #467 in [ROCm/triton](https://github.com/ROCm/triton/pull/467).

**Author:** vgokhale
**Merged:** 2024-01-17

## Overview

This PR reverts a previous attempt (PR #459) to introduce dynamic autotuning for the Flash Attention kernel in the `perf-kernels` module. The reversion highlights a subtle limitation in Triton's autotuner—specifically, the reliability of modifying kernel arguments dynamically via the `pre_hook` function.

## Problem Statement

When optimizing kernels like Flash Attention, performance relies heavily on selecting the correct block dimensions (`BLOCK_M`, `BLOCK_N`) and execution parameters (like `waves_per_eu` and `num_warps`). 

In PR #459, `@triton.autotune` was added to explore different combinations of these parameters. However, the sequence padding logic (governed by the kernel arguments `need_padding` and `extra_tokens_n`) strictly depends on the size of `BLOCK_N`. Since `BLOCK_N` varies across autotuning configurations, it is impossible to calculate these padding arguments prior to the autotuner selecting a config.

To solve this, the developer used a `pre_hook` function. In Triton, a `pre_hook` can be passed to the autotuner to manipulate the runtime arguments (`nargs`) immediately before the kernel runs. The logic was:
1. Autotuner selects a config (e.g., `BLOCK_N = 64`).
2. `pre_hook` executes, reading `BLOCK_N` and the sequence length.
3. `pre_hook` computes `need_padding` and modifies the `nargs` dictionary.
4. The kernel launches with the updated padding arguments.

### The Bug

The `pre_hook` mechanism in Triton was found to be unreliable for mutating kernel arguments in this specific context. In some execution paths, the changes made to the `nargs` dictionary within the `pre_hook` were not successfully propagated to the kernel. This resulted in the kernel receiving stale, uninitialized, or incorrect padding bounds, leading to execution failures or incorrect outputs.

## Resolution

PR #467 reverts the autotuning approach, removing `pre_hook` entirely. Instead of dynamically searching for the best hardware and tile configurations, the code falls back to static heuristic derivation based on the query dimension (`Lq`).

```python
# Static heuristic selection (fallback logic)
BLOCK_M = 256
BLOCK_N = 128 if Lq == 128 else 64
waves_per_eu = 2 if Lq == 128 else 3
num_warps = 8 if Lq == 128 else 4
pre_load_v = False if Lq == 128 else True
```

Because `BLOCK_N` is now determined statically on the host, `need_padding` and `extra_tokens_n` can be accurately calculated directly in the Python wrapper before kernel dispatch, sidestepping the need for `pre_hook`.

## Architectural Insights

- **Triton API Limitations**: Modifying kernel arguments in `pre_hook` is brittle. Triton's dispatch and compilation caching mechanisms may capture arguments before the hook alters them, or the hook may only mutate a local copy of the dictionary depending on the internal execution flow.
- **Dependent Parameter Resolution**: If kernel parameters (like bounds checks) are fundamentally tied to autotuned constexprs, it is safer to perform the computation inside the device kernel using `tl.constexpr` parameters, rather than attempting to pass them as modified host arguments.
- **AMD Tuning Dimensions**: The tuned configs manipulated `waves_per_eu` (Waves Per Execution Unit). This is an AMD-specific `triton-rocm` parameter that controls wavefront scheduling density and register limits, playing a critical role in occupancy tuning for CDNA architectures.

## References

- [PR #467 - Revert "Add autotuning for FA (#459)"](https://github.com/ROCm/triton/pull/467)
- [PR #459 - Add autotuning for FA](https://github.com/ROCm/triton/pull/459)

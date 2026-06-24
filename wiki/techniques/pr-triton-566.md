---
id: pr-triton-566
title: "Fix varlen mqa test"
type: source-pr
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, triton, flash-attention, mqa, varlen, testing]
confidence: verified
repo: ROCm/triton
pr: 566
author: vgokhale
date: '2024-04-24'
url: https://github.com/ROCm/triton/pull/566
source_category: upstream-code
techniques: []
hardware_features: []
kernel_types: [flash-attention, attention]
languages: [triton-rocm, python]
captured_at: '2026-06-18'
status: merged
inclusion_reason: "Provides insight into the testing infrastructure and API shapes for Variable-Length Multi-Query Attention in ROCm Triton."
---

# Fix varlen mqa test in ROCm Triton

## Executive Summary

PR #566 in the ROCm/triton repository fixes a Python testing harness bug for **Variable-Length Multi-Query Attention (varlen MQA)**. The fix updates the `varlen_input_helper` invocation in `test_op_varlen_mqa_fwd` to correctly pass two context length arguments instead of one. While a one-line change, it highlights the architectural requirement of separating query sequence length from key/value sequence length—a crucial feature for optimizing LLM inference (prefill vs. decode phases).

## Context & Intent

In large language model (LLM) inference, Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) are employed to reduce the memory bandwidth bottleneck of the KV cache. To handle batches of sequences with varying lengths efficiently, **Variable-Length (varlen)** Flash Attention is used. Instead of padding sequences to a maximum length—which wastes compute and memory bandwidth—sequences are packed continuously in memory, and an auxiliary array (often called `cu_seqlens` or `input_metadata`) is used to delineate sequence boundaries.

Prior to this PR, the testing helper `varlen_input_helper` was updated to support asymmetric lengths for Queries versus Keys/Values (i.e., `seqlen_q` vs `seqlen_k`). This separation is necessary because:
- **Prefill Phase:** The context lengths for queries and keys are identical (`seqlen_q == seqlen_k == N_CTX`).
- **Decode Phase:** The query context length is typically 1 (a single token is generated), while the key/value context length contains the entire historical sequence (`seqlen_q = 1`, `seqlen_k = N_CTX`).

The test `test_op_varlen_mqa_fwd` was still passing a single `N_CTX` argument. The author resolved this by explicitly passing `N_CTX` twice (for both query and key/value lengths) to properly model the prefill behavior in the forward test.

## Code Analysis

The fix takes place in the Triton performance kernels testing suite: `python/perf-kernels/flash-attention.py`.

```diff
@@ -1222,7 +1222,7 @@ def test_op_varlen_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
                           ])
 @pytest.mark.parametrize('causal', [False])
 def test_op_varlen_mqa_fwd(Z, HQ, HK, N_CTX, D_HEAD, causal, dtype=torch.float16):
-    q, k, v, input_metadata = varlen_input_helper(Z, HQ, HK, N_CTX, D_HEAD, dtype)
+    q, k, v, input_metadata = varlen_input_helper(Z, HQ, HK, N_CTX, N_CTX, D_HEAD, dtype)
     ref_out = torch.empty_like(q)
     tri_out = torch.empty_like(q)
```

**Variable Definitions:**
- `Z`: Batch size
- `HQ`: Number of Query Heads
- `HK`: Number of Key/Value Heads (in MQA, `HK = 1`; in GQA, `1 < HK < HQ`)
- `N_CTX`: Context length (Sequence Length)
- `D_HEAD`: Dimension of each attention head

By passing `N_CTX, N_CTX`, the test accurately shapes the query array `q` and the key/value arrays `k, v` to have the same sequence length, restoring the test's validity.

## Memory Bounds & Optimization Impact

The introduction of varlen MQA addresses critical bottlenecks in LLM workloads:

1. **Memory Bandwidth Reduction (MQA):** By sharing keys and values across multiple query heads, the size of the KV cache is dramatically reduced. This transforms the attention phase from a severely **memory-bandwidth-bound** operation into one that is more **compute-bound** (or at least more balanced), increasing the overall throughput of GPU execution.
2. **Elimination of Padding (Varlen):** Varlen support prevents the GPU from reading memory and executing MFMA (Matrix Fused Multiply-Add) instructions on padding tokens. This improves hardware utilization (reducing wasted FLOPs and LDS/VGPR footprint) and scales efficiently to extreme batch sizes.
3. **Asymmetric Q/KV Lengths:** Allowing `seqlen_q != seqlen_k` correctly maps to the generation phase. In Triton/ROCm, splitting the generation of `q` and `k,v` metadata ensures the kernel can properly index into high-bandwidth memory without misalignment.

## Conclusion

This PR solidifies the testing harness for ROCm Triton's Flash Attention implementation. By correcting the API usage of the `varlen_input_helper`, it enables continuous integration testing of varlen MQA kernels. These kernels are essential for high-performance, high-occupancy LLM deployment on CDNA architectures.

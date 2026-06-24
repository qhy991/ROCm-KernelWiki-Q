---
id: technique-fused-shared-expert-mapping
title: "Fused Shared Expert (FSE) Slot Mapping for AITER MoE"
type: wiki-technique
confidence: verified
architectures:
  - cdna3
kernel_types:
  - moe
tags:
  - fused-shared-expert
  - fse
  - qwen3.5
  - aiter
sources:
  - pr-vllm-rocm-44434
---

# Fused Shared Expert (FSE) Slot Mapping for AITER MoE

## The Performance Gap
In Mixture of Experts (MoE) architectures like Qwen3.5, the model contains multiple routed experts and one "shared expert" that processes all tokens. Historically, computing the shared expert separately from the routed experts introduces severe kernel launch overhead and breaks memory fusion, leaving up to 25% throughput on the table.

## The Fused Shared Expert (FSE) Strategy
The ROCm AITER backend introduces Fused Shared Experts (FSE) by treating the shared expert simply as an additional routed expert slot (`E+1`). 

### Implementation Challenge
When bridging the Hugging Face checkpoint weights (which treat the shared expert as a completely separate `mlp.shared_expert` tensor) to the FSE backend, the data loader must dynamically remap the parameter names on-the-fly.
By intercepting `shared_expert.*` and remapping it to `experts.{num_experts}.*`, the weights are correctly loaded into the contiguous fused buffer used by AITER's `FusedMoE` kernel.
Additionally, since shared experts often use separate `gate_proj` and `up_proj` whereas the routed experts might be pre-fused, the `is_fused_expert` flag must be explicitly reset for the shared slot.

## Performance
On MI355X with Qwen3.5-397B, mapping the shared expert into the FSE slot improves end-to-end serving throughput by **25.9%** without accuracy degradation.

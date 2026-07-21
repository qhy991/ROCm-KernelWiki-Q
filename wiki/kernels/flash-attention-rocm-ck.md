---
id: kernel-flash-attention-rocm-ck
title: FlashAttention on ROCm via Composable Kernel
type: wiki-kernel
architectures: [cdna3, cdna4]
tags: [attention, flash-attention, flash-attention-3, composable_kernel, ck-tile, fp8, paged-attention, asynchronous-execution, ping-pong-buffering, persistent-kernel]
confidence: source-reported
kernel_types: [attention, flash-attention]
languages: [ck-dsl, hip-cpp, python]
techniques: [ck-tile-programming, double-buffering, persistent-kernel, async-copy]
hardware_features: [mfma, lds]
related:
  - kernel-cdna4-hipblaslt-scaled-mfma-gemm
  - kernel-ck-tile-gemm-rocm
sources:
  - pr-flash-attention-rocm-117
  - pr-rocm-libraries-9214
  - pr-rocm-libraries-8350
  - pr-rocm-libraries-8424
  - pr-rocm-libraries-8262
  - pr-rocm-libraries-8609
  - pr-flash-attention-103
  - pr-rocm-libraries-8492
reproducibility: concept
---

# FlashAttention on ROCm via Composable Kernel

FlashAttention-3 (FA3) on AMD GPUs is **not a port of the CUDA implementation** — it is a
native realization built on the **Composable Kernel (CK Tile)** engine. The defining
thesis of FA3 is radical overlap of memory and compute (asynchrony), and on ROCm that
thesis is expressed through CK Tile's explicit LDS ping-pong buffering, `s_waitcnt` /
`s_wait_dscnt` synchronization, and persistent-kernel dispatch. Understanding ROCm
attention means understanding these three levers, not any single kernel file.

## FA3 Asynchrony: Ping-Pong LDS

The core FA3 mechanism on CDNA3 (`pr-flash-attention-rocm-117`):

1. **Asynchronous global→LDS DMA** for Key/Value blocks — the copy is fully decoupled from
   compute using CK's async-copy path.
2. **Ping-pong buffering**: while the MFMA unit computes QK^T / PV from **LDS Bank 0**,
   the next K/V block streams into **LDS Bank 1**. Synchronization is via `s_waitcnt`
   (memory) and `s_wait_dscnt` (DMA signal count), so the MFMA never stalls waiting for
   the next block.
3. **FP8 block-scaling**: the CK backend is bound to the FP8 tensor-core pathway so QK^T
   and PV execute at CDNA3 peak theoretical throughput.
4. **Persistent-kernel dispatch**: thread blocks stay resident and are fed tiles
   dynamically, bypassing per-wave launch overhead.

The net effect is that the attention kernel is **compute-bound at FP8 peak** rather than
the memory-bound profile of FA1/FA2 — which is the whole point of FA3.

## CK Tile Codegen Model

The FMHA kernels live under `projects/composablekernel/example/ck_tile/01_fmha/` and are
**Python-generated** (`fmha_batch_prefill.py` etc.) into HIP (`fmha_fwd.hpp`). This means
the tile shapes are not hardcoded in C++ — they are codegen parameters, tuned per
(target-GPU, dtype, head-dim) combination. Representative evidence:

- `pr-8350`: adds a tile size for **MI308X / fp8 / hdim=256** batch prefill
- `pr-8492`: adds a tile size for **MI308X / bf16** batch prefill

So "tuning attention for a new shape" on ROCm is usually *adding a codegen tile entry*,
not rewriting a kernel.

## FMHA Kernel Variants

| Variant | Purpose | Evidence |
|---------|---------|----------|
| `mha_batch_prefill` | Batched prefill (long-prompt) | `pr-8350`, `pr-8492`, `pr-9214` |
| paged-KV | Paged KV-cache (vLLM-style block tables) | `pr-9214`, `pr-flash-attention-103` |
| async qr pipeline | Q/R split async pipeline (constraint bk0=bk1) | `pr-8424` |
| BWD | Backward, with graph-capture + stream-async workspace | `pr-8262`, `pr-flash-attention-rocm-112/114/183` |
| split-KV | Long-sequence KV split heuristic | `pr-flash-attention-rocm-147` |
| CK DSL unified | gfx1250 unified attention/moe/topK/RoPE | `pr-8609` |

The paged-KV path (`pr-9214`) is worth noting: it fixes a **32-bit SRD address overflow**
in the ck_tile paged-KV gather path that surfaced at high GPU virtual addresses — a
reminder that paged attention's pointer arithmetic is an active correctness frontier, not
just a perf feature.

## Evidence Map

| Source | What it contributes |
|--------|---------------------|
| `pr-flash-attention-rocm-117` | FA3 asynchrony via CK: ping-pong LDS, `s_waitcnt`/`s_wait_dscnt`, FP8, persistent dispatch |
| `pr-rocm-libraries-9214` | paged-KV gather path; 32-bit VA overflow fix; reveals ck_tile FMHA layout |
| `pr-rocm-libraries-8350` | MI308X fp8 hdim=256 prefill tile (codegen) |
| `pr-rocm-libraries-8492` | MI308X bf16 prefill tile |
| `pr-rocm-libraries-8424` | FMHA qr async pipeline constraint (bk0=bk1) |
| `pr-rocm-libraries-8262` | FMHA backward graph-capture support |
| `pr-rocm-libraries-8609` | CK DSL gfx1250 unified attention/moe/topK/RoPE |
| `pr-flash-attention-103` | Paged attention in mha_varlen_fwd |

## Database Use

Index this page under `operator=attention`, `backend=composable_kernel`, `algorithm=flash-attention-3`,
`precision∈{fp8,bf16}`. Retrieve it for any query about ROCm attention implementation,
FMHA prefill/decode, paged KV-cache kernels, or FA3 asynchrony. It pairs with
`kernel-cdna4-hipblaslt-scaled-mfma-gemm` (FP8 block-scale GEMM) and
`kernel-ck-tile-gemm-rocm` (the CK Tile programming model that attention builds on).

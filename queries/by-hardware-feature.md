# Index: By Hardware Feature


## block-scale (35 pages)

- [Scaled MFMA (CDNA4 Block-Scaled Matrix Operations)](../wiki/hardware/scaled-mfma.md) `[wiki-hardware]` arch:cdna4
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide](../wiki/migration/cdna3-to-cdna4.md) `[wiki-migration]` arch:cdna3, cdna4
- [[Triton] batched_gemm_a16wfp4 (gfx950): fuse dot_scaled accumulator; branchless mxfp4 quant; tune small-N configs](../sources/prs/hipblaslt/PR-3058.md) `[source-pr]` arch:cdna4
- [gfx1201 gemm_a8w8: blockscale HIP→triton fallback + tuning configs (plain + blockscale_preshuffled)](../sources/prs/hipblaslt/PR-3343.md) `[source-pr]` arch:rdna4
- [Add GLM-5.1 FP8 blockscale GEMM/FMoE tunings for gfx942 (MI300X/MI325)](../sources/prs/hipblaslt/PR-3422.md) `[source-pr]` arch:cdna3
- [[FLYDSL MOE] mixed_moe + moe_gemm_2stage: fx internal-types cleanup (ASM-identical)](../sources/prs/hipblaslt/PR-3450.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Fused SplitK zero-init for FP8 a8w8 blockscale GEMMs (y_is_zeroed) + re-enable CKTile SplitK](../sources/prs/hipblaslt/PR-3457.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[PERF] MXFP4 (a4w4) MoE backend for gfx950](../sources/prs/hipblaslt/PR-3470.md) `[source-pr]` arch:cdna4
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md) `[source-pr]` arch:cdna4
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md) `[source-pr]` arch:cdna4
- [AMD - gpt-oss vllm mxfp4: AITER tuning + n-gram spec decode + server …](../sources/prs/hipblaslt/PR-1657.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE](../sources/prs/hipblaslt/PR-330.md) `[source-pr]` arch:cdna4
- [opt(gemm): add AITER MXFP4 preshuffle fast path](../sources/prs/hipblaslt/PR-366.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Add Triton ROCm backend for FP4 GEMM ops (MXFP4 + NVFP4) on MI300X / MI350X](../sources/prs/hipblaslt/PR-381.md) `[source-pr]` arch:cdna4
- [Add A16W4 MoE GEMM stage2 kernel (BF16 activations x MXFP4 weights)](../sources/prs/hipblaslt/PR-431.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Adds Grouped and Batched GEMM kernels with blockscaling matching DeepGEMM API](../sources/prs/hipblaslt/PR-433.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/hipblaslt/PR-605.md) `[source-pr]` arch:rdna4
- [[max/kernels] Fix MXFP4 dequant matmul on MI300X (CDNA3): use FP8 fnuz dtype](../sources/prs/hipblaslt/PR-6474.md) `[source-pr]` arch:cdna3
- [xe: gemm: fixup bdpas scale arg layout](../sources/prs/hipblaslt/PR-5303.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_TILE] Scope NumWarps==8 CompV3 tail/epilogue logic to EightWaves …](../sources/prs/hipblaslt/PR-7669.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [# [TensileLite] Decouple MXFP8 scale DepthU from data DepthU (`ScaleDepthURatio`)](../sources/prs/hipblaslt/PR-7767.md) `[source-pr]` arch:cdna4
- [[Hipblaslt] Allow Subtile path to use BF16 any-K and MX K%32 tail loop](../sources/prs/hipblaslt/PR-7782.md) `[source-pr]` arch:cdna4
- [[CK Tile Engine] Add block-scale GEMM operators: gemm_aquant, gemm_bquant, gemm_abquant](../sources/prs/hipblaslt/PR-8519.md) `[source-pr]` arch:cdna4, rdna4
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md) `[source-pr]` arch:rdna4
- [[TE] Improve backward performance for CK Tile FP8 Group GEMM](../sources/prs/transformerengine/PR-544.md) `[source-pr]` arch:cdna4
- [[CI] Add aiter installation to CI image for MXFP4 FP4 GEMM kernels](../sources/prs/transformerengine/PR-562.md) `[source-pr]` arch:cdna4
- [HipKittens MXFP8 GEMM Support](../sources/prs/transformerengine/PR-566.md) `[source-pr]` arch:cdna4
- [NVFP4: Work around intermittent incorrect results for backward GEMMs](../sources/prs/transformerengine/PR-580.md) `[source-pr]` arch:cdna3, cdna4
- [Fix CK FP8 grouped GEMM dtype gating for columnwise operands](../sources/prs/transformerengine/PR-594.md) `[source-pr]` arch:cdna4
- [Mxfp8 grouped and multi quantize](../sources/prs/transformerengine/PR-598.md) `[source-pr]` arch:cdna4
- [[ROCm][Kernel][AITER] BlockScale FP8 SplitK zero-init fusion](../sources/prs/vllm/PR-44976.md) `[source-pr]` arch:cdna3, cdna4

## bpermute (6 pages)

- [AMD GCN Assembly Cross-Lane Operations](../sources/blogs/gcn-cross-lane.md) `[source-blog]` arch:cdna1, cdna2, cdna3, cdna4
- [DPP — Data-Parallel Primitives (Cross-Lane Operations)](../wiki/hardware/dpp-cross-lane.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [KV Cache Paged Attention on ROCm](../wiki/kernels/kv-cache-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Reduction Tree](../wiki/patterns/reduction-tree.md) `[wiki-pattern]` arch:cdna1, cdna2, cdna3, cdna4
- [Triton Layer Normalization Optimization on AMD CDNA](../wiki/techniques/pr-triton-641.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4

## compute-unit (7 pages)

- [Compute Unit (CU) Microarchitecture](../wiki/hardware/compute-unit.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [XCD Chiplet Architecture](../wiki/hardware/xcd-chiplet.md) `[wiki-hardware]` arch:cdna3
- [Migration Guide: CDNA2 to CDNA3 Architecture](../wiki/migration/cdna2-to-cdna3.md) `[wiki-migration]` arch:cdna2, cdna3
- [Persistent Softmax Optimization in Triton](../wiki/techniques/pr-triton-634.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [SGPR and Scalar Unit Optimization](../wiki/techniques/sgpr-scalar-unit.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [Persistent Loop-Based RMSNorm Kernel (Triton)](../wiki/techniques/pr-triton-676.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Multi-Wavefront Scheduling Strategies](../wiki/techniques/wavefront-scheduling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4

## dpp (11 pages)

- [AMD GCN Assembly Cross-Lane Operations](../sources/blogs/gcn-cross-lane.md) `[source-blog]` arch:cdna1, cdna2, cdna3, cdna4
- [DPP — Data-Parallel Primitives (Cross-Lane Operations)](../wiki/hardware/dpp-cross-lane.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Parallel Prefix Sum (Scan) on ROCm](../wiki/kernels/prefix-sum-scan.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3, cdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [RMSNorm and Normalization Kernels on ROCm](../wiki/kernels/rmsnorm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Reduction Tree](../wiki/patterns/reduction-tree.md) `[wiki-pattern]` arch:cdna1, cdna2, cdna3, cdna4
- [Triton Layer Normalization Optimization on AMD CDNA](../wiki/techniques/pr-triton-641.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Cross-Lane Communication with DPP (Warp Shuffle Equivalent)](../wiki/techniques/warp-shuffle-dpp.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Wavefront Reduction using DPP](../wiki/techniques/wave-reduction.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3

## dual-cma (8 pages)

- [Matrix Core Programming on CDNA](../sources/blogs/matrix-cores-cdna.md) `[source-blog]` arch:cdna2, cdna3, cdna4
- [AMD CDNA4 Instruction Set Architecture Reference](../sources/docs/cdna4-isa.md) `[source-doc]` arch:cdna4
- [AMD Instinct MI350 Series Architecture Overview](../sources/docs/cdna4-whitepaper.md) `[source-doc]` arch:cdna4
- [Dual CMA (Compute Matrix Array) Engines in CDNA4](../wiki/hardware/dual-cma.md) `[wiki-hardware]` arch:cdna3, cdna4
- [MFMA Matrix Core (CDNA1–CDNA4)](../wiki/hardware/mfma-matrix-core.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [XCD Chiplet Architecture](../wiki/hardware/xcd-chiplet.md) `[wiki-hardware]` arch:cdna3
- [Compute-Bound MFMA Pattern on AMD GPUs](../wiki/patterns/compute-bound-mfma-amd.md) `[wiki-pattern]` arch:cdna1, cdna2, cdna3, cdna4
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4

## gws (4 pages)

- [GWS — Global Wave Sync](../wiki/hardware/gws.md) `[wiki-hardware]` arch:cdna2, cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Kernel Launch Overhead Optimization](../wiki/techniques/kernel-launch-overhead.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4

## lds (64 pages)

- [Compute Unit (CU) Microarchitecture](../wiki/hardware/compute-unit.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS — Local Data Share](../wiki/hardware/lds.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Read-with-Transpose (CDNA4)](../wiki/hardware/lds-transpose.md) `[wiki-hardware]` arch:cdna4
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [GEMM Implementation on AMD CDNA](../wiki/kernels/gemm-rocm.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3, rdna3, rdna4
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Efficient Histogram Computation on ROCm](../wiki/kernels/histogram-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [KV Cache Paged Attention on ROCm](../wiki/kernels/kv-cache-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Parallel Prefix Sum (Scan) on ROCm](../wiki/kernels/prefix-sum-scan.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3, cdna4
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md) `[wiki-kernel]` arch:rdna3, rdna4
- [Reduction Kernels on ROCm](../wiki/kernels/reduction-rocm.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3, cdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [RMSNorm and Normalization Kernels on ROCm](../wiki/kernels/rmsnorm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Compute-Bound MFMA Pattern on AMD GPUs](../wiki/patterns/compute-bound-mfma-amd.md) `[wiki-pattern]` arch:cdna1, cdna2, cdna3, cdna4
- [Cooperative Loading](../wiki/patterns/cooperative-loading.md) `[wiki-pattern]` arch:cdna2, cdna3, cdna4
- [Reduction Tree](../wiki/patterns/reduction-tree.md) `[wiki-pattern]` arch:cdna1, cdna2, cdna3, cdna4
- [Wavefront Specialization (Warp Specialization)](../wiki/patterns/warp-specialization.md) `[wiki-pattern]` arch:cdna2, cdna3, cdna4
- [[CK TILE] Fix grouped conv kernels splitk and double lds](../sources/prs/composable_kernel/PR-3527.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK] Add FP8 KV_BLOCKSCALE support for batch prefill](../sources/prs/composable_kernel/PR-3696.md) `[source-pr]` arch:cdna3, cdna4
- [[CK_TILE] Add support and tests for V6 pipeline in conv fwd](../sources/prs/composable_kernel/PR-3708.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md) `[source-pr]` arch:cdna4
- [[CK_TILE] Sparge attention](../sources/prs/composable_kernel/PR-3727.md) `[source-pr]` arch:cdna3, cdna4
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md) `[source-pr]` arch:cdna4
- [[ck_tile/fmha] Fix sink un-mask under right-window and emit fp8bf16 batch_prefill sink kernels](../sources/prs/composable_kernel/PR-3732.md) `[source-pr]` arch:cdna3, cdna4
- [[CK_TILE] fix(fmha): clamp paged KV lookups in batch prefill](../sources/prs/composable_kernel/PR-3733.md) `[source-pr]` arch:cdna3, cdna4
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md) `[source-pr]` arch:cdna4
- [[CK Tile] Prepare mixed batch-prefill FP8 KV contract](../sources/prs/composable_kernel/PR-3745.md) `[source-pr]` arch:cdna3, cdna4
- [Ck tile/flash attention](../sources/prs/flash-attention/PR-61.md) `[source-pr]` arch:cdna2, cdna3
- [Integrate ck tile backward](../sources/prs/flash-attention/PR-65.md) `[source-pr]` arch:cdna2, cdna3
- [Improve FMHA bwd](../sources/prs/flash-attention/PR-70.md) `[source-pr]` arch:cdna2, cdna3
- [Ck tile/kvcache](../sources/prs/flash-attention/PR-74.md) `[source-pr]` arch:cdna2, cdna3
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) `[source-pr]` arch:cdna4
- [# [TensileLite] Decouple MXFP8 scale DepthU from data DepthU (`ScaleDepthURatio`)](../sources/prs/hipblaslt/PR-7767.md) `[source-pr]` arch:cdna4
- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md) `[source-pr]` arch:cdna4, rdna4
- [[CK DSL] conv heuristic: fix gemm_k_per_block, add K_per_C + log features, update all models to 101 features](../sources/prs/hipblaslt/PR-8620.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[AIROCMLIR-798] Add LDS usage estimate CAPI function](../sources/prs/hipblaslt/PR-2400.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[PR 4/7] Multi-arch ROCm kernel support with runtime optimization](../sources/prs/sglang/PR-27745.md) `[source-pr]` arch:cdna3, cdna4
- [[AMD][Perf] Fuse QK RMSNorm + 3D mRoPE + KV-cache store into single aiter op for Qwen3.5-397B-A17B-MXFP4 (TP=2, ROCm/aiter) on HIP](../sources/prs/sglang/PR-28700.md) `[source-pr]` arch:cdna4
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md) `[source-pr]` arch:cdna4
- [[AMD] Optimize o_proj gemm and attn output rope performance](../sources/prs/sglang/PR-28722.md) `[source-pr]` arch:cdna4
- [[ROCm] Faster Custom Paged Attention kernels](../sources/prs/vllm/PR-12348.md) `[source-pr]` arch:cdna3
- [MoE wvSplitK_int4: CU-count grid + skip duplicate MatA to LDS + gfx1151 N=1 K<1024 retune](../sources/prs/vllm/PR-920.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)](../wiki/techniques/async-copy-lds.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [HIP Atomic Operations and Contention Reduction](../wiki/techniques/atomic-operations-hip.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [合并内存访问模式 (Coalesced Memory Access Patterns)](../wiki/techniques/coalesced-memory.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [LDS Double Buffering](../wiki/techniques/double-buffering.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Direct Read](../wiki/techniques/lds-direct-read.md) `[wiki-technique]` arch:cdna3, cdna4
- [Occupancy Tuning on ROCm](../wiki/techniques/occupancy-tuning.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [Utility Tools: Layout Plotting for Triton MLIR on ROCm](../wiki/techniques/pr-triton-635.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Triton Layer Normalization Optimization on AMD CDNA](../wiki/techniques/pr-triton-641.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [ROCm Profiling and Performance Analysis (rocprof, Omniperf)](../wiki/techniques/rocm-profiling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [LDS Address Swizzling](../wiki/techniques/swizzling.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3
- [Vectorized Global Memory Loads](../wiki/techniques/vectorized-loads.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4

## lds-transpose (7 pages)

- [AMD CDNA4 Instruction Set Architecture Reference](../sources/docs/cdna4-isa.md) `[source-doc]` arch:cdna4
- [AMD Instinct MI350 Series Architecture Overview](../sources/docs/cdna4-whitepaper.md) `[source-doc]` arch:cdna4
- [LDS — Local Data Share](../wiki/hardware/lds.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Read-with-Transpose (CDNA4)](../wiki/hardware/lds-transpose.md) `[wiki-hardware]` arch:cdna4
- [CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide](../wiki/migration/cdna3-to-cdna4.md) `[wiki-migration]` arch:cdna3, cdna4
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md) `[source-pr]` arch:cdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) `[source-pr]` arch:cdna4

## mfma (87 pages)

- [ROCm FlashAttention Performance Notes](../sources/blogs/flash-attention-rocm.md) `[source-blog]` arch:cdna2, cdna3, cdna4
- [Matrix Core Programming on CDNA](../sources/blogs/matrix-cores-cdna.md) `[source-blog]` arch:cdna2, cdna3, cdna4
- [AMD CDNA4 Instruction Set Architecture Reference](../sources/docs/cdna4-isa.md) `[source-doc]` arch:cdna4
- [AMD Instinct MI350 Series Architecture Overview](../sources/docs/cdna4-whitepaper.md) `[source-doc]` arch:cdna4
- [ROCm Flash Attention Repository](../sources/docs/flash-attention-rocm.md) `[source-doc]` arch:cdna2, cdna3, cdna4
- [Compute Unit (CU) Microarchitecture](../wiki/hardware/compute-unit.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [Dual CMA (Compute Matrix Array) Engines in CDNA4](../wiki/hardware/dual-cma.md) `[wiki-hardware]` arch:cdna3, cdna4
- [MFMA Matrix Core (CDNA1–CDNA4)](../wiki/hardware/mfma-matrix-core.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [Scaled MFMA (CDNA4 Block-Scaled Matrix Operations)](../wiki/hardware/scaled-mfma.md) `[wiki-hardware]` arch:cdna4
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [GEMM Implementation on AMD CDNA](../wiki/kernels/gemm-rocm.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3, rdna3, rdna4
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [W8A8 Quantized GEMM](../wiki/kernels/quantized-gemm-w8a8.md) `[wiki-kernel]` arch:cdna2, cdna3
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md) `[wiki-kernel]` arch:rdna3, rdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide](../wiki/migration/cdna3-to-cdna4.md) `[wiki-migration]` arch:cdna3, cdna4
- [Triton CUDA to ROCm Migration Guide](../wiki/migration/triton-cuda-to-rocm.md) `[wiki-migration]` arch:cdna2, cdna3, cdna4
- [Compute-Bound Optimization Patterns (算力密集优化模式)](../wiki/patterns/compute-bound-optimization.md) `[wiki-pattern]` arch:cdna2, cdna3, cdna4
- [Compute-Bound MFMA Pattern on AMD GPUs](../wiki/patterns/compute-bound-mfma-amd.md) `[wiki-pattern]` arch:cdna1, cdna2, cdna3, cdna4
- [Add support to fp16 + compute fp16 and bf16 + compute bf16 contractions](../sources/prs/composable_kernel/PR-3598.md) `[source-pr]` arch:rdna3, rdna4
- [Implement device grouped gemm fixed nk multi abd for rdna4](../sources/prs/composable_kernel/PR-3619.md) `[source-pr]` arch:rdna4
- [[CK] Add FP8 KV_BLOCKSCALE support for batch prefill](../sources/prs/composable_kernel/PR-3696.md) `[source-pr]` arch:cdna3, cdna4
- [[CK_TILE] Add support and tests for V6 pipeline in conv fwd](../sources/prs/composable_kernel/PR-3708.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md) `[source-pr]` arch:cdna4
- [[CK_TILE] Sparge attention](../sources/prs/composable_kernel/PR-3727.md) `[source-pr]` arch:cdna3, cdna4
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md) `[source-pr]` arch:cdna4
- [[ck_tile/fmha] Fix sink un-mask under right-window and emit fp8bf16 batch_prefill sink kernels](../sources/prs/composable_kernel/PR-3732.md) `[source-pr]` arch:cdna3, cdna4
- [[CK_TILE] fix(fmha): clamp paged KV lookups in batch prefill](../sources/prs/composable_kernel/PR-3733.md) `[source-pr]` arch:cdna3, cdna4
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md) `[source-pr]` arch:cdna4
- [[CK Tile] Prepare mixed batch-prefill FP8 KV contract](../sources/prs/composable_kernel/PR-3745.md) `[source-pr]` arch:cdna3, cdna4
- [[CK_TILE] Update CK and enable RDNA backward](../sources/prs/flash-attention/PR-184.md) `[source-pr]` arch:cdna3, rdna3, rdna4
- [Ck tile/flash attention](../sources/prs/flash-attention/PR-61.md) `[source-pr]` arch:cdna2, cdna3
- [Integrate ck tile backward](../sources/prs/flash-attention/PR-65.md) `[source-pr]` arch:cdna2, cdna3
- [Improve FMHA bwd](../sources/prs/flash-attention/PR-70.md) `[source-pr]` arch:cdna2, cdna3
- [Ck tile/kvcache](../sources/prs/flash-attention/PR-74.md) `[source-pr]` arch:cdna2, cdna3
- [fix: AMD gfx1201 (RDNA4/ROCm) — INT8 Triton f32 MFMA, LTX Video device fix, validate_settings KeyError](../sources/prs/hipblaslt/PR-1822.md) `[source-pr]` arch:rdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) `[source-pr]` arch:cdna4
- [[CK Tile] Wavelet gemm pipeline for conv fwd](../sources/prs/hipblaslt/PR-7196.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[tensile] gfx12 assembly compatibility](../sources/prs/hipblaslt/PR-7655.md) `[source-pr]` arch:rdna4
- [rocWMMA: add gfx1032 (RDNA2) support with software WMMA fallback](../sources/prs/hipblaslt/PR-8209.md) `[source-pr]` arch:rdna2
- [[hipblaslt][tensilelite] Single-hop next-neighbor StreamK work stealing](../sources/prs/hipblaslt/PR-8442.md) `[source-pr]` arch:cdna4
- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md) `[source-pr]` arch:cdna4, rdna4
- [[origami] Subtile-aware heuristic: reject gfx950 BF16 TN subtile kernels for K<512 with large free dim](../sources/prs/hipblaslt/PR-8604.md) `[source-pr]` arch:cdna4
- [[hipBLASLt] Overlap accum init (initD) with GR across all Subtile paths](../sources/prs/hipblaslt/PR-8615.md) `[source-pr]` arch:cdna4
- [[CK DSL] conv heuristic: fix gemm_k_per_block, add K_per_C + log features, update all models to 101 features](../sources/prs/hipblaslt/PR-8620.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[Fix] compressed-tensors block FP8: requantize weight scales to UE8M0 for DeepGEMM on Blackwell](../sources/prs/sglang/PR-28662.md) `[source-pr]` arch:cdna3, cdna4
- [[RL] MXFP8 flashinfer_trtllm_routed MoE for V4](../sources/prs/sglang/PR-28676.md) `[source-pr]` arch:cdna3, cdna4
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md) `[source-pr]` arch:cdna4
- [[minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra](../sources/prs/sglang/PR-28715.md) `[source-pr]` arch:cdna3, cdna4
- [[AMD] Optimize o_proj gemm and attn output rope performance](../sources/prs/sglang/PR-28722.md) `[source-pr]` arch:cdna4
- [CK Tile MXFP8 Group GEMM gfx1250](../sources/prs/transformerengine/PR-578.md) `[source-pr]` arch:rdna4
- [[AMD] Restrict BlockPingPong scheduling for loop-variant masked loads](../sources/prs/hipblaslt/PR-10585.md) `[source-pr]` arch:cdna4
- [[ROCm] Faster Custom Paged Attention kernels](../sources/prs/vllm/PR-12348.md) `[source-pr]` arch:cdna3
- [[Bugfix][ROCm] Fix OOB query read in paged_attention_rocm for head_size < 128](../sources/prs/vllm/PR-40745.md) `[source-pr]` arch:cdna3, cdna4
- [[ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode](../sources/prs/vllm/PR-40977.md) `[source-pr]` arch:rdna3, rdna4
- [[ROCm][Kernel][AITER] BlockScale FP8 SplitK zero-init fusion](../sources/prs/vllm/PR-44976.md) `[source-pr]` arch:cdna3, cdna4
- [[ROCm][Kernel] Extend skinny gemm N=5 to N=8 cases on GFX12 (RDNA4) using SWMMAC optimization](../sources/prs/vllm/PR-45559.md) `[source-pr]` arch:rdna4
- [[ROCm][Perf] MiniMax-M3 MXFP8 gemm/group gemm dispatch AITER](../sources/prs/vllm/PR-46063.md) `[source-pr]` arch:cdna4
- [[Attention][DSA] support dcp for FLASHINFER_MLA_SPARSE](../sources/prs/vllm/PR-46076.md) `[source-pr]` arch:cdna3, cdna4
- [[ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3](../sources/prs/vllm/PR-46117.md) `[source-pr]` arch:cdna4
- [[ROCm][Perf] Optional FlyDSL BF16 MoE for the MXFP8-emulation path on MiniMax-M3](../sources/prs/vllm/PR-46123.md) `[source-pr]` arch:cdna3
- [[AMD][OCP MX][CI] Fix tests to not dispatch on UNFUSED_TRITON backend on MI300, improve w_mxfp4_a_fp8 emulation support](../sources/prs/vllm/PR-46142.md) `[source-pr]` arch:cdna2, cdna3
- [[ROCm][CI] Only require q_scale==1.0 for fp8 query in RocmAttention](../sources/prs/vllm/PR-46148.md) `[source-pr]` arch:cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [LDS Double Buffering](../wiki/techniques/double-buffering.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Direct Read](../wiki/techniques/lds-direct-read.md) `[wiki-technique]` arch:cdna3, cdna4
- [CDNA4 FP8 Scaled MFMA](../wiki/techniques/mfma-fp8-cdna4.md) `[wiki-technique]` arch:cdna4
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [Mixed Precision Computing in HIP](../wiki/techniques/mixed-precision-hip.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Occupancy Tuning on ROCm](../wiki/techniques/occupancy-tuning.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [PR Insight: triton #457 - [Tuning] Gemm tuning v3](../wiki/techniques/pr-triton-457.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [PR Insight: triton #463 - Refine GEMM test_correctness](../wiki/techniques/pr-triton-463.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Explicit Multiply-Reduce GEMM for Small Block Sizes in Triton](../wiki/techniques/pr-triton-621.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Utility Tools: Layout Plotting for Triton MLIR on ROCm](../wiki/techniques/pr-triton-635.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Triton 8-bit GEMM Scaling Support](../wiki/techniques/pr-triton-677.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Register Tiling for MFMA Kernels](../wiki/techniques/register-tiling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [ROCm Profiling and Performance Analysis (rocprof, Omniperf)](../wiki/techniques/rocm-profiling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)](../wiki/techniques/vgpr-pressure.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [XDLOPS 底层编程 (XDLOPS Low-level Programming)](../wiki/techniques/xdlops-programming.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3

## scaled-mfma (39 pages)

- [AMD CDNA4 Instruction Set Architecture Reference](../sources/docs/cdna4-isa.md) `[source-doc]` arch:cdna4
- [AMD Instinct MI350 Series Architecture Overview](../sources/docs/cdna4-whitepaper.md) `[source-doc]` arch:cdna4
- [Dual CMA (Compute Matrix Array) Engines in CDNA4](../wiki/hardware/dual-cma.md) `[wiki-hardware]` arch:cdna3, cdna4
- [Scaled MFMA (CDNA4 Block-Scaled Matrix Operations)](../wiki/hardware/scaled-mfma.md) `[wiki-hardware]` arch:cdna4
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide](../wiki/migration/cdna3-to-cdna4.md) `[wiki-migration]` arch:cdna3, cdna4
- [[Triton] batched_gemm_a16wfp4 (gfx950): fuse dot_scaled accumulator; branchless mxfp4 quant; tune small-N configs](../sources/prs/hipblaslt/PR-3058.md) `[source-pr]` arch:cdna4
- [[FLYDSL MOE] mixed_moe + moe_gemm_2stage: fx internal-types cleanup (ASM-identical)](../sources/prs/hipblaslt/PR-3450.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[PERF] MXFP4 (a4w4) MoE backend for gfx950](../sources/prs/hipblaslt/PR-3470.md) `[source-pr]` arch:cdna4
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md) `[source-pr]` arch:cdna4
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md) `[source-pr]` arch:cdna4
- [AMD - gpt-oss vllm mxfp4: AITER tuning + n-gram spec decode + server …](../sources/prs/hipblaslt/PR-1657.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE](../sources/prs/hipblaslt/PR-330.md) `[source-pr]` arch:cdna4
- [opt(gemm): add AITER MXFP4 preshuffle fast path](../sources/prs/hipblaslt/PR-366.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Add Triton ROCm backend for FP4 GEMM ops (MXFP4 + NVFP4) on MI300X / MI350X](../sources/prs/hipblaslt/PR-381.md) `[source-pr]` arch:cdna4
- [Add A16W4 MoE GEMM stage2 kernel (BF16 activations x MXFP4 weights)](../sources/prs/hipblaslt/PR-431.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/hipblaslt/PR-605.md) `[source-pr]` arch:rdna4
- [[max/kernels] Fix MXFP4 dequant matmul on MI300X (CDNA3): use FP8 fnuz dtype](../sources/prs/hipblaslt/PR-6474.md) `[source-pr]` arch:cdna3
- [xe: gemm: fixup bdpas scale arg layout](../sources/prs/hipblaslt/PR-5303.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) `[source-pr]` arch:cdna4
- [[CK_TILE] Scope NumWarps==8 CompV3 tail/epilogue logic to EightWaves …](../sources/prs/hipblaslt/PR-7669.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [# [TensileLite] Decouple MXFP8 scale DepthU from data DepthU (`ScaleDepthURatio`)](../sources/prs/hipblaslt/PR-7767.md) `[source-pr]` arch:cdna4
- [TensileLite: Add multi-DU support.](../sources/prs/hipblaslt/PR-7781.md) `[source-pr]` arch:cdna4
- [[Hipblaslt] Allow Subtile path to use BF16 any-K and MX K%32 tail loop](../sources/prs/hipblaslt/PR-7782.md) `[source-pr]` arch:cdna4
- [[CK Tile Engine] Add block-scale GEMM operators: gemm_aquant, gemm_bquant, gemm_abquant](../sources/prs/hipblaslt/PR-8519.md) `[source-pr]` arch:cdna4, rdna4
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md) `[source-pr]` arch:cdna4
- [MXFP4: Add GEMM kernel tuning and MXFP4Quantizer.copy()](../sources/prs/transformerengine/PR-535.md) `[source-pr]` arch:cdna4
- [Full MXFP4 Training Recipe](../sources/prs/transformerengine/PR-537.md) `[source-pr]` arch:cdna4
- [[TE] Improve backward performance for CK Tile FP8 Group GEMM](../sources/prs/transformerengine/PR-544.md) `[source-pr]` arch:cdna4
- [[CI] Add aiter installation to CI image for MXFP4 FP4 GEMM kernels](../sources/prs/transformerengine/PR-562.md) `[source-pr]` arch:cdna4
- [HipKittens MXFP8 GEMM Support](../sources/prs/transformerengine/PR-566.md) `[source-pr]` arch:cdna4
- [NVFP4: Work around intermittent incorrect results for backward GEMMs](../sources/prs/transformerengine/PR-580.md) `[source-pr]` arch:cdna3, cdna4
- [Fix CK FP8 grouped GEMM dtype gating for columnwise operands](../sources/prs/transformerengine/PR-594.md) `[source-pr]` arch:cdna4
- [[ROCm][Perf] MiniMax-M3 MXFP8 gemm/group gemm dispatch AITER](../sources/prs/vllm/PR-46063.md) `[source-pr]` arch:cdna4
- [[ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3](../sources/prs/vllm/PR-46117.md) `[source-pr]` arch:cdna4
- [CDNA4 FP8 Scaled MFMA](../wiki/techniques/mfma-fp8-cdna4.md) `[wiki-technique]` arch:cdna4

## scratch-memory (1 pages)

- [Scratch Memory Spill Management](../wiki/techniques/scratch-memory.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4

## wavefront (44 pages)

- [AMD GCN Assembly Cross-Lane Operations](../sources/blogs/gcn-cross-lane.md) `[source-blog]` arch:cdna1, cdna2, cdna3, cdna4
- [Compute Unit (CU) Microarchitecture](../wiki/hardware/compute-unit.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [Wavefront (64-thread execution unit)](../wiki/hardware/wavefront.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [KV Cache Paged Attention on ROCm](../wiki/kernels/kv-cache-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md) `[wiki-kernel]` arch:rdna3, rdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [RMSNorm and Normalization Kernels on ROCm](../wiki/kernels/rmsnorm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Compute-Bound MFMA Pattern on AMD GPUs](../wiki/patterns/compute-bound-mfma-amd.md) `[wiki-pattern]` arch:cdna1, cdna2, cdna3, cdna4
- [Wavefront Specialization (Warp Specialization)](../wiki/patterns/warp-specialization.md) `[wiki-pattern]` arch:cdna2, cdna3, cdna4
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md) `[source-pr]` arch:cdna4
- [[CK_TILE] Update CK and enable RDNA backward](../sources/prs/flash-attention/PR-184.md) `[source-pr]` arch:cdna3, rdna3, rdna4
- [[hipblaslt][tensilelite] Single-hop next-neighbor StreamK work stealing](../sources/prs/hipblaslt/PR-8442.md) `[source-pr]` arch:cdna4
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md) `[source-pr]` arch:rdna4
- [[hipBLASLt] Overlap accum init (initD) with GR across all Subtile paths](../sources/prs/hipblaslt/PR-8615.md) `[source-pr]` arch:cdna4
- [[PR 4/7] Multi-arch ROCm kernel support with runtime optimization](../sources/prs/sglang/PR-27745.md) `[source-pr]` arch:cdna3, cdna4
- [Fix Qwen MoE precision issue with PP and all-reduce fusion](../sources/prs/sglang/PR-28619.md) `[source-pr]` arch:cdna3, cdna4
- [[feat] add ag_gemm and moe_rs overlap kernels for dsv4 prefill](../sources/prs/sglang/PR-28639.md) `[source-pr]` arch:cdna3, cdna4
- [[AMD] Fuse shared-expert sigmoid + bf16->fp32 cast into the MoE append kernel (3 kernels -> 1)](../sources/prs/sglang/PR-28658.md) `[source-pr]` arch:cdna3, cdna4
- [[AMD][Perf] Fuse QK RMSNorm + 3D mRoPE + KV-cache store into single aiter op for Qwen3.5-397B-A17B-MXFP4 (TP=2, ROCm/aiter) on HIP](../sources/prs/sglang/PR-28700.md) `[source-pr]` arch:cdna4
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md) `[source-pr]` arch:cdna4
- [gfx1250 swizzle_xor changes for FP4](../sources/prs/transformerengine/PR-571.md) `[source-pr]` arch:rdna4
- [[Fix] TE RMSNorm Triton Kernel Optimization](../sources/prs/transformerengine/PR-615.md) `[source-pr]` arch:cdna3, cdna4
- [[ROCm] Faster Custom Paged Attention kernels](../sources/prs/vllm/PR-12348.md) `[source-pr]` arch:cdna3
- [[Bugfix][ROCm] Fix OOB query read in paged_attention_rocm for head_size < 128](../sources/prs/vllm/PR-40745.md) `[source-pr]` arch:cdna3, cdna4
- [[ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode](../sources/prs/vllm/PR-40977.md) `[source-pr]` arch:rdna3, rdna4
- [[ROCm][Kernel] Extend skinny gemm N=5 to N=8 cases on GFX12 (RDNA4) using SWMMAC optimization](../sources/prs/vllm/PR-45559.md) `[source-pr]` arch:rdna4
- [[Attention Backend] add HPC-Ops Attention backend](../sources/prs/vllm/PR-46020.md) `[source-pr]` arch:cdna3, cdna4
- [[ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3](../sources/prs/vllm/PR-46117.md) `[source-pr]` arch:cdna4
- [异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)](../wiki/techniques/async-copy-lds.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [合并内存访问模式 (Coalesced Memory Access Patterns)](../wiki/techniques/coalesced-memory.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Occupancy Tuning on ROCm](../wiki/techniques/occupancy-tuning.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [Explicit Multiply-Reduce GEMM for Small Block Sizes in Triton](../wiki/techniques/pr-triton-621.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [SGPR and Scalar Unit Optimization](../wiki/techniques/sgpr-scalar-unit.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [Cross-Lane Communication with DPP (Warp Shuffle Equivalent)](../wiki/techniques/warp-shuffle-dpp.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Wavefront Reduction using DPP](../wiki/techniques/wave-reduction.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3
- [Multi-Wavefront Scheduling Strategies](../wiki/techniques/wavefront-scheduling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [XDLOPS 底层编程 (XDLOPS Low-level Programming)](../wiki/techniques/xdlops-programming.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3

## wmma (15 pages)

- [[CK] Add support for large tensor index handling into conv bwd data WMMA](../sources/prs/hipblaslt/PR-8518.md) `[source-pr]` arch:rdna3, rdna4
- [[hipblaslt][tensilelite] Add cluster barrier support for subtile gfx1250 kernels](../sources/prs/hipblaslt/PR-8523.md) `[source-pr]` arch:rdna4
- [[hipblaslt][tensilelite] Add multicast tdm for subtile kernel](../sources/prs/hipblaslt/PR-8524.md) `[source-pr]` arch:rdna4
- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md) `[source-pr]` arch:cdna4, rdna4
- [[GFX1250][CK_TILE] Coalesce MX scale16 scale load](../sources/prs/hipblaslt/PR-8566.md) `[source-pr]` arch:rdna4
- [[tensilelite] Fix subtile PGR=0 WMMA-source WAR hazard on gfx1250](../sources/prs/hipblaslt/PR-8603.md) `[source-pr]` arch:rdna4
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md) `[source-pr]` arch:rdna4
- [[hipblaslt][tensilelite] Reorganize and expand coverage of GFX1250 StreamK tests](../sources/prs/hipblaslt/PR-8622.md) `[source-pr]` arch:rdna4
- [add MXFP8 pre-swizzling for gfx1250 GEMM](../sources/prs/transformerengine/PR-568.md) `[source-pr]` arch:rdna4
- [CK Tile Group GEMM gfx1250](../sources/prs/transformerengine/PR-576.md) `[source-pr]` arch:rdna4
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/transformerengine/PR-605.md) `[source-pr]` arch:rdna4
- [CK MXFP8 Group Gemm gfx1250 Enablement](../sources/prs/transformerengine/PR-613.md) `[source-pr]` arch:rdna4
- [gfx1250 mxfp8 gemm: loosen restrictions on K](../sources/prs/transformerengine/PR-627.md) `[source-pr]` arch:rdna4
- [gfx1250 mxfp8 gemm: add NN/NT transpose workaround](../sources/prs/transformerengine/PR-630.md) `[source-pr]` arch:rdna4
- [add dsv4 production mxfp8 gemm shapes](../sources/prs/transformerengine/PR-636.md) `[source-pr]` arch:rdna4
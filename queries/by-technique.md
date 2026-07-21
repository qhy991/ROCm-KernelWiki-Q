# Index: By Technique


## aiter-dispatch (1 pages)

- [Full MXFP4 Training Recipe](../sources/prs/transformerengine/PR-537.md)

## assembly-emission (1 pages)

- [[hipblaslt][tensilelite] Add cluster barrier support for subtile gfx1250 kernels](../sources/prs/hipblaslt/PR-8523.md)

## async-copy (6 pages)

- [FlashAttention on ROCm via Composable Kernel](../wiki/kernels/flash-attention-rocm-ck.md)
- [Cooperative Loading](../wiki/patterns/cooperative-loading.md)
- [生产者-消费者流水线 (Producer-Consumer Pipeline)](../wiki/patterns/producer-consumer-pipeline.md)
- [Wavefront Specialization (Warp Specialization)](../wiki/patterns/warp-specialization.md)
- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md)
- [VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)](../wiki/techniques/vgpr-pressure.md)

## autotuning (1 pages)

- [[CK DSL] conv heuristic: fix gemm_k_per_block, add K_per_C + log features, update all models to 101 features](../sources/prs/hipblaslt/PR-8620.md)

## backend-capability-gating (1 pages)

- [gfx1250 mxfp8 gemm: loosen restrictions on K](../sources/prs/transformerengine/PR-627.md)

## backend-dispatch (1 pages)

- [[Attention Backend] add HPC-Ops Attention backend](../sources/prs/vllm/PR-46020.md)

## backward-kernel (2 pages)

- [Integrate ck tile backward](../sources/prs/flash-attention/PR-65.md)
- [Improve FMHA bwd](../sources/prs/flash-attention/PR-70.md)

## bank-conflict-padding (13 pages)

- [AMDGPU Kernel Optimization Guide](../sources/blogs/amdgpu-kernel-opt.md)
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/hipblaslt/PR-605.md)
- [hipblaslt: fix uninitialized read of a_type/b_type in swizzle validation](../sources/prs/rocm-libraries/PR-8147.md)
- [[Hipblaslt] [Subtile] [gfx1250] Remove Bank conflicts + small scheduling improvements](../sources/prs/rocm-libraries/PR-8211.md)
- [feat: CK Tile unification - swizzle support + gfx950 mixed prec scale + misc](../sources/prs/rocm-libraries/PR-8315.md)
- [[hipBLASlt] Silence spurious 'invalid values of lda' test-client warning](../sources/prs/rocm-libraries/PR-8602.md)
- [add MXFP8 pre-swizzling for gfx1250 GEMM](../sources/prs/transformerengine/PR-568.md)
- [gfx1250 swizzle_xor changes for FP4](../sources/prs/transformerengine/PR-571.md)
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/transformerengine/PR-605.md)
- [[ROCm] Add hipblaslt swizzle gemm kernel](../sources/prs/vllm/PR-830.md)
- [Revert "[ROCm] Add hipblaslt swizzle gemm kernel"](../sources/prs/vllm/PR-837.md)
- [异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)](../wiki/techniques/async-copy-lds.md)
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md)

## bbs (1 pages)

- [Tune gfx1100 BBS GEMM kernels for Llama-3.1-8b-Instruct](../sources/prs/hipblaslt/PR-8631.md)

## block-scale (5 pages)

- [CDNA4 (gfx950/MI350) Scaled-MFMA GEMM in hipBLASLt](../wiki/kernels/cdna4-hipblaslt-scaled-mfma-gemm.md)
- [[CK Tile Engine] Add block-scale GEMM operators: gemm_aquant, gemm_bquant, gemm_abquant](../sources/prs/hipblaslt/PR-8519.md)
- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md)
- [[GFX1250][CK_TILE] Coalesce MX scale16 scale load](../sources/prs/hipblaslt/PR-8566.md)
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md)

## cache-invalidation (1 pages)

- [[RL] MXFP8 flashinfer_trtllm_routed MoE for V4](../sources/prs/sglang/PR-28676.md)

## ck-tile-programming (143 pages)

- [Composable Kernel Tile Tutorial](../sources/blogs/ck-tutorial.md)
- [Composable Kernel Repository Structure](../sources/docs/ck-structure.md)
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md)
- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md)
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md)
- [FlashAttention on ROCm via Composable Kernel](../wiki/kernels/flash-attention-rocm-ck.md)
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md)
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md)
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md)
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md)
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md)
- [生产者-消费者流水线 (Producer-Consumer Pipeline)](../wiki/patterns/producer-consumer-pipeline.md)
- [Tile Quantization and Dequantization](../wiki/patterns/tile-quantize-dequant.md)
- [[CK_Tile] Enable PreshuffleB for 2d block scale Gemm](../sources/prs/composable_kernel/PR-3298.md)
- [[CK_TILE] Port hw independent changes from internal repo to develop branch](../sources/prs/composable_kernel/PR-3301.md)
- [[CK_TILE] Fix for comp pipeline v4](../sources/prs/composable_kernel/PR-3307.md)
- [[CK_TILE] Add indexing optimizations for conv bwd data](../sources/prs/composable_kernel/PR-3309.md)
- [[ck_tile] remove duplicate functions in ck_tile](../sources/prs/composable_kernel/PR-3311.md)
- [[CK_TILE] Fix Quant GEMM build](../sources/prs/composable_kernel/PR-3320.md)
- [[CK_TILE] Add indexing optimizations for conv bwd weight](../sources/prs/composable_kernel/PR-3321.md)
- [[CK_TILE] Generate random tensor values with multiple threads](../sources/prs/composable_kernel/PR-3324.md)
- [[CK_TILE] Disable cast_tile_pk_fp16bf16_fp32 as It Causes Extra spills on Recent Compilers](../sources/prs/composable_kernel/PR-3327.md)
- [[CK_TILE] Fix for Moving DataTypeTraits into a Common File](../sources/prs/composable_kernel/PR-3335.md)
- [[CK_TILE][FMHA] Add sparse attention VSA](../sources/prs/composable_kernel/PR-3341.md)
- [Merge some updates for ck_tile headers](../sources/prs/composable_kernel/PR-3342.md)
- [[CK_TILE] fix enforcing fixed vectorsizes for ck tile conv](../sources/prs/composable_kernel/PR-3344.md)
- [[CK_TILE] Support more layouts for BQuant GEMM](../sources/prs/composable_kernel/PR-3349.md)
- [[CK_TILE] Split-K autodeduction](../sources/prs/composable_kernel/PR-3351.md)
- [[CK_TILE] Add splitk support to ck tile conv bwd data](../sources/prs/composable_kernel/PR-3353.md)
- [[CK_TILE][FMHA] Add logits soft-capping support for FAv3 (WIP)](../sources/prs/composable_kernel/PR-3355.md)
- [[CK-Tile] move out memory operation from cshuffle epilogue class](../sources/prs/composable_kernel/PR-3359.md)
- [[CK_TILE] Stream-K Tree Reduction and Cache Skipping Integration](../sources/prs/composable_kernel/PR-3371.md)
- [[CK_TILE MOE] add NT & preshuffle permute to cktile MOE](../sources/prs/composable_kernel/PR-3377.md)
- [[CK-Tile] fixup codegen for tile engine ops gemm multid and gemm preshuffle](../sources/prs/composable_kernel/PR-3383.md)
- [[CK_TILE] Minor splitk bugfix for gemms and conv](../sources/prs/composable_kernel/PR-3387.md)
- [[CK_TILE][FMHA] Fix Python 3.8 compatibility in fmha codegen](../sources/prs/composable_kernel/PR-3388.md)
- [[CK_TILE] support split-k a16w4 gemm1 ](../sources/prs/composable_kernel/PR-3389.md)
- [[CK_TILE] Fix some inconsistencies with OverrideBDatatype in BQuant GEMM](../sources/prs/composable_kernel/PR-3394.md)
- [[CK_TILE] Add FP8xF4 Flatmm](../sources/prs/composable_kernel/PR-3401.md)
- [[CK_TILE] Grouped gemm quant tensor layouts](../sources/prs/composable_kernel/PR-3414.md)
- [[CK_TILE][FMHA] Add FP8 support for batch_prefill kernel](../sources/prs/composable_kernel/PR-3425.md)
- [[CK-TILE] Guard against compiler lexer diagnostic](../sources/prs/composable_kernel/PR-3444.md)
- [[CK_Tile] Support for various group sizes Preshuffle quant for 2d block scale gemm](../sources/prs/composable_kernel/PR-3445.md)
- [[CK_TILE] MX Flatmm Use Byte Pointer Arithmetic for A Tensor](../sources/prs/composable_kernel/PR-3446.md)
- [[CK_Tile]  Support for group size 128 for Preshuffle quant for 2d block scale gemm](../sources/prs/composable_kernel/PR-3462.md)
- [[CK_TILE] FMHA Ignore BWD Failed Cases in Smoke Test](../sources/prs/composable_kernel/PR-3480.md)
- [[CK_TILE] Align FMHA BWD Reference with Kernel Implementation](../sources/prs/composable_kernel/PR-3486.md)
- [[CK_TILE] MX FLATMM Fix M Padding](../sources/prs/composable_kernel/PR-3489.md)
- [[CK_TILE][FMHA] Enable gpt-oss sink](../sources/prs/composable_kernel/PR-3490.md)
- [[CK_TILE] add preshuffleB mode for ABQuant GEMM](../sources/prs/composable_kernel/PR-3495.md)
- [[CK-Tile] add persistent async input scheduler parameters to kernel device-side and host-side args](../sources/prs/composable_kernel/PR-3520.md)
- [[CK_TILE ENGINE] Fix incorrect List import in reduce_parameter.py](../sources/prs/composable_kernel/PR-3555.md)
- [[CK_TILE] Temporarily disable CK Tile Stream-K reduction tests](../sources/prs/composable_kernel/PR-3559.md)
- [[CK_TILE][FMHA] Add new tile size for async](../sources/prs/composable_kernel/PR-3586.md)
- [[CK_Tile] Support for a4w4 (fp4) in block scale gemm AB quant](../sources/prs/composable_kernel/PR-3603.md)
- [[CK_TILE][FMHA] Revert new tile size for async (#3586)"](../sources/prs/composable_kernel/PR-3613.md)
- [[CK_TILE] Fix Int32 Overflow in Deterministic FMHA BWD](../sources/prs/composable_kernel/PR-3615.md)
- [[CK_TILE][FMHA]Add new tile size for async](../sources/prs/composable_kernel/PR-3623.md)
- [[CK_TILE] Fix alignment in Stream-K workspace buffer](../sources/prs/composable_kernel/PR-3625.md)
- [[CK_Tile] Adding support for preshuffleQuant in AB quant Block Scale Gemm](../sources/prs/composable_kernel/PR-3629.md)
- [[CK_TILE] ABQuant New Preshuffle](../sources/prs/composable_kernel/PR-3638.md)
- [[CK_TILE] Stream-K Tile Engine Test Config File Generation](../sources/prs/composable_kernel/PR-3662.md)
- [[CK_TILE] Fix incompatible vector type arguments for the intrinsic calls](../sources/prs/composable_kernel/PR-3672.md)
- [[CK] Add FP8 KV_BLOCKSCALE support for batch prefill](../sources/prs/composable_kernel/PR-3696.md)
- [[CK_TILE] Add support and tests for V6 pipeline in conv fwd](../sources/prs/composable_kernel/PR-3708.md)
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md)
- [[CK_TILE] Sparge attention](../sources/prs/composable_kernel/PR-3727.md)
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md)
- [[ck_tile/fmha] Fix sink un-mask under right-window and emit fp8bf16 batch_prefill sink kernels](../sources/prs/composable_kernel/PR-3732.md)
- [[CK_TILE] fix(fmha): clamp paged KV lookups in batch prefill](../sources/prs/composable_kernel/PR-3733.md)
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md)
- [[CK Tile] Prepare mixed batch-prefill FP8 KV contract](../sources/prs/composable_kernel/PR-3745.md)
- [[CK_TILE] FAv3 bwd test case & api usage update](../sources/prs/flash-attention/PR-112.md)
- [[CK_TILE] FAv3 bwd minor changes](../sources/prs/flash-attention/PR-113.md)
- [[CK_TILE] Enable FAv3 bwd for head_size=64 dtype=bf16 atomic32](../sources/prs/flash-attention/PR-114.md)
- [[CK_TILE] Use more reasonable splitkv heuristic](../sources/prs/flash-attention/PR-147.md)
- [[CK_TILE] Update CK and add RDNA build support](../sources/prs/flash-attention/PR-178.md)
- [[CK_TILE] Fix NaN for FMHA BWD When seq_q=0](../sources/prs/flash-attention/PR-179.md)
- [[CK_TILE] Update CK and enable RDNA backward](../sources/prs/flash-attention/PR-184.md)
- [Ck tile/flash attention](../sources/prs/flash-attention/PR-61.md)
- [Integrate ck tile backward](../sources/prs/flash-attention/PR-65.md)
- [Improve FMHA bwd](../sources/prs/flash-attention/PR-70.md)
- [Ck tile/kvcache](../sources/prs/flash-attention/PR-74.md)
- [[CK_TILE] Fix fmha fwd splitkv block table read out-of-bound](../sources/prs/flash-attention/PR-98.md)
- [[CK_TILE] Use Unified Workspace for FMHA BWD](../sources/prs/flash-attention/PR-182.md)
- [[CK_TILE] FMHA BWD: stream-async workspace prepare](../sources/prs/flash-attention/PR-183.md)
- [aquant block scale gemm](../sources/prs/rocm-libraries/PR-5268.md)
- [[CK Tile] Add transposed tile load implementation, and tests for load_and_convert_tile](../sources/prs/rocm-libraries/PR-5510.md)
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/rocm-libraries/PR-5813.md)
- [feat(composablekernel): More data type tests for ck tile batched grouped gemm](../sources/prs/rocm-libraries/PR-6521.md)
- [feat: [CK Tile] mxfp8 support for qr async pipeline](../sources/prs/rocm-libraries/PR-6526.md)
- [[CK Tile] Async support pipeline V3](../sources/prs/rocm-libraries/PR-6565.md)
- [[CKTile] Fix MX GEMM: num_loop==3 dispatch, split-K, unsupported-shape guard](../sources/prs/rocm-libraries/PR-6663.md)
- [[CK TILE] Unification Work – Integration of unification framework into CK Tile](../sources/prs/rocm-libraries/PR-7407.md)
- [style: [CK TILE] Unification Work – Unify format MFMA part](../sources/prs/rocm-libraries/PR-7850.md)
- [feat: [CK Tile] Adding gfx1250 wrappers for dense and scale builtins](../sources/prs/rocm-libraries/PR-7852.md)
- [feat: [CK TILE] Unification Work – Add WMMA Scale Mixed Types Support](../sources/prs/rocm-libraries/PR-8020.md)
- [[GFX1250][CK_TILE] Add scale16 (ScaleBlockSize=16) support to MX GEMM TDM pipeline](../sources/prs/rocm-libraries/PR-8202.md)
- [[CK Tile] WAVELET pipeline for backward-data grouped convolution](../sources/prs/rocm-libraries/PR-8220.md)
- [[CK TILE] Unification Work – Remove unification Flag structs in favor of new WarpGemmParams](../sources/prs/rocm-libraries/PR-8227.md)
- [ [CK_TILE] Add graph capture support for FMHA backward(new branch)](../sources/prs/rocm-libraries/PR-8262.md)
- [[CK][CK Tile] Grouped Conv GFX1250 fixes for dispatcher and builder g…](../sources/prs/rocm-libraries/PR-8271.md)
- [feat: CK Tile unification - swizzle support + gfx950 mixed prec scale + misc](../sources/prs/rocm-libraries/PR-8315.md)
- [[GFX1250][MX GEMM] Unified FLATMM GroupedGemm Implementation for MX Data Types](../sources/prs/rocm-libraries/PR-8325.md)
- [Add tile shape for FMHA batch prefill on MI308X (on fp8, hdim=256)](../sources/prs/rocm-libraries/PR-8350.md)
- [feat(ck): Extend and optimize Quant Gemm Kernel for Aiter a8w8](../sources/prs/rocm-libraries/PR-8423.md)
- [Add missing constraint in the FMHA qr async pipeline to enforce bk0=bk1 ](../sources/prs/rocm-libraries/PR-8424.md)
- [Skip tests on gfx11 that have intermittent failures](../sources/prs/rocm-libraries/PR-8487.md)
- [Add tile size for FMHA batch prefill bf16 for MI308X](../sources/prs/rocm-libraries/PR-8492.md)
- [feat(ck-tile): add block-scale GEMM operators (aquant, bquant, abquant)](../sources/prs/rocm-libraries/PR-8519.md)
- [[CK_TILE] Use launched block size for GEMM occupancy query](../sources/prs/rocm-libraries/PR-8531.md)
- [[CK Tile] EightWaves pipeline int8 support](../sources/prs/rocm-libraries/PR-8535.md)
- [refactor(ck): mx gemm kernel unification](../sources/prs/rocm-libraries/PR-8554.md)
- [[CK][CK Tile] Drop profiler for experimental builder codegen](../sources/prs/rocm-libraries/PR-8573.md)
- [fix(ck): Clean up Stream-K remnants in old CK and fix static_assert in CK Tile](../sources/prs/rocm-libraries/PR-8595.md)
- [[CK] Fix compilation](../sources/prs/rocm-libraries/PR-8637.md)
- [fix(ck-tile): add missing fp8/bf8 warp_gemm dispatcher entries for gfx950](../sources/prs/rocm-libraries/PR-8799.md)
- [feat(ck-tile): add stream_k variant to GEMM Dispatcher codegen](../sources/prs/rocm-libraries/PR-8985.md)
- [feat(ck-tile): TE to dispatcher GEMM bridge (fp16/bf16, all layouts)](../sources/prs/rocm-libraries/PR-8997.md)
- [feat(ck-tile): TE to dispatcher GEMM bridge for fp8/bf8/int8 (all layouts)](../sources/prs/rocm-libraries/PR-8998.md)
- [feat(ck-tile): add grouped GEMM variant to TE to dispatcher bridge](../sources/prs/rocm-libraries/PR-9000.md)
- [feat(ck-tile): stream-K GEMM TE to dispatcher bridge](../sources/prs/rocm-libraries/PR-9028.md)
- [fix: FMHA batch-prefill paged-KV 32-bit VA overflow at high GPU base addresses](../sources/prs/rocm-libraries/PR-9214.md)
- [fix(ck-tile): Fix compiler issue](../sources/prs/rocm-libraries/PR-9359.md)
- [Revert PR #6526 "feat: [CK Tile] mxfp8 support for qr async pipeline (#6526)"](../sources/prs/rocm-libraries/PR-9461.md)
- [fix(ck_tile): CK CI Aiter test error](../sources/prs/rocm-libraries/PR-9473.md)
- [fix(ck-tile): close GEMM_GROUPED entry in arch_filter OPERATOR_TILE_CONSTRAINTS](../sources/prs/rocm-libraries/PR-9520.md)
- [fix(CK_TILE): fix error found by Aiter repo tests](../sources/prs/rocm-libraries/PR-9562.md)
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md)
- [[CK_TILE] Scope NumWarps==8 CompV3 tail/epilogue logic to EightWaves …](../sources/prs/hipblaslt/PR-7669.md)
- [[CK] feat(ssd): add fp16/bf16 support with fp32 accumulation](../sources/prs/hipblaslt/PR-7851.md)
- [[CK_TILE] Add Tile Engine -> Dispatcher bridge for GEMM](../sources/prs/hipblaslt/PR-8123.md)
- [[CK_TILE] Use launched block size for GEMM occupancy query](../sources/prs/hipblaslt/PR-8531.md)
- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md)
- [[GFX1250][CK_TILE] Coalesce MX scale16 scale load](../sources/prs/hipblaslt/PR-8566.md)
- [[TE] Improve backward performance for CK Tile FP8 Group GEMM](../sources/prs/transformerengine/PR-544.md)
- [CK Tile Group GEMM gfx1250](../sources/prs/transformerengine/PR-576.md)
- [CK Tile MXFP8 Group GEMM gfx1250](../sources/prs/transformerengine/PR-578.md)
- [CK MXFP8 Group Gemm gfx1250 Enablement](../sources/prs/transformerengine/PR-613.md)
- [support ck-tile blockquant gemm in vllm](../sources/prs/vllm/PR-642.md)
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md)
- [LDS Direct Read](../wiki/techniques/lds-direct-read.md)
- [CDNA4 FP8 Scaled MFMA](../wiki/techniques/mfma-fp8-cdna4.md)

## cluster-barrier (1 pages)

- [[hipblaslt][tensilelite] Add cluster barrier support for subtile gfx1250 kernels](../sources/prs/hipblaslt/PR-8523.md)

## coalesced-load (1 pages)

- [[GFX1250][CK_TILE] Coalesce MX scale16 scale load](../sources/prs/hipblaslt/PR-8566.md)

## code-generation (1 pages)

- [[CK_TILE] Sparge attention](../sources/prs/composable_kernel/PR-3727.md)

## codegen (1 pages)

- [[tensilelite] Fix rocisa instruction mnemonics and add gfx12+ scalar ops](../sources/prs/hipblaslt/PR-8586.md)

## collective-fusion (1 pages)

- [Fix Qwen MoE precision issue with PP and all-reduce fusion](../sources/prs/sglang/PR-28619.md)

## communication-computation-overlap (1 pages)

- [[feat] add ag_gemm and moe_rs overlap kernels for dsv4 prefill](../sources/prs/sglang/PR-28639.md)

## convolution (1 pages)

- [[CK_TILE] Add support and tests for V6 pipeline in conv fwd](../sources/prs/composable_kernel/PR-3708.md)

## correctness-guard (1 pages)

- [[CK][CK DSL] Pass vector sizes as arguments for implicit gemm](../sources/prs/hipblaslt/PR-8624.md)

## csv-tuning (1 pages)

- [MXFP4: Add GEMM kernel tuning and MXFP4Quantizer.copy()](../sources/prs/transformerengine/PR-535.md)

## double-buffering (35 pages)

- [AMDGPU Kernel Optimization Guide](../sources/blogs/amdgpu-kernel-opt.md)
- [CDNA4 (gfx950/MI350) Scaled-MFMA GEMM in hipBLASLt](../wiki/kernels/cdna4-hipblaslt-scaled-mfma-gemm.md)
- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md)
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md)
- [FlashAttention on ROCm via Composable Kernel](../wiki/kernels/flash-attention-rocm-ck.md)
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md)
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md)
- [Compute-Bound Optimization Patterns (算力密集优化模式)](../wiki/patterns/compute-bound-optimization.md)
- [Compute-Bound MFMA Pattern on AMD GPUs](../wiki/patterns/compute-bound-mfma-amd.md)
- [Latency Hiding (延迟隐藏)](../wiki/patterns/latency-hiding.md)
- [生产者-消费者流水线 (Producer-Consumer Pipeline)](../wiki/patterns/producer-consumer-pipeline.md)
- [Wavefront Specialization (Warp Specialization)](../wiki/patterns/warp-specialization.md)
- [[CK TILE] Fix basic gemm pipelines add v1 interwave pipeline](../sources/prs/composable_kernel/PR-3611.md)
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md)
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md)
- [[CK_TILE] fix(fmha): clamp paged KV lookups in batch prefill](../sources/prs/composable_kernel/PR-3733.md)
- [[hipblaslt][tensilelite] Add PrefetchAcrossPersistent](../sources/prs/rocm-libraries/PR-7053.md)
- [[tensilelite] Use 32-bit GL2PrefetchInc to reduce sgpr pressure](../sources/prs/rocm-libraries/PR-8303.md)
- [[tensilelite ] fix gfx1250 gl2_prefetch unit test](../sources/prs/rocm-libraries/PR-8803.md)
- [perf(hipblaslt) Add prefetch gl2 support for subtile kernel](../sources/prs/rocm-libraries/PR-9161.md)
- [fix(tensilelite): Fix PAP+TDM redundant descriptor rebuild](../sources/prs/rocm-libraries/PR-9196.md)
- [feat(tensilelite): prefetchGL2 generalization](../sources/prs/rocm-libraries/PR-9326.md)
- [fix(tensilelite): do not bump numSgprStreamK for SkPrefetchPrimed](../sources/prs/rocm-libraries/PR-9486.md)
- [Revert "feat(tensilelite): prefetchGL2 generalization"](../sources/prs/rocm-libraries/PR-9501.md)
- [feat(tensilelite): reapply prefetchGL2 generalization](../sources/prs/rocm-libraries/PR-9534.md)
- [fix (Tensilelite) (StinkyTofu) support-106sgpr-gfx1250](../sources/prs/rocm-libraries/PR-9538.md)
- [[CK Tile] Wavelet gemm pipeline for conv fwd](../sources/prs/hipblaslt/PR-7196.md)
- [Remove SGPR for SwInstructionPrefetchRelStaticPass](../sources/prs/hipblaslt/PR-8340.md)
- [[hipBLASLt] Overlap accum init (initD) with GR across all Subtile paths](../sources/prs/hipblaslt/PR-8615.md)
- [异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)](../wiki/techniques/async-copy-lds.md)
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md)
- [LDS Double Buffering](../wiki/techniques/double-buffering.md)
- [Occupancy Tuning on ROCm](../wiki/techniques/occupancy-tuning.md)
- [VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)](../wiki/techniques/vgpr-pressure.md)
- [Multi-Wavefront Scheduling Strategies](../wiki/techniques/wavefront-scheduling.md)

## ds-swizzle (1 pages)

- [gfx1250 swizzle_xor changes for FP4](../sources/prs/transformerengine/PR-571.md)

## fp8-quantization (1 pages)

- [[feat] add ag_gemm and moe_rs overlap kernels for dsv4 prefill](../sources/prs/sglang/PR-28639.md)

## fused-cast-transpose (1 pages)

- [Full MXFP4 Training Recipe](../sources/prs/transformerengine/PR-537.md)

## gemm-pipeline (1 pages)

- [[CK_TILE] Add support and tests for V6 pipeline in conv fwd](../sources/prs/composable_kernel/PR-3708.md)

## global-read-emission (1 pages)

- [[hipblaslt][tensilelite] Add multicast tdm for subtile kernel](../sources/prs/hipblaslt/PR-8524.md)

## grouped-convolution (1 pages)

- [[CK] Add support for large tensor index handling into conv bwd data WMMA](../sources/prs/hipblaslt/PR-8518.md)

## grouped-gemm (2 pages)

- [[CK_TILE] Use launched block size for GEMM occupancy query](../sources/prs/hipblaslt/PR-8531.md)
- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md)

## hadamard-transform (1 pages)

- [Full MXFP4 Training Recipe](../sources/prs/transformerengine/PR-537.md)

## hardware-bounds-checking (1 pages)

- [Flat vs Buffer Addressing Modes](../wiki/techniques/flat-addressing.md)

## hardware-modeling (1 pages)

- [[hipblaslt][origami] Model changes for mi350P](../sources/prs/hipblaslt/PR-8600.md)

## hazard-avoidance (1 pages)

- [[tensilelite] Fix subtile PGR=0 WMMA-source WAR hazard on gfx1250](../sources/prs/hipblaslt/PR-8603.md)

## host-problem-construction (1 pages)

- [[hipBLASLt] Fix int8 GEMM crash on alpha=1065353216](../sources/prs/hipblaslt/PR-8579.md)

## index-localization (1 pages)

- [[Attention][DSA] support dcp for FLASHINFER_MLA_SPARSE](../sources/prs/vllm/PR-46076.md)

## instruction-compatibility (1 pages)

- [gfx1250 swizzle_xor changes for FP4](../sources/prs/transformerengine/PR-571.md)

## instruction-selection (1 pages)

- [[tensilelite] Fix rocisa instruction mnemonics and add gfx12+ scalar ops](../sources/prs/hipblaslt/PR-8586.md)

## jit-compilation (1 pages)

- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md)

## kernel-fusion (5 pages)

- [[AMD] Fuse shared-expert sigmoid + bf16->fp32 cast into the MoE append kernel (3 kernels -> 1)](../sources/prs/sglang/PR-28658.md)
- [[AMD][Perf] Fuse QK RMSNorm + 3D mRoPE + KV-cache store into single aiter op for Qwen3.5-397B-A17B-MXFP4 (TP=2, ROCm/aiter) on HIP](../sources/prs/sglang/PR-28700.md)
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md)
- [[AMD] Optimize o_proj gemm and attn output rope performance](../sources/prs/sglang/PR-28722.md)
- [[Attention Backend] add HPC-Ops Attention backend](../sources/prs/vllm/PR-46020.md)

## kv-cache (1 pages)

- [Ck tile/kvcache](../sources/prs/flash-attention/PR-74.md)

## large-indexing (1 pages)

- [[CK] Add support for large tensor index handling into conv bwd data WMMA](../sources/prs/hipblaslt/PR-8518.md)

## launch-configuration (1 pages)

- [[CK_TILE] Use launched block size for GEMM occupancy query](../sources/prs/hipblaslt/PR-8531.md)

## layout-transform (6 pages)

- [[GFX1250][CK_TILE] Coalesce MX scale16 scale load](../sources/prs/hipblaslt/PR-8566.md)
- [[RL] MXFP8 flashinfer_trtllm_routed MoE for V4](../sources/prs/sglang/PR-28676.md)
- [[AMD] Optimize o_proj gemm and attn output rope performance](../sources/prs/sglang/PR-28722.md)
- [[TE] Improve backward performance for CK Tile FP8 Group GEMM](../sources/prs/transformerengine/PR-544.md)
- [add MXFP8 pre-swizzling for gfx1250 GEMM](../sources/prs/transformerengine/PR-568.md)
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/transformerengine/PR-605.md)

## llm-inference (1 pages)

- [Tune gfx1100 BBS GEMM kernels for Llama-3.1-8b-Instruct](../sources/prs/hipblaslt/PR-8631.md)

## logical-scheduling (1 pages)

- [[tensilelite] Fix subtile PGR=0 WMMA-source WAR hazard on gfx1250](../sources/prs/hipblaslt/PR-8603.md)

## manifest-runner (1 pages)

- [[CK][CK DSL] Pass vector sizes as arguments for implicit gemm](../sources/prs/hipblaslt/PR-8624.md)

## masking (1 pages)

- [[ck_tile/fmha] Fix sink un-mask under right-window and emit fp8bf16 batch_prefill sink kernels](../sources/prs/composable_kernel/PR-3732.md)

## memory-mapping (1 pages)

- [RCCL Multi-GPU Communication](../wiki/techniques/multi-gpu-rccl.md)

## mfma-scheduling (40 pages)

- [Matrix Core Programming on CDNA](../sources/blogs/matrix-cores-cdna.md)
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md)
- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md)
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md)
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md)
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md)
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md)
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md)
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md)
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md)
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md)
- [Compute-Bound Optimization Patterns (算力密集优化模式)](../wiki/patterns/compute-bound-optimization.md)
- [Compute-Bound MFMA Pattern on AMD GPUs](../wiki/patterns/compute-bound-mfma-amd.md)
- [Latency Hiding (延迟隐藏)](../wiki/patterns/latency-hiding.md)
- [生产者-消费者流水线 (Producer-Consumer Pipeline)](../wiki/patterns/producer-consumer-pipeline.md)
- [Wavefront Specialization (Warp Specialization)](../wiki/patterns/warp-specialization.md)
- [[CK-Tile] add persistent async input scheduler parameters to kernel device-side and host-side args](../sources/prs/composable_kernel/PR-3520.md)
- [Add support to fp16 + compute fp16 and bf16 + compute bf16 contractions](../sources/prs/composable_kernel/PR-3598.md)
- [[CK TILE] Fix basic gemm pipelines add v1 interwave pipeline](../sources/prs/composable_kernel/PR-3611.md)
- [Implement device grouped gemm fixed nk multi abd for rdna4](../sources/prs/composable_kernel/PR-3619.md)
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md)
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md)
- [refactor(tensilelite): add stage-tagged TypeAlias for LogicalScheduler pipeline](../sources/prs/rocm-libraries/PR-7540.md)
- [[CK DSL] gfx950 GEMM: arch-resolve compv4 schedule hints (~+2%)](../sources/prs/rocm-libraries/PR-8320.md)
- [[stinkytofu] Add tests for tensor_load_to_lds in-flight throttle](../sources/prs/rocm-libraries/PR-8416.md)
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/rocm-libraries/PR-8609.md)
- [refactor(stinkytofu): consolidate CDNA5 scheduling tunables into named constants](../sources/prs/rocm-libraries/PR-8990.md)
- [fix(stinkytofu): represent exec-masked spans as atomic DAG nodes](../sources/prs/rocm-libraries/PR-9004.md)
- [feat(stinkytofu): expose CDNA5 scheduler tuning knobs via ModuleOptions](../sources/prs/rocm-libraries/PR-9103.md)
- [[origami] Subtile-aware heuristic: reject gfx950 BF16 TN subtile kernels for K<512 with large free dim](../sources/prs/hipblaslt/PR-8604.md)
- [[hipBLASLt] Overlap accum init (initD) with GR across all Subtile paths](../sources/prs/hipblaslt/PR-8615.md)
- [CK Tile MXFP8 Group GEMM gfx1250](../sources/prs/transformerengine/PR-578.md)
- [[ROCm] Faster Custom Paged Attention kernels](../sources/prs/vllm/PR-12348.md)
- [[Bugfix][ROCm] Fix OOB query read in paged_attention_rocm for head_size < 128](../sources/prs/vllm/PR-40745.md)
- [[ROCm][Kernel] Extend skinny gemm N=5 to N=8 cases on GFX12 (RDNA4) using SWMMAC optimization](../sources/prs/vllm/PR-45559.md)
- [LDS Direct Read](../wiki/techniques/lds-direct-read.md)
- [CDNA4 FP8 Scaled MFMA](../wiki/techniques/mfma-fp8-cdna4.md)
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md)
- [Multi-Wavefront Scheduling Strategies](../wiki/techniques/wavefront-scheduling.md)
- [XDLOPS 底层编程 (XDLOPS Low-level Programming)](../wiki/techniques/xdlops-programming.md)

## model-selection (1 pages)

- [[CK DSL] conv heuristic: fix gemm_k_per_block, add K_per_C + log features, update all models to 101 features](../sources/prs/hipblaslt/PR-8620.md)

## multicast (1 pages)

- [[hipblaslt][tensilelite] Add multicast tdm for subtile kernel](../sources/prs/hipblaslt/PR-8524.md)

## occupancy-query (1 pages)

- [[CK_TILE] Use launched block size for GEMM occupancy query](../sources/prs/hipblaslt/PR-8531.md)

## occupancy-tuning (30 pages)

- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md)
- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md)
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md)
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md)
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md)
- [KV Cache Paged Attention on ROCm](../wiki/kernels/kv-cache-rocm.md)
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md)
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md)
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md)
- [RMSNorm and Normalization Kernels on ROCm](../wiki/kernels/rmsnorm-rocm.md)
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md)
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md)
- [Compute-Bound MFMA Pattern on AMD GPUs](../wiki/patterns/compute-bound-mfma-amd.md)
- [Latency Hiding (延迟隐藏)](../wiki/patterns/latency-hiding.md)
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md)
- [[hipblaslt][tensilelite] Single-hop next-neighbor StreamK work stealing](../sources/prs/hipblaslt/PR-8442.md)
- [[origami] Subtile-aware heuristic: reject gfx950 BF16 TN subtile kernels for K<512 with large free dim](../sources/prs/hipblaslt/PR-8604.md)
- [[hipBLASLt] Overlap accum init (initD) with GR across all Subtile paths](../sources/prs/hipblaslt/PR-8615.md)
- [[PR 4/7] Multi-arch ROCm kernel support with runtime optimization](../sources/prs/sglang/PR-27745.md)
- [Mxfp8 grouped and multi quantize](../sources/prs/transformerengine/PR-598.md)
- [[Fix] TE RMSNorm Triton Kernel Optimization](../sources/prs/transformerengine/PR-615.md)
- [[ROCm] Faster Custom Paged Attention kernels](../sources/prs/vllm/PR-12348.md)
- [[ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode](../sources/prs/vllm/PR-40977.md)
- [[ROCm][Kernel][AITER] BlockScale FP8 SplitK zero-init fusion](../sources/prs/vllm/PR-44976.md)
- [[ROCm][Perf] MiniMax-M3 MXFP8 gemm/group gemm dispatch AITER](../sources/prs/vllm/PR-46063.md)
- [[ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3](../sources/prs/vllm/PR-46117.md)
- [Occupancy Tuning on ROCm](../wiki/techniques/occupancy-tuning.md)
- [Persistent Softmax Optimization in Triton](../wiki/techniques/pr-triton-634.md)
- [SGPR and Scalar Unit Optimization](../wiki/techniques/sgpr-scalar-unit.md)
- [Multi-Wavefront Scheduling Strategies](../wiki/techniques/wavefront-scheduling.md)

## operator-builder (1 pages)

- [[CK Tile Engine] Add block-scale GEMM operators: gemm_aquant, gemm_bquant, gemm_abquant](../sources/prs/hipblaslt/PR-8519.md)

## overlap-compute-transfer (1 pages)

- [RCCL Multi-GPU Communication](../wiki/techniques/multi-gpu-rccl.md)

## paged-attention (1 pages)

- [[CK] Add FP8 KV_BLOCKSCALE support for batch prefill](../sources/prs/composable_kernel/PR-3696.md)

## persistent-kernel (27 pages)

- [FlashAttention on ROCm via Composable Kernel](../wiki/kernels/flash-attention-rocm-ck.md)
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md)
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md)
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md)
- [Grid-Stride Loop](../wiki/patterns/grid-stride-loop.md)
- [[CK Tile] Grouped GEMM aquant mode and non-persistent kernel](../sources/prs/composable_kernel/PR-3337.md)
- [[CK-Tile] add persistent async input scheduler parameters to kernel device-side and host-side args](../sources/prs/composable_kernel/PR-3520.md)
- [[CK Tile] Async support pipeline V3](../sources/prs/rocm-libraries/PR-6565.md)
- [[hipblaslt][tensilelite] Add PrefetchAcrossPersistent](../sources/prs/rocm-libraries/PR-7053.md)
- [chore(tensilelite): Remove legacy StreamK modes](../sources/prs/rocm-libraries/PR-7980.md)
- [[GFX1250][MX GEMM] Unified FLATMM GroupedGemm Implementation for MX Data Types](../sources/prs/rocm-libraries/PR-8325.md)
- [[hipBLASLt] Instruct agents to use SPDX license headers and the rocm-libraries PR template](../sources/prs/rocm-libraries/PR-8431.md)
- [[CK_TILE] Use launched block size for GEMM occupancy query](../sources/prs/rocm-libraries/PR-8531.md)
- [[CK][CI] Expand other stages to use healthy-node retry logic.](../sources/prs/rocm-libraries/PR-8644.md)
- [fix(tensilelite): Fix PAP+TDM redundant descriptor rebuild](../sources/prs/rocm-libraries/PR-9196.md)
- [[hipblaslt][tensilelite] Remove legacy StreamK modes](../sources/prs/hipblaslt/PR-7980.md)
- [[hipblaslt][tensilelite] Single-hop next-neighbor StreamK work stealing](../sources/prs/hipblaslt/PR-8442.md)
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md)
- [[hipblaslt][tensilelite] Reorganize and expand coverage of GFX1250 StreamK tests](../sources/prs/hipblaslt/PR-8622.md)
- [implement persistent loop based rmsnorm kernel](../sources/prs/triton/PR-676.md)
- [Fix perfCI for streamk/persistent gemm on gfx950](../sources/prs/triton/PR-843.md)
- [[ROCm][Kernel][AITER] BlockScale FP8 SplitK zero-init fusion](../sources/prs/vllm/PR-44976.md)
- [[ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3](../sources/prs/vllm/PR-46117.md)
- [Kernel Launch Overhead Optimization](../wiki/techniques/kernel-launch-overhead.md)
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md)
- [Persistent Softmax Optimization in Triton](../wiki/techniques/pr-triton-634.md)
- [Persistent Loop-Based RMSNorm Kernel (Triton)](../wiki/techniques/pr-triton-676.md)

## pipeline-unification (1 pages)

- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md)

## production-shape-testing (1 pages)

- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/transformerengine/PR-605.md)

## quantization (4 pages)

- [[CK] Add FP8 KV_BLOCKSCALE support for batch prefill](../sources/prs/composable_kernel/PR-3696.md)
- [[CK Tile Engine] Add block-scale GEMM operators: gemm_aquant, gemm_bquant, gemm_abquant](../sources/prs/hipblaslt/PR-8519.md)
- [[minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra](../sources/prs/sglang/PR-28715.md)
- [[ROCm][CI] Only require q_scale==1.0 for fp8 query in RocmAttention](../sources/prs/vllm/PR-46148.md)

## quantizer-copy (1 pages)

- [MXFP4: Add GEMM kernel tuning and MXFP4Quantizer.copy()](../sources/prs/transformerengine/PR-535.md)

## register-tiling (16 pages)

- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md)
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md)
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md)
- [Compute-Bound Optimization Patterns (算力密集优化模式)](../wiki/patterns/compute-bound-optimization.md)
- [Compute-Bound MFMA Pattern on AMD GPUs](../wiki/patterns/compute-bound-mfma-amd.md)
- [Tile Quantization and Dequantization](../wiki/patterns/tile-quantize-dequant.md)
- [Implement device grouped gemm fixed nk multi abd for rdna4](../sources/prs/composable_kernel/PR-3619.md)
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md)
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md)
- [[ROCm][Kernel] Extend skinny gemm N=5 to N=8 cases on GFX12 (RDNA4) using SWMMAC optimization](../sources/prs/vllm/PR-45559.md)
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md)
- [Occupancy Tuning on ROCm](../wiki/techniques/occupancy-tuning.md)
- [Explicit Multiply-Reduce GEMM for Small Block Sizes in Triton](../wiki/techniques/pr-triton-621.md)
- [Register Tiling for MFMA Kernels](../wiki/techniques/register-tiling.md)
- [VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)](../wiki/techniques/vgpr-pressure.md)
- [XDLOPS 底层编程 (XDLOPS Low-level Programming)](../wiki/techniques/xdlops-programming.md)

## regression-test (2 pages)

- [[CK] Add support for large tensor index handling into conv bwd data WMMA](../sources/prs/hipblaslt/PR-8518.md)
- [[hipBLASLt] Fix int8 GEMM crash on alpha=1065353216](../sources/prs/hipblaslt/PR-8579.md)

## runtime-arch-dispatch (1 pages)

- [CK MXFP8 Group Gemm gfx1250 Enablement](../sources/prs/transformerengine/PR-613.md)

## runtime-dispatch (9 pages)

- [MIOpen Convolution Kernel Strategy on ROCm](../wiki/kernels/miopen-conv-strategy-rocm.md)
- [[Fix] compressed-tensors block FP8: requantize weight scales to UE8M0 for DeepGEMM on Blackwell](../sources/prs/sglang/PR-28662.md)
- [[AMD][Perf] Fuse QK RMSNorm + 3D mRoPE + KV-cache store into single aiter op for Qwen3.5-397B-A17B-MXFP4 (TP=2, ROCm/aiter) on HIP](../sources/prs/sglang/PR-28700.md)
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md)
- [[minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra](../sources/prs/sglang/PR-28715.md)
- [[AMD] Optimize o_proj gemm and attn output rope performance](../sources/prs/sglang/PR-28722.md)
- [[ROCm][Perf] MiniMax-M3 MXFP8 gemm/group gemm dispatch AITER](../sources/prs/vllm/PR-46063.md)
- [[Attention][DSA] support dcp for FLASHINFER_MLA_SPARSE](../sources/prs/vllm/PR-46076.md)
- [[ROCm][Perf] Optional FlyDSL BF16 MoE for the MXFP8-emulation path on MiniMax-M3](../sources/prs/vllm/PR-46123.md)

## runtime-guard (2 pages)

- [Fix Qwen MoE precision issue with PP and all-reduce fusion](../sources/prs/sglang/PR-28619.md)
- [[ROCm][CI] Only require q_scale==1.0 for fp8 query in RocmAttention](../sources/prs/vllm/PR-46148.md)

## sampling (1 pages)

- [[CK Tile Engine] Add block-scale GEMM operators: gemm_aquant, gemm_bquant, gemm_abquant](../sources/prs/hipblaslt/PR-8519.md)

## scalar-type-dispatch (1 pages)

- [[hipBLASLt] Fix int8 GEMM crash on alpha=1065353216](../sources/prs/hipblaslt/PR-8579.md)

## scale-layout-transform (1 pages)

- [gfx1250 mxfp8 gemm: add NN/NT transpose workaround](../sources/prs/transformerengine/PR-630.md)

## scale-preshuffle (3 pages)

- [add MXFP8 pre-swizzling for gfx1250 GEMM](../sources/prs/transformerengine/PR-568.md)
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/transformerengine/PR-605.md)
- [CK MXFP8 Group Gemm gfx1250 Enablement](../sources/prs/transformerengine/PR-613.md)

## scale-requantization (1 pages)

- [[Fix] compressed-tensors block FP8: requantize weight scales to UE8M0 for DeepGEMM on Blackwell](../sources/prs/sglang/PR-28662.md)

## shape-aware-heuristic (2 pages)

- [[hipblaslt][origami] Model changes for mi350P](../sources/prs/hipblaslt/PR-8600.md)
- [[CK DSL] conv heuristic: fix gemm_k_per_block, add K_per_C + log features, update all models to 101 features](../sources/prs/hipblaslt/PR-8620.md)

## shape-based-kernel-selection (10 pages)

- [CDNA4 (gfx950/MI350) Scaled-MFMA GEMM in hipBLASLt](../wiki/kernels/cdna4-hipblaslt-scaled-mfma-gemm.md)
- [hipSPARSELt SpMM (Sparse × Dense) on ROCm](../wiki/kernels/hipsparselt-spmm-rocm.md)
- [MIOpen Convolution Kernel Strategy on ROCm](../wiki/kernels/miopen-conv-strategy-rocm.md)
- [MXFP4: Add GEMM kernel tuning and MXFP4Quantizer.copy()](../sources/prs/transformerengine/PR-535.md)
- [[CI] Add aiter installation to CI image for MXFP4 FP4 GEMM kernels](../sources/prs/transformerengine/PR-562.md)
- [HipKittens MXFP8 GEMM Support](../sources/prs/transformerengine/PR-566.md)
- [CK Tile Group GEMM gfx1250](../sources/prs/transformerengine/PR-576.md)
- [NVFP4: Work around intermittent incorrect results for backward GEMMs](../sources/prs/transformerengine/PR-580.md)
- [Fix CK FP8 grouped GEMM dtype gating for columnwise operands](../sources/prs/transformerengine/PR-594.md)
- [add dsv4 production mxfp8 gemm shapes](../sources/prs/transformerengine/PR-636.md)

## shape-constraint-relaxation (1 pages)

- [gfx1250 mxfp8 gemm: loosen restrictions on K](../sources/prs/transformerengine/PR-627.md)

## shape-tuning (1 pages)

- [Tune gfx1100 BBS GEMM kernels for Llama-3.1-8b-Instruct](../sources/prs/hipblaslt/PR-8631.md)

## solution-library (1 pages)

- [Tune gfx1100 BBS GEMM kernels for Llama-3.1-8b-Instruct](../sources/prs/hipblaslt/PR-8631.md)

## solution-selection (1 pages)

- [[hipblaslt][origami] Model changes for mi350P](../sources/prs/hipblaslt/PR-8600.md)

## sparse-attention (2 pages)

- [[CK_TILE] Sparge attention](../sources/prs/composable_kernel/PR-3727.md)
- [[minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra](../sources/prs/sglang/PR-28715.md)

## split-kv (1 pages)

- [Ck tile/kvcache](../sources/prs/flash-attention/PR-74.md)

## staged-gemm (1 pages)

- [[ROCm][Perf] Optional FlyDSL BF16 MoE for the MXFP8-emulation path on MiniMax-M3](../sources/prs/vllm/PR-46123.md)

## stream-k (3 pages)

- [CDNA4 (gfx950/MI350) Scaled-MFMA GEMM in hipBLASLt](../wiki/kernels/cdna4-hipblaslt-scaled-mfma-gemm.md)
- [[CK_TILE] Use launched block size for GEMM occupancy query](../sources/prs/hipblaslt/PR-8531.md)
- [[hipblaslt][tensilelite] Reorganize and expand coverage of GFX1250 StreamK tests](../sources/prs/hipblaslt/PR-8622.md)

## stream-overlap (1 pages)

- [[feat] add ag_gemm and moe_rs overlap kernels for dsv4 prefill](../sources/prs/sglang/PR-28639.md)

## subtile (4 pages)

- [CDNA4 (gfx950/MI350) Scaled-MFMA GEMM in hipBLASLt](../wiki/kernels/cdna4-hipblaslt-scaled-mfma-gemm.md)
- [[hipblaslt][tensilelite] Add cluster barrier support for subtile gfx1250 kernels](../sources/prs/hipblaslt/PR-8523.md)
- [[hipblaslt][tensilelite] Add multicast tdm for subtile kernel](../sources/prs/hipblaslt/PR-8524.md)
- [[tensilelite] Fix subtile PGR=0 WMMA-source WAR hazard on gfx1250](../sources/prs/hipblaslt/PR-8603.md)

## swizzling (1 pages)

- [Cooperative Loading](../wiki/patterns/cooperative-loading.md)

## tdm (2 pages)

- [[hipblaslt][tensilelite] Add multicast tdm for subtile kernel](../sources/prs/hipblaslt/PR-8524.md)
- [[hipblaslt][tensilelite] Reorganize and expand coverage of GFX1250 StreamK tests](../sources/prs/hipblaslt/PR-8622.md)

## temporary-buffering (1 pages)

- [gfx1250 mxfp8 gemm: add NN/NT transpose workaround](../sources/prs/transformerengine/PR-630.md)

## test-matrix (1 pages)

- [[hipblaslt][tensilelite] Reorganize and expand coverage of GFX1250 StreamK tests](../sources/prs/hipblaslt/PR-8622.md)

## tiling (3 pages)

- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md)
- [HipKittens MXFP8 GEMM Support](../sources/prs/transformerengine/PR-566.md)
- [[ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3](../sources/prs/vllm/PR-46117.md)

## transpose-workaround (1 pages)

- [gfx1250 mxfp8 gemm: add NN/NT transpose workaround](../sources/prs/transformerengine/PR-630.md)

## vectorized-load (32 pages)

- [AMDGPU Kernel Optimization Guide](../sources/blogs/amdgpu-kernel-opt.md)
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md)
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md)
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md)
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md)
- [KV Cache Paged Attention on ROCm](../wiki/kernels/kv-cache-rocm.md)
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md)
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md)
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md)
- [RMSNorm and Normalization Kernels on ROCm](../wiki/kernels/rmsnorm-rocm.md)
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md)
- [Cooperative Loading](../wiki/patterns/cooperative-loading.md)
- [Grid-Stride Loop](../wiki/patterns/grid-stride-loop.md)
- [Memory-Bound Optimization Patterns](../wiki/patterns/memory-bound-optimization.md)
- [Scatter/Gather Memory Access Patterns](../wiki/patterns/scatter-gather.md)
- [Tile Quantization and Dequantization](../wiki/patterns/tile-quantize-dequant.md)
- [[BN] Finalize batch norm OpenCL kernel optimization](../sources/prs/MIOpen/PR-3564.md)
- [Vectorize Resize](../sources/prs/amdmigraphx/PR-4967.md)
- [[CK_TILE] fix enforcing fixed vectorsizes for ck tile conv](../sources/prs/composable_kernel/PR-3344.md)
- [[CK_TILE] fix(fmha): clamp paged KV lookups in batch prefill](../sources/prs/composable_kernel/PR-3733.md)
- [[tensilelite] Fix rocisa instruction mnemonics and add gfx12+ scalar ops](../sources/prs/hipblaslt/PR-8586.md)
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md)
- [[CK][CK DSL] Pass vector sizes as arguments for implicit gemm](../sources/prs/hipblaslt/PR-8624.md)
- [[AMD] Fuse shared-expert sigmoid + bf16->fp32 cast into the MoE append kernel (3 kernels -> 1)](../sources/prs/sglang/PR-28658.md)
- [Mxfp8 grouped and multi quantize](../sources/prs/transformerengine/PR-598.md)
- [[Fix] TE RMSNorm Triton Kernel Optimization](../sources/prs/transformerengine/PR-615.md)
- [gfx1250 mxfp8 gemm: add NN/NT transpose workaround](../sources/prs/transformerengine/PR-630.md)
- [[ROCm] Faster Custom Paged Attention kernels](../sources/prs/vllm/PR-12348.md)
- [[ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode](../sources/prs/vllm/PR-40977.md)
- [合并内存访问模式 (Coalesced Memory Access Patterns)](../wiki/techniques/coalesced-memory.md)
- [Persistent Softmax Optimization in Triton](../wiki/techniques/pr-triton-634.md)
- [Vectorized Global Memory Loads](../wiki/techniques/vectorized-loads.md)

## vgpr-reduction (1 pages)

- [Flat vs Buffer Addressing Modes](../wiki/techniques/flat-addressing.md)

## wait-state-insertion (1 pages)

- [[tensilelite] Fix subtile PGR=0 WMMA-source WAR hazard on gfx1250](../sources/prs/hipblaslt/PR-8603.md)

## wave-reduction (7 pages)

- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md)
- [RMSNorm and Normalization Kernels on ROCm](../wiki/kernels/rmsnorm-rocm.md)
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md)
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md)
- [Reduction Tree](../wiki/patterns/reduction-tree.md)
- [HIP Atomic Operations and Contention Reduction](../wiki/techniques/atomic-operations-hip.md)
- [Cross-Lane Communication with DPP (Warp Shuffle Equivalent)](../wiki/techniques/warp-shuffle-dpp.md)

## wave-synchronization (1 pages)

- [[hipblaslt][tensilelite] Add cluster barrier support for subtile gfx1250 kernels](../sources/prs/hipblaslt/PR-8523.md)

## weight-caching (1 pages)

- [Full MXFP4 Training Recipe](../sources/prs/transformerengine/PR-537.md)

## wmma (1 pages)

- [[CK] Add support for large tensor index handling into conv bwd data WMMA](../sources/prs/hipblaslt/PR-8518.md)

## wmma-scheduling (1 pages)

- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md)
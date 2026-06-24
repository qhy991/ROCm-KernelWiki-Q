# Index: By Technique


## aiter-dispatch (1 pages)

- [Full MXFP4 Training Recipe](../sources/prs/transformerengine/PR-537.md)

## assembly-emission (1 pages)

- [[hipblaslt][tensilelite] Add cluster barrier support for subtile gfx1250 kernels](../sources/prs/hipblaslt/PR-8523.md)

## async-copy (5 pages)

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

## bank-conflict-padding (4 pages)

- [AMDGPU Kernel Optimization Guide](../sources/blogs/amdgpu-kernel-opt.md)
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/hipblaslt/PR-605.md)
- [异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)](../wiki/techniques/async-copy-lds.md)
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md)

## bbs (1 pages)

- [Tune gfx1100 BBS GEMM kernels for Llama-3.1-8b-Instruct](../sources/prs/hipblaslt/PR-8631.md)

## block-scale (4 pages)

- [[CK Tile Engine] Add block-scale GEMM operators: gemm_aquant, gemm_bquant, gemm_abquant](../sources/prs/hipblaslt/PR-8519.md)
- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md)
- [[GFX1250][CK_TILE] Coalesce MX scale16 scale load](../sources/prs/hipblaslt/PR-8566.md)
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md)

## cache-invalidation (1 pages)

- [[RL] MXFP8 flashinfer_trtllm_routed MoE for V4](../sources/prs/sglang/PR-28676.md)

## ck-tile-programming (37 pages)

- [Composable Kernel Tile Tutorial](../sources/blogs/ck-tutorial.md)
- [Composable Kernel Repository Structure](../sources/docs/ck-structure.md)
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md)
- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md)
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md)
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md)
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md)
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md)
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md)
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md)
- [生产者-消费者流水线 (Producer-Consumer Pipeline)](../wiki/patterns/producer-consumer-pipeline.md)
- [Tile Quantization and Dequantization](../wiki/patterns/tile-quantize-dequant.md)
- [[CK] Add FP8 KV_BLOCKSCALE support for batch prefill](../sources/prs/composable_kernel/PR-3696.md)
- [[CK_TILE] Add support and tests for V6 pipeline in conv fwd](../sources/prs/composable_kernel/PR-3708.md)
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md)
- [[CK_TILE] Sparge attention](../sources/prs/composable_kernel/PR-3727.md)
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md)
- [[ck_tile/fmha] Fix sink un-mask under right-window and emit fp8bf16 batch_prefill sink kernels](../sources/prs/composable_kernel/PR-3732.md)
- [[CK_TILE] fix(fmha): clamp paged KV lookups in batch prefill](../sources/prs/composable_kernel/PR-3733.md)
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md)
- [[CK Tile] Prepare mixed batch-prefill FP8 KV contract](../sources/prs/composable_kernel/PR-3745.md)
- [[CK_TILE] Update CK and enable RDNA backward](../sources/prs/flash-attention/PR-184.md)
- [Ck tile/flash attention](../sources/prs/flash-attention/PR-61.md)
- [Integrate ck tile backward](../sources/prs/flash-attention/PR-65.md)
- [Improve FMHA bwd](../sources/prs/flash-attention/PR-70.md)
- [Ck tile/kvcache](../sources/prs/flash-attention/PR-74.md)
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md)
- [[CK_TILE] Scope NumWarps==8 CompV3 tail/epilogue logic to EightWaves …](../sources/prs/hipblaslt/PR-7669.md)
- [[CK] feat(ssd): add fp16/bf16 support with fp32 accumulation](../sources/prs/hipblaslt/PR-7851.md)
- [[CK_TILE] Add Tile Engine -> Dispatcher bridge for GEMM](../sources/prs/hipblaslt/PR-8123.md)
- [[TE] Improve backward performance for CK Tile FP8 Group GEMM](../sources/prs/transformerengine/PR-544.md)
- [CK Tile Group GEMM gfx1250](../sources/prs/transformerengine/PR-576.md)
- [CK Tile MXFP8 Group GEMM gfx1250](../sources/prs/transformerengine/PR-578.md)
- [CK MXFP8 Group Gemm gfx1250 Enablement](../sources/prs/transformerengine/PR-613.md)
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

## double-buffering (22 pages)

- [AMDGPU Kernel Optimization Guide](../sources/blogs/amdgpu-kernel-opt.md)
- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md)
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md)
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md)
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md)
- [Compute-Bound Optimization Patterns (算力密集优化模式)](../wiki/patterns/compute-bound-optimization.md)
- [Compute-Bound MFMA Pattern on AMD GPUs](../wiki/patterns/compute-bound-mfma-amd.md)
- [Latency Hiding (延迟隐藏)](../wiki/patterns/latency-hiding.md)
- [生产者-消费者流水线 (Producer-Consumer Pipeline)](../wiki/patterns/producer-consumer-pipeline.md)
- [Wavefront Specialization (Warp Specialization)](../wiki/patterns/warp-specialization.md)
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md)
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md)
- [[CK_TILE] fix(fmha): clamp paged KV lookups in batch prefill](../sources/prs/composable_kernel/PR-3733.md)
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

## mfma-scheduling (31 pages)

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
- [Add support to fp16 + compute fp16 and bf16 + compute bf16 contractions](../sources/prs/composable_kernel/PR-3598.md)
- [Implement device grouped gemm fixed nk multi abd for rdna4](../sources/prs/composable_kernel/PR-3619.md)
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md)
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md)
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

## persistent-kernel (14 pages)

- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md)
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md)
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md)
- [Grid-Stride Loop](../wiki/patterns/grid-stride-loop.md)
- [[hipblaslt][tensilelite] Remove legacy StreamK modes](../sources/prs/hipblaslt/PR-7980.md)
- [[hipblaslt][tensilelite] Single-hop next-neighbor StreamK work stealing](../sources/prs/hipblaslt/PR-8442.md)
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md)
- [[hipblaslt][tensilelite] Reorganize and expand coverage of GFX1250 StreamK tests](../sources/prs/hipblaslt/PR-8622.md)
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

## runtime-dispatch (8 pages)

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

## shape-based-kernel-selection (7 pages)

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

## stream-k (2 pages)

- [[CK_TILE] Use launched block size for GEMM occupancy query](../sources/prs/hipblaslt/PR-8531.md)
- [[hipblaslt][tensilelite] Reorganize and expand coverage of GFX1250 StreamK tests](../sources/prs/hipblaslt/PR-8622.md)

## stream-overlap (1 pages)

- [[feat] add ag_gemm and moe_rs overlap kernels for dsv4 prefill](../sources/prs/sglang/PR-28639.md)

## subtile (3 pages)

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

## vectorized-load (28 pages)

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
- [[CK_TILE] fix(fmha): clamp paged KV lookups in batch prefill](../sources/prs/composable_kernel/PR-3733.md)
- [[tensilelite] Fix rocisa instruction mnemonics and add gfx12+ scalar ops](../sources/prs/hipblaslt/PR-8586.md)
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md)
- [[CK][CK DSL] Pass vector sizes as arguments for implicit gemm](../sources/prs/hipblaslt/PR-8624.md)
- [[AMD] Fuse shared-expert sigmoid + bf16->fp32 cast into the MoE append kernel (3 kernels -> 1)](../sources/prs/sglang/PR-28658.md)
- [Mxfp8 grouped and multi quantize](../sources/prs/transformerengine/PR-598.md)
- [[Fix] TE RMSNorm Triton Kernel Optimization](../sources/prs/transformerengine/PR-615.md)
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
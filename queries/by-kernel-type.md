# Index: By Kernel Type


## attention (20 pages)

- [Flash Attention on ROCm](wiki/kernels/flash-attention-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FMHA] Support page_size=1 (linear layout) in batch prefill pipeline](sources/prs/composable_kernel/PR-3545.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FMHA] Enable page size 16 for batch prefill kernel](sources/prs/composable_kernel/PR-3568.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Add new tile size for async](sources/prs/composable_kernel/PR-3586.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Revert new tile size for async (#3586)"](sources/prs/composable_kernel/PR-3613.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix Int32 Overflow in Deterministic FMHA BWD](sources/prs/composable_kernel/PR-3615.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA]Add new tile size for async](sources/prs/composable_kernel/PR-3623.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert " Fp8 block scale quantization for fmha  fwd"](sources/prs/composable_kernel/PR-3633.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "Revert " Fp8 block scale quantization for fmha  fwd""](sources/prs/composable_kernel/PR-3635.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Support page attention in mha_varlen_fwd](sources/prs/flash-attention/PR-103.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [head, seq, batch grid order for triton flash attention bwd.](sources/prs/flash-attention/PR-141.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix NaN for FMHA BWD When seq_q=0](sources/prs/flash-attention/PR-179.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Use Unified Workspace for FMHA BWD](sources/prs/flash-attention/PR-182.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] FMHA BWD: stream-async workspace prepare](sources/prs/flash-attention/PR-183.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [vulkan: Intel Xe flash attention, GEMM optimizations, and optional weight compression (Xe-LPG Plus/Xe2/Xe3) [MEGA PR]](sources/prs/hipblaslt/PR-24408.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[feat] FP8 (DeepSeek-V4 layout) sparse paged prefill attention](sources/prs/hipblaslt/PR-3583.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Causal Flash attention](sources/prs/hipblaslt/PR-3943.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][AITER] Use pre-shuffled FP8 GEMM for Quark per-channel attention weights](sources/prs/hipblaslt/PR-44626.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[gfx1201] Mistral-3 + Qwen3-8B-FP8 on RDNA4 via native triton attention](sources/prs/hipblaslt/PR-811.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [CK Tile Programming Model](wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4

## conv (15 pages)

- [Fix grouped conv bwd data wmma check](sources/prs/composable_kernel/PR-3562.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [WMMA grouped conv fwd large tensor extra flavors](sources/prs/composable_kernel/PR-3582.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [WMMA grouped conv fwd large tensor bias bnorm clamp](sources/prs/composable_kernel/PR-3595.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] Replace reference conv with old ck implementation](sources/prs/composable_kernel/PR-3604.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] conv bwd weight testing](sources/prs/composable_kernel/PR-3618.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE] Enable CK TILE Conv Fwd tests in CI and fix check_err](sources/prs/composable_kernel/PR-3624.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Grouped conv fwd direct load vector=2](sources/prs/composable_kernel/PR-3632.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Add new instances for merging multiple fwd conv groups into a single GEMM batch](sources/prs/composable_kernel/PR-3639.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Grouped Conv Bwd Weight Direct Load](sources/prs/composable_kernel/PR-3648.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Conv] Enable bwd weight splitk autodeduction with cap](sources/prs/composable_kernel/PR-3656.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Enable Grouped Conv Tile Fwd Tests daily](sources/prs/composable_kernel/PR-3680.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix path to ck tile conv fwd instance generator](sources/prs/composable_kernel/PR-3699.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AIROCMLIR-71] Add gemm+gemm and conv+gemm support to quickTuningGen.py](sources/prs/hipblaslt/PR-2262.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[TRITON] Conv Kernels First Commit to AITER](sources/prs/hipblaslt/PR-2886.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Tile] Wavelet gemm pipeline for conv fwd](sources/prs/hipblaslt/PR-7196.md) conf:source-reported arch:cdna2, cdna3, cdna4

## flash-attention (1 pages)

- [Flash Attention on ROCm](wiki/kernels/flash-attention-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4

## gemm (45 pages)

- [[CK TILE GEMM] Add bf8 support to tile engine streamk generator](sources/prs/composable_kernel/PR-3543.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Batched gemm softmax gemm descriptor fix](sources/prs/composable_kernel/PR-3564.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Remove code duplications in batched gemm wmma](sources/prs/composable_kernel/PR-3580.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fixing GEMM Multi D on Tile Engine](sources/prs/composable_kernel/PR-3583.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE QUANT GEMM] use OverrideADataType in aquant pipeline](sources/prs/composable_kernel/PR-3584.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_Tile] Support for a4w4 (fp4) in block scale gemm AB quant](sources/prs/composable_kernel/PR-3603.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE] Fix basic gemm pipelines add v1 interwave pipeline](sources/prs/composable_kernel/PR-3611.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Remove code duplications in batched gemm (multi D) gemm (multi D) wmma](sources/prs/composable_kernel/PR-3617.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Implement device grouped gemm fixed nk multi abd for rdna4](sources/prs/composable_kernel/PR-3619.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [GEMM Blockscale ABQuant Optimization](sources/prs/composable_kernel/PR-3620.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_Tile] Adding support for preshuffleQuant in AB quant Block Scale Gemm](sources/prs/composable_kernel/PR-3629.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Add new instances for merging multiple fwd conv groups into a single GEMM batch](sources/prs/composable_kernel/PR-3639.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [feat: add split_k support for block scale gemm bquant mode.](sources/prs/composable_kernel/PR-3653.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "Implement device grouped gemm fixed nk multi abd for rdna4"](sources/prs/composable_kernel/PR-3705.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Optimize Featherstone GEMM kernels](sources/prs/hipblaslt/PR-1.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add coda-kernels-rust: CODA GEMM-epilogue kernels in Rust, CPU + CUDA](sources/prs/hipblaslt/PR-16.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[TLX][AMD] gfx9 fp16 GEMM tutorial (a16w16) v0-v9 on gfx950](sources/prs/hipblaslt/PR-1663.md) conf:source-reported arch:cdna4
- [[Kernel][Nemotron] SM100 FP8 dense GEMM + ReLU² fusions and Mamba2/RMSNorm fusions for Nemotron-3-Super NVFP4 (B200)](sources/prs/hipblaslt/PR-2.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AIROCMLIR-191] Add gemm+gemm support to CI](sources/prs/hipblaslt/PR-2175.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AIROCMLIR-71] Add gemm+gemm and conv+gemm support to quickTuningGen.py](sources/prs/hipblaslt/PR-2262.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [vulkan: Intel Xe flash attention, GEMM optimizations, and optional weight compression (Xe-LPG Plus/Xe2/Xe3) [MEGA PR]](sources/prs/hipblaslt/PR-24408.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [webgpu: adjust the parms for gemm-subgroup kernel](sources/prs/hipblaslt/PR-28760.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add GLM-4.7-FP8 tuned/untuned BF16 GEMM configs (gfx950)](sources/prs/hipblaslt/PR-3285.md) conf:source-reported arch:cdna4
- [[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE](sources/prs/hipblaslt/PR-330.md) conf:source-reported arch:cdna4
- [feat(cutile): add cutile backend to bmm_bf16 (BF16 batched GEMM)](sources/prs/hipblaslt/PR-3413.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add GLM-5.1 FP8 blockscale GEMM/FMoE tunings for gfx942 (MI300X/MI325)](sources/prs/hipblaslt/PR-3422.md) conf:source-reported arch:cdna3
- [Tune fused GEMM AFP4WFP4 A16W16 gfx950 config and add benchmark](sources/prs/hipblaslt/PR-3642.md) conf:source-reported arch:cdna4
- [opt(gemm): add AITER MXFP4 preshuffle fast path](sources/prs/hipblaslt/PR-366.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add Triton ROCm backend for FP4 GEMM ops (MXFP4 + NVFP4) on MI300X / MI350X](sources/prs/hipblaslt/PR-381.md) conf:source-reported arch:cdna4
- [[ROCm] Add Triton Backend for BF16 Grouped GEMM Backward Kernels](sources/prs/hipblaslt/PR-388.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add BF16xFP4 MoE GEMM stage1 kernel](sources/prs/hipblaslt/PR-424.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add A16W4 MoE GEMM stage2 kernel (BF16 activations x MXFP4 weights)](sources/prs/hipblaslt/PR-431.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Adds Grouped and Batched GEMM kernels with blockscaling matching DeepGEMM API](sources/prs/hipblaslt/PR-433.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][AITER] Use pre-shuffled FP8 GEMM for Quark per-channel attention weights](sources/prs/hipblaslt/PR-44626.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [reference/fp8-gemm-dsr1-rocm: closed-loop case study on AMD MI355X](sources/prs/hipblaslt/PR-5.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [xe: gemm: fixup bdpas scale arg layout](sources/prs/hipblaslt/PR-5303.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](sources/prs/hipblaslt/PR-5813.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](sources/prs/hipblaslt/PR-605.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Kokkos-Polybench: Add GEMM kernel](sources/prs/hipblaslt/PR-686.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add fused_gemm_benchmark.py: fused two-GEMM SwiGLU kernel benchmark](sources/prs/hipblaslt/PR-7152.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Tile] Wavelet gemm pipeline for conv fwd](sources/prs/hipblaslt/PR-7196.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Add Tile Engine -> Dispatcher bridge for GEMM](sources/prs/hipblaslt/PR-8123.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[hipBLASLt] Add GEKO GEMM kernel optimizer and Ductile genetic-algorithm tuning backend](sources/prs/hipblaslt/PR-8302.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [MoE: Grouped Triton GEMM for TTFT improvements](sources/prs/hipblaslt/PR-970.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [CK Tile Programming Model](wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4

## grouped-gemm (5 pages)

- [Implement device grouped gemm fixed nk multi abd for rdna4](sources/prs/composable_kernel/PR-3619.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "Implement device grouped gemm fixed nk multi abd for rdna4"](sources/prs/composable_kernel/PR-3705.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE](sources/prs/hipblaslt/PR-330.md) conf:source-reported arch:cdna4
- [[ROCm] Add Triton Backend for BF16 Grouped GEMM Backward Kernels](sources/prs/hipblaslt/PR-388.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [CK Tile Programming Model](wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4

## layernorm (2 pages)

- [Fix redundant cast in model sensitive rmsnorm](sources/prs/composable_kernel/PR-3681.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Kernel][Nemotron] SM100 FP8 dense GEMM + ReLU² fusions and Mamba2/RMSNorm fusions for Nemotron-3-Super NVFP4 (B200)](sources/prs/hipblaslt/PR-2.md) conf:source-reported arch:cdna2, cdna3, cdna4

## moe (7 pages)

- [[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE](sources/prs/hipblaslt/PR-330.md) conf:source-reported arch:cdna4
- [[FLYDSL MOE] mixed_moe + moe_gemm_2stage: fx internal-types cleanup (ASM-identical)](sources/prs/hipblaslt/PR-3450.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[PERF] MXFP4 (a4w4) MoE backend for gfx950](sources/prs/hipblaslt/PR-3470.md) conf:source-reported arch:cdna4
- [Add BF16xFP4 MoE GEMM stage1 kernel](sources/prs/hipblaslt/PR-424.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add A16W4 MoE GEMM stage2 kernel (BF16 activations x MXFP4 weights)](sources/prs/hipblaslt/PR-431.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [MoE: Grouped Triton GEMM for TTFT improvements](sources/prs/hipblaslt/PR-970.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [CK Tile Programming Model](wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4

## softmax (2 pages)

- [Batched gemm softmax gemm descriptor fix](sources/prs/composable_kernel/PR-3564.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix softmax unit test](sources/prs/composable_kernel/PR-3683.md) conf:source-reported arch:cdna2, cdna3, cdna4
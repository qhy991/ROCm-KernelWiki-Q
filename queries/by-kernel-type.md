# Index: By Kernel Type


## attention (79 pages)

- [ROCm FlashAttention Performance Notes](../sources/blogs/flash-attention-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [ROCm Flash Attention Repository](../sources/docs/flash-attention-rocm.md) conf:verified arch:cdna2, cdna3, cdna4
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "[Triton] Declare triton>=3.6.0 dependency "](../sources/prs/hipblaslt/PR-3272.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[feat] FP8 (DeepSeek-V4 layout) sparse paged prefill attention](../sources/prs/hipblaslt/PR-3583.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FMHA] Support page_size=1 (linear layout) in batch prefill pipeline](../sources/prs/composable_kernel/PR-3545.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FMHA] Enable page size 16 for batch prefill kernel](../sources/prs/composable_kernel/PR-3568.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Add new tile size for async](../sources/prs/composable_kernel/PR-3586.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Revert new tile size for async (#3586)"](../sources/prs/composable_kernel/PR-3613.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix Int32 Overflow in Deterministic FMHA BWD](../sources/prs/composable_kernel/PR-3615.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA]Add new tile size for async](../sources/prs/composable_kernel/PR-3623.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert " Fp8 block scale quantization for fmha  fwd"](../sources/prs/composable_kernel/PR-3633.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "Revert " Fp8 block scale quantization for fmha  fwd""](../sources/prs/composable_kernel/PR-3635.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Compiler] Addressing new compiler warnings](../sources/prs/composable_kernel/PR-3640.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Enable MQA/GQA in backward](../sources/prs/flash-attention/PR-100.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Dropout](../sources/prs/flash-attention/PR-101.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Support page attention in mha_varlen_fwd](../sources/prs/flash-attention/PR-103.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix mha_varlen_fwd num_split and change ck interface](../sources/prs/flash-attention/PR-104.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [add RDNA CI](../sources/prs/flash-attention/PR-105.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Minor fixes](../sources/prs/flash-attention/PR-107.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] FAv3 bwd test case & api usage update](../sources/prs/flash-attention/PR-112.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] FAv3 bwd minor changes](../sources/prs/flash-attention/PR-113.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Enable FAv3 bwd for head_size=64 dtype=bf16 atomic32](../sources/prs/flash-attention/PR-114.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [fp8 forward](../sources/prs/flash-attention/PR-116.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [fa3 update ck](../sources/prs/flash-attention/PR-117.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [update CK](../sources/prs/flash-attention/PR-118.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [fp8 backward](../sources/prs/flash-attention/PR-119.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [reenable gfx1100 ci](../sources/prs/flash-attention/PR-121.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Performant backward Triton implementation with separated dkdv and dq kernels](../sources/prs/flash-attention/PR-122.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Quick Fixes](../sources/prs/flash-attention/PR-124.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [change version to 3.0.0.r1](../sources/prs/flash-attention/PR-125.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [update triton commit](../sources/prs/flash-attention/PR-128.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [update base docker image](../sources/prs/flash-attention/PR-129.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Casting Kernel](../sources/prs/flash-attention/PR-130.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Clean up readme](../sources/prs/flash-attention/PR-131.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [use triton==3.2.0](../sources/prs/flash-attention/PR-132.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Update README.md](../sources/prs/flash-attention/PR-134.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Bench](../sources/prs/flash-attention/PR-135.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fused Bwd](../sources/prs/flash-attention/PR-137.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Enable Alibi](../sources/prs/flash-attention/PR-138.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add alibi in the new bwd kernel](../sources/prs/flash-attention/PR-139.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add fp8 to fused kernel](../sources/prs/flash-attention/PR-140.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [head, seq, batch grid order for triton flash attention bwd.](../sources/prs/flash-attention/PR-141.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix keys](../sources/prs/flash-attention/PR-144.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Use more reasonable splitkv heuristic](../sources/prs/flash-attention/PR-147.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Pad LSE](../sources/prs/flash-attention/PR-148.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Pass logits soft-capping arguments](../sources/prs/flash-attention/PR-150.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Sliding Window Forward](../sources/prs/flash-attention/PR-151.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix Device Segfault](../sources/prs/flash-attention/PR-152.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix fp8 docs](../sources/prs/flash-attention/PR-154.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Sliding Window block classification logic](../sources/prs/flash-attention/PR-155.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Tune BWD](../sources/prs/flash-attention/PR-156.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Enable FA V3](../sources/prs/flash-attention/PR-157.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [AITER integration](../sources/prs/flash-attention/PR-159.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Tune FP8 Perf](../sources/prs/flash-attention/PR-160.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Update to ROCm/composable_kernel@e951863](../sources/prs/flash-attention/PR-173.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [update CK](../sources/prs/flash-attention/PR-174.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Update to ROCm/rocm-libraries#4368 (ROCm/rocm-libraries@17f7dfc)](../sources/prs/flash-attention/PR-176.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Update to ROCm/rocm-libraries@a358a21](../sources/prs/flash-attention/PR-177.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Update CK and add RDNA build support](../sources/prs/flash-attention/PR-178.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix NaN for FMHA BWD When seq_q=0](../sources/prs/flash-attention/PR-179.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Update CK submodule and fix fmha_bwd_args](../sources/prs/flash-attention/PR-181.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Use Unified Workspace for FMHA BWD](../sources/prs/flash-attention/PR-182.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] FMHA BWD: stream-async workspace prepare](../sources/prs/flash-attention/PR-183.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[WIP] test: cut unit-test CI wall time](../sources/prs/hipblaslt/PR-3601.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[gfx1201] Mistral-3 + Qwen3-8B-FP8 on RDNA4 via native triton attention](../sources/prs/hipblaslt/PR-811.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [vulkan: Intel Xe flash attention, GEMM optimizations, and optional weight compression (Xe-LPG Plus/Xe2/Xe3) [MEGA PR]](../sources/prs/hipblaslt/PR-24408.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm] Enable native AsyncTP](../sources/prs/hipblaslt/PR-177961.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Add Tile Engine -> Dispatcher bridge for GEMM](../sources/prs/hipblaslt/PR-8123.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AIROCMLIR-798] Add LDS usage estimate CAPI function](../sources/prs/hipblaslt/PR-2400.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AMD/gfx950] FlyDSL kgather diagnostic backend for DSv4 sparse FP8 MLA decode](../sources/prs/hipblaslt/PR-13.md) conf:source-reported arch:cdna4
- [[ROCm][AITER] Use pre-shuffled FP8 GEMM for Quark per-channel attention weights](../sources/prs/hipblaslt/PR-44626.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) conf:verified arch:cdna1, cdna2, cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [LDS Double Buffering](../wiki/techniques/double-buffering.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Register Tiling for MFMA Kernels](../wiki/techniques/register-tiling.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Vectorized Global Memory Loads](../wiki/techniques/vectorized-loads.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4

## conv (34 pages)

- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[TRITON] Conv Kernels First Commit to AITER](../sources/prs/hipblaslt/PR-2886.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Add base class GridwiseGemm_xdl_cshuffle_base for all gridwise_gemm_xdl classes](../sources/prs/composable_kernel/PR-3544.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] Convert convolution traits to a struct with factory functions](../sources/prs/composable_kernel/PR-3547.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Profiler] Initialize tensors on GPU in CK profiler](../sources/prs/composable_kernel/PR-3550.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Refactor GPU verification kernel to gather error stats on GPU](../sources/prs/composable_kernel/PR-3551.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE] Add grouped convolution forward tests](../sources/prs/composable_kernel/PR-3556.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix grouped conv bwd data wmma check](../sources/prs/composable_kernel/PR-3562.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Disable ActiveWorkgroupsPerCU for different arch in wmma kernels](../sources/prs/composable_kernel/PR-3566.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [CK: removed the api reference](../sources/prs/composable_kernel/PR-3571.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [WMMA grouped conv fwd large tensor extra flavors](../sources/prs/composable_kernel/PR-3582.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Profiler] Restore CPU tensor initialization when verification is not done on GPU](../sources/prs/composable_kernel/PR-3594.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [WMMA grouped conv fwd large tensor bias bnorm clamp](../sources/prs/composable_kernel/PR-3595.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] Replace reference conv with old ck implementation](../sources/prs/composable_kernel/PR-3604.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] conv bwd weight testing](../sources/prs/composable_kernel/PR-3618.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Adding CK Tile to the doc](../sources/prs/composable_kernel/PR-3621.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE] Enable CK TILE Conv Fwd tests in CI and fix check_err](../sources/prs/composable_kernel/PR-3624.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Grouped conv fwd direct load vector=2](../sources/prs/composable_kernel/PR-3632.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [ck: add CK_USE_GFX950 macro](../sources/prs/composable_kernel/PR-3636.md) conf:source-reported arch:cdna4
- [[CK] Add new instances for merging multiple fwd conv groups into a single GEMM batch](../sources/prs/composable_kernel/PR-3639.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Compiler] Addressing new compiler warnings](../sources/prs/composable_kernel/PR-3640.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Grouped Conv Bwd Weight Direct Load](../sources/prs/composable_kernel/PR-3648.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] Integrate CKB validation with CK verification](../sources/prs/composable_kernel/PR-3649.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Conv] Enable bwd weight splitk autodeduction with cap](../sources/prs/composable_kernel/PR-3656.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] removing api ref etc](../sources/prs/composable_kernel/PR-3659.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Update pytorch version in convolution dataset test generation](../sources/prs/composable_kernel/PR-3667.md) conf:source-reported arch:cdna4
- [[CK_BUILDER] fix test related to changed xdl bwd cshuf v3 interface](../sources/prs/composable_kernel/PR-3677.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Enable Grouped Conv Tile Fwd Tests daily](../sources/prs/composable_kernel/PR-3680.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix path to ck tile conv fwd instance generator](../sources/prs/composable_kernel/PR-3699.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AIROCMLIR-71] Add gemm+gemm and conv+gemm support to quickTuningGen.py](../sources/prs/hipblaslt/PR-2262.md) conf:source-reported arch:cdna4
- [[CK Tile] Wavelet gemm pipeline for conv fwd](../sources/prs/hipblaslt/PR-7196.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Add Tile Engine -> Dispatcher bridge for GEMM](../sources/prs/hipblaslt/PR-8123.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AMD] Restrict BlockPingPong scheduling for loop-variant masked loads](../sources/prs/hipblaslt/PR-10585.md) conf:source-reported arch:cdna4
- [Vectorized Global Memory Loads](../wiki/techniques/vectorized-loads.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4

## flash-attention (1 pages)

- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4

## gemm (103 pages)

- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[TRITON] Conv Kernels First Commit to AITER](../sources/prs/hipblaslt/PR-2886.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Triton] batched_gemm_a16wfp4 (gfx950): fuse dot_scaled accumulator; branchless mxfp4 quant; tune small-N configs](../sources/prs/hipblaslt/PR-3058.md) conf:source-reported arch:cdna4
- [[TRITON] gfx1201: gemm_a8w8 tuning configs (Mistral-3 / Qwen3 shapes)](../sources/prs/hipblaslt/PR-3168.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "[Triton] Declare triton>=3.6.0 dependency "](../sources/prs/hipblaslt/PR-3272.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add GLM-4.7-FP8 tuned/untuned BF16 GEMM configs (gfx950)](../sources/prs/hipblaslt/PR-3285.md) conf:source-reported arch:cdna4
- [gfx1201 gemm_a8w8: blockscale HIP→triton fallback + tuning configs (plain + blockscale_preshuffled)](../sources/prs/hipblaslt/PR-3343.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[gluon gemm_afp4wfp4] Fix data access pattern to remove redundant data loads](../sources/prs/hipblaslt/PR-3355.md) conf:source-reported arch:cdna4
- [Add GLM-5.1 FP8 blockscale GEMM/FMoE tunings for gfx942 (MI300X/MI325)](../sources/prs/hipblaslt/PR-3422.md) conf:source-reported arch:cdna3
- [[FLYDSL MOE] mixed_moe + moe_gemm_2stage: fx internal-types cleanup (ASM-identical)](../sources/prs/hipblaslt/PR-3450.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fused SplitK zero-init for FP8 a8w8 blockscale GEMMs (y_is_zeroed) + re-enable CKTile SplitK](../sources/prs/hipblaslt/PR-3457.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[PERF] MXFP4 (a4w4) MoE backend for gfx950](../sources/prs/hipblaslt/PR-3470.md) conf:source-reported arch:cdna4
- [Tune fused GEMM AFP4WFP4 A16W16 gfx950 config and add benchmark](../sources/prs/hipblaslt/PR-3642.md) conf:source-reported arch:cdna4
- [feat: Add Interwave scheduler for aquant memory pipeline](../sources/prs/composable_kernel/PR-3540.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [refactor: remove Default scheduler implementation as it not used anymore](../sources/prs/composable_kernel/PR-3542.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE GEMM] Add bf8 support to tile engine streamk generator](../sources/prs/composable_kernel/PR-3543.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Add base class GridwiseGemm_xdl_cshuffle_base for all gridwise_gemm_xdl classes](../sources/prs/composable_kernel/PR-3544.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE ENGINE] CI fix for Basic Tile Engine](../sources/prs/composable_kernel/PR-3554.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Temporarily disable CK Tile Stream-K reduction tests](../sources/prs/composable_kernel/PR-3559.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Batched gemm softmax gemm descriptor fix](../sources/prs/composable_kernel/PR-3564.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Disable ActiveWorkgroupsPerCU for different arch in wmma kernels](../sources/prs/composable_kernel/PR-3566.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [CK: removed the api reference](../sources/prs/composable_kernel/PR-3571.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Multi AB support for wave transfer](../sources/prs/composable_kernel/PR-3578.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Remove code duplications in batched gemm wmma](../sources/prs/composable_kernel/PR-3580.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fixing GEMM Multi D on Tile Engine](../sources/prs/composable_kernel/PR-3583.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE QUANT GEMM] use OverrideADataType in aquant pipeline](../sources/prs/composable_kernel/PR-3584.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Mx fp6 flatmm](../sources/prs/composable_kernel/PR-3601.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_Tile] Support for a4w4 (fp4) in block scale gemm AB quant](../sources/prs/composable_kernel/PR-3603.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Re enable f8 x bf8 tests on compv3 and compv4](../sources/prs/composable_kernel/PR-3605.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE] Fix basic gemm pipelines add v1 interwave pipeline](../sources/prs/composable_kernel/PR-3611.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Remove code duplications in batched gemm (multi D) gemm (multi D) wmma](../sources/prs/composable_kernel/PR-3617.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Implement device grouped gemm fixed nk multi abd for rdna4](../sources/prs/composable_kernel/PR-3619.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [GEMM Blockscale ABQuant Optimization](../sources/prs/composable_kernel/PR-3620.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix alignment in Stream-K workspace buffer](../sources/prs/composable_kernel/PR-3625.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_Tile] Adding support for preshuffleQuant in AB quant Block Scale Gemm](../sources/prs/composable_kernel/PR-3629.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] ABQuant New Preshuffle](../sources/prs/composable_kernel/PR-3638.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Add new instances for merging multiple fwd conv groups into a single GEMM batch](../sources/prs/composable_kernel/PR-3639.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Compiler] Addressing new compiler warnings](../sources/prs/composable_kernel/PR-3640.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE ENGINE] Updating supported warp tile in Tile Engine](../sources/prs/composable_kernel/PR-3643.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Grouped Conv Bwd Weight Direct Load](../sources/prs/composable_kernel/PR-3648.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [feat: add split_k support for block scale gemm bquant mode.](../sources/prs/composable_kernel/PR-3653.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] removing api ref etc](../sources/prs/composable_kernel/PR-3659.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Stream-K Tile Engine Test Config File Generation](../sources/prs/composable_kernel/PR-3662.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix incompatible vector type arguments for the intrinsic calls](../sources/prs/composable_kernel/PR-3672.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix one more lifetimebound error.](../sources/prs/composable_kernel/PR-3703.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "Implement device grouped gemm fixed nk multi abd for rdna4"](../sources/prs/composable_kernel/PR-3705.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [feat(cutile): add cutile backend to bmm_bf16 (BF16 batched GEMM)](../sources/prs/hipblaslt/PR-3413.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[WIP] test: cut unit-test CI wall time](../sources/prs/hipblaslt/PR-3601.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Optimize Featherstone GEMM kernels](../sources/prs/hipblaslt/PR-1.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add coda-kernels-rust: CODA GEMM-epilogue kernels in Rust, CPU + CUDA](../sources/prs/hipblaslt/PR-16.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [AMD - gpt-oss vllm mxfp4: AITER tuning + n-gram spec decode + server …](../sources/prs/hipblaslt/PR-1657.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[TLX][AMD] gfx9 fp16 GEMM tutorial (a16w16) v0-v9 on gfx950](../sources/prs/hipblaslt/PR-1663.md) conf:source-reported arch:cdna4
- [[Kernel][Nemotron] SM100 FP8 dense GEMM + ReLU² fusions and Mamba2/RMSNorm fusions for Nemotron-3-Super NVFP4 (B200)](../sources/prs/hipblaslt/PR-2.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AIROCMLIR-191] Add gemm+gemm support to CI](../sources/prs/hipblaslt/PR-2175.md) conf:source-reported arch:cdna4
- [[AIROCMLIR-71] Add gemm+gemm and conv+gemm support to quickTuningGen.py](../sources/prs/hipblaslt/PR-2262.md) conf:source-reported arch:cdna4
- [Made a new branch to add necessary files for batched_gemm_a16w8_block…](../sources/prs/hipblaslt/PR-2297.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [feat(skills): enhance FlyDSL optimization skills](../sources/prs/hipblaslt/PR-269.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE](../sources/prs/hipblaslt/PR-330.md) conf:source-reported arch:cdna4
- [feat: triton_grouped_gemm: add work-stealing variant with ws_mode API ](../sources/prs/hipblaslt/PR-353.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [opt(gemm): add AITER MXFP4 preshuffle fast path](../sources/prs/hipblaslt/PR-366.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add Triton ROCm backend for FP4 GEMM ops (MXFP4 + NVFP4) on MI300X / MI350X](../sources/prs/hipblaslt/PR-381.md) conf:source-reported arch:cdna4
- [[ROCm] Add Triton Backend for BF16 Grouped GEMM Backward Kernels](../sources/prs/hipblaslt/PR-388.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add BF16xFP4 MoE GEMM stage1 kernel](../sources/prs/hipblaslt/PR-424.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add A16W4 MoE GEMM stage2 kernel (BF16 activations x MXFP4 weights)](../sources/prs/hipblaslt/PR-431.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Adds Grouped and Batched GEMM kernels with blockscaling matching DeepGEMM API](../sources/prs/hipblaslt/PR-433.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [reference/fp8-gemm-dsr1-rocm: closed-loop case study on AMD MI355X](../sources/prs/hipblaslt/PR-5.md) conf:source-reported arch:cdna4
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/hipblaslt/PR-605.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Kokkos-Polybench: Add GEMM kernel](../sources/prs/hipblaslt/PR-686.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [MoE: Grouped Triton GEMM for TTFT improvements](../sources/prs/hipblaslt/PR-970.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add fused_gemm_benchmark.py: fused two-GEMM SwiGLU kernel benchmark](../sources/prs/hipblaslt/PR-7152.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [vulkan: Intel Xe flash attention, GEMM optimizations, and optional weight compression (Xe-LPG Plus/Xe2/Xe3) [MEGA PR]](../sources/prs/hipblaslt/PR-24408.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [xe: gemm: fixup bdpas scale arg layout](../sources/prs/hipblaslt/PR-5303.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [webgpu: adjust the parms for gemm-subgroup kernel](../sources/prs/hipblaslt/PR-28760.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm] Enable native AsyncTP](../sources/prs/hipblaslt/PR-177961.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add dense FlexGEMM QuACK tuning](../sources/prs/hipblaslt/PR-187108.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) conf:source-reported arch:cdna4
- [Add Triton specialization paths to origami (gated on target_t)](../sources/prs/hipblaslt/PR-6604.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Tensilelite] Fix incorrect output for SIA0 + PGR](../sources/prs/hipblaslt/PR-6993.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Tile] Wavelet gemm pipeline for conv fwd](../sources/prs/hipblaslt/PR-7196.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[hipBLASLt] [TensileLite] Add initial tail loop support for Subtile path](../sources/prs/hipblaslt/PR-7636.md) conf:source-reported arch:cdna4
- [[CK_TILE] Scope NumWarps==8 CompV3 tail/epilogue logic to EightWaves …](../sources/prs/hipblaslt/PR-7669.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [# [TensileLite] Decouple MXFP8 scale DepthU from data DepthU (`ScaleDepthURatio`)](../sources/prs/hipblaslt/PR-7767.md) conf:source-reported arch:cdna4
- [TensileLite: Add multi-DU support.](../sources/prs/hipblaslt/PR-7781.md) conf:source-reported arch:cdna4
- [[Hipblaslt] Allow Subtile path to use BF16 any-K and MX K%32 tail loop](../sources/prs/hipblaslt/PR-7782.md) conf:source-reported arch:cdna4
- [RotateCoObject](../sources/prs/hipblaslt/PR-7788.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[WIP] Add TensileLite race detection CI workflow](../sources/prs/hipblaslt/PR-7951.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[hipblaslt][tensilelite] Remove legacy StreamK modes](../sources/prs/hipblaslt/PR-7980.md) conf:source-reported arch:cdna4
- [Move cluster barrier implementation from tensilelite to stinkytofu](../sources/prs/hipblaslt/PR-8101.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[hipblaslt][tensilelite] Add `smoke` test suite](../sources/prs/hipblaslt/PR-8117.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Add Tile Engine -> Dispatcher bridge for GEMM](../sources/prs/hipblaslt/PR-8123.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[hipBLASLt] Add GEKO GEMM kernel optimizer and Ductile genetic-algorithm tuning backend](../sources/prs/hipblaslt/PR-8302.md) conf:source-reported arch:cdna4
- [[hipblaslt][G-F-A] Demo calling triton fp4 kernel](../sources/prs/hipblaslt/PR-8336.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AIROCMLIR-798] Add LDS usage estimate CAPI function](../sources/prs/hipblaslt/PR-2400.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AMD][ROCm] Fix CI failures on gfx950, gfx1100, gfx1151, and gfx1201](../sources/prs/hipblaslt/PR-2326.md) conf:source-reported arch:cdna4
- [[AMD] Restrict BlockPingPong scheduling for loop-variant masked loads](../sources/prs/hipblaslt/PR-10585.md) conf:source-reported arch:cdna4
- [[ROCm][AITER] Use pre-shuffled FP8 GEMM for Quark per-channel attention weights](../sources/prs/hipblaslt/PR-44626.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) conf:verified arch:cdna1, cdna2, cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [LDS Double Buffering](../wiki/techniques/double-buffering.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Register Tiling for MFMA Kernels](../wiki/techniques/register-tiling.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Vectorized Global Memory Loads](../wiki/techniques/vectorized-loads.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4

## grouped-gemm (11 pages)

- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Implement device grouped gemm fixed nk multi abd for rdna4](../sources/prs/composable_kernel/PR-3619.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_Tile] Adding support for preshuffleQuant in AB quant Block Scale Gemm](../sources/prs/composable_kernel/PR-3629.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "Implement device grouped gemm fixed nk multi abd for rdna4"](../sources/prs/composable_kernel/PR-3705.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE](../sources/prs/hipblaslt/PR-330.md) conf:source-reported arch:cdna4
- [feat: triton_grouped_gemm: add work-stealing variant with ws_mode API ](../sources/prs/hipblaslt/PR-353.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add Triton ROCm backend for FP4 GEMM ops (MXFP4 + NVFP4) on MI300X / MI350X](../sources/prs/hipblaslt/PR-381.md) conf:source-reported arch:cdna4
- [[ROCm] Add Triton Backend for BF16 Grouped GEMM Backward Kernels](../sources/prs/hipblaslt/PR-388.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) conf:source-reported arch:cdna2, cdna3, cdna4

## layernorm (3 pages)

- [Fix redundant cast in model sensitive rmsnorm](../sources/prs/composable_kernel/PR-3681.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Kernel][Nemotron] SM100 FP8 dense GEMM + ReLU² fusions and Mamba2/RMSNorm fusions for Nemotron-3-Super NVFP4 (B200)](../sources/prs/hipblaslt/PR-2.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [rocWMMA: add gfx1032 (RDNA2) support with software WMMA fallback](../sources/prs/hipblaslt/PR-8209.md) conf:source-reported arch:cdna2, cdna3, cdna4

## moe (11 pages)

- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add GLM-5.1 FP8 blockscale GEMM/FMoE tunings for gfx942 (MI300X/MI325)](../sources/prs/hipblaslt/PR-3422.md) conf:source-reported arch:cdna3
- [[FLYDSL MOE] mixed_moe + moe_gemm_2stage: fx internal-types cleanup (ASM-identical)](../sources/prs/hipblaslt/PR-3450.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[PERF] MXFP4 (a4w4) MoE backend for gfx950](../sources/prs/hipblaslt/PR-3470.md) conf:source-reported arch:cdna4
- [[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE](../sources/prs/hipblaslt/PR-330.md) conf:source-reported arch:cdna4
- [Add BF16xFP4 MoE GEMM stage1 kernel](../sources/prs/hipblaslt/PR-424.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add A16W4 MoE GEMM stage2 kernel (BF16 activations x MXFP4 weights)](../sources/prs/hipblaslt/PR-431.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [MoE: Grouped Triton GEMM for TTFT improvements](../sources/prs/hipblaslt/PR-970.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Register Tiling for MFMA Kernels](../wiki/techniques/register-tiling.md) conf:source-reported arch:cdna2, cdna3, cdna4

## reduction (14 pages)

- [[CK_TILE ENGINE] Fix incorrect List import in reduce_parameter.py](../sources/prs/composable_kernel/PR-3555.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Temporarily disable CK Tile Stream-K reduction tests](../sources/prs/composable_kernel/PR-3559.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] Update owners file for more reviews for CK Builder](../sources/prs/composable_kernel/PR-3572.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Optimize sequence_gen and uniform_sequence_gen to reduce template instantiation depth](../sources/prs/composable_kernel/PR-3585.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Tile] multi reduce improvements](../sources/prs/composable_kernel/PR-3607.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix alignment in Stream-K workspace buffer](../sources/prs/composable_kernel/PR-3625.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add missing check target in reduce tile engine op](../sources/prs/composable_kernel/PR-3631.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Solve the staging compiler regression and enhance the docker container execution in script/tools](../sources/prs/composable_kernel/PR-3634.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add python analysis scripts for Clang's time trace](../sources/prs/composable_kernel/PR-3644.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Stream-K Tile Engine Test Config File Generation](../sources/prs/composable_kernel/PR-3662.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add a flag to build CK libs required for HipTensor.](../sources/prs/composable_kernel/PR-3684.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Remove builds on legacy OSs from CI](../sources/prs/composable_kernel/PR-3693.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AMD][ROCm] Fix CI failures on gfx950, gfx1100, gfx1151, and gfx1201](../sources/prs/hipblaslt/PR-2326.md) conf:source-reported arch:cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) conf:verified arch:cdna1, cdna2, cdna3, cdna4

## rmsnorm (4 pages)

- [Fused SplitK zero-init for FP8 a8w8 blockscale GEMMs (y_is_zeroed) + re-enable CKTile SplitK](../sources/prs/hipblaslt/PR-3457.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix redundant cast in model sensitive rmsnorm](../sources/prs/composable_kernel/PR-3681.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Kernel][Nemotron] SM100 FP8 dense GEMM + ReLU² fusions and Mamba2/RMSNorm fusions for Nemotron-3-Super NVFP4 (B200)](../sources/prs/hipblaslt/PR-2.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add dense FlexGEMM QuACK tuning](../sources/prs/hipblaslt/PR-187108.md) conf:source-reported arch:cdna2, cdna3, cdna4

## softmax (5 pages)

- [[CK] Add base class GridwiseGemm_xdl_cshuffle_base for all gridwise_gemm_xdl classes](../sources/prs/composable_kernel/PR-3544.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Batched gemm softmax gemm descriptor fix](../sources/prs/composable_kernel/PR-3564.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Solve the staging compiler regression and enhance the docker container execution in script/tools](../sources/prs/composable_kernel/PR-3634.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix softmax unit test](../sources/prs/composable_kernel/PR-3683.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Pad LSE](../sources/prs/flash-attention/PR-148.md) conf:source-reported arch:cdna2, cdna3, cdna4
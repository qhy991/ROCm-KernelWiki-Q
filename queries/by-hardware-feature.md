# Index: By Hardware Feature


## block-scale (29 pages)

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
- [feat: Add Interwave scheduler for aquant memory pipeline](../sources/prs/composable_kernel/PR-3540.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_Tile] Support for a4w4 (fp4) in block scale gemm AB quant](../sources/prs/composable_kernel/PR-3603.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [GEMM Blockscale ABQuant Optimization](../sources/prs/composable_kernel/PR-3620.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_Tile] Adding support for preshuffleQuant in AB quant Block Scale Gemm](../sources/prs/composable_kernel/PR-3629.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_TILE] ABQuant New Preshuffle](../sources/prs/composable_kernel/PR-3638.md) `[source-pr]` arch:cdna2, cdna3, cdna4
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

## bpermute (3 pages)

- [AMD GCN Assembly Cross-Lane Operations](../sources/blogs/gcn-cross-lane.md) `[source-blog]` arch:cdna1, cdna2, cdna3, cdna4
- [DPP — Data-Parallel Primitives (Cross-Lane Operations)](../wiki/hardware/dpp-cross-lane.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4

## dpp (6 pages)

- [AMD GCN Assembly Cross-Lane Operations](../sources/blogs/gcn-cross-lane.md) `[source-blog]` arch:cdna1, cdna2, cdna3, cdna4
- [DPP — Data-Parallel Primitives (Cross-Lane Operations)](../wiki/hardware/dpp-cross-lane.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Wavefront Reduction using DPP](../wiki/techniques/wave-reduction.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3

## dual-cma (5 pages)

- [Matrix Core Programming on CDNA](../sources/blogs/matrix-cores-cdna.md) `[source-blog]` arch:cdna2, cdna3, cdna4
- [AMD CDNA4 Instruction Set Architecture Reference](../sources/docs/cdna4-isa.md) `[source-doc]` arch:cdna4
- [AMD Instinct MI350 Series Architecture Overview](../sources/docs/cdna4-whitepaper.md) `[source-doc]` arch:cdna4
- [MFMA Matrix Core (CDNA1–CDNA4)](../wiki/hardware/mfma-matrix-core.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4

## gws (3 pages)

- [GWS — Global Wave Sync](../wiki/hardware/gws.md) `[wiki-hardware]` arch:cdna2, cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4

## lds (25 pages)

- [LDS — Local Data Share](../wiki/hardware/lds.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Read-with-Transpose (CDNA4)](../wiki/hardware/lds-transpose.md) `[wiki-hardware]` arch:cdna4
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [GEMM Implementation on AMD CDNA](../wiki/kernels/gemm-rocm.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Reduction Kernels on ROCm](../wiki/kernels/reduction-rocm.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [[CK_Tile] Support for a4w4 (fp4) in block scale gemm AB quant](../sources/prs/composable_kernel/PR-3603.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) `[source-pr]` arch:cdna4
- [# [TensileLite] Decouple MXFP8 scale DepthU from data DepthU (`ScaleDepthURatio`)](../sources/prs/hipblaslt/PR-7767.md) `[source-pr]` arch:cdna4
- [[AIROCMLIR-798] Add LDS usage estimate CAPI function](../sources/prs/hipblaslt/PR-2400.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [LDS Double Buffering](../wiki/techniques/double-buffering.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Address Swizzling](../wiki/techniques/swizzling.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3
- [Vectorized Global Memory Loads](../wiki/techniques/vectorized-loads.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4

## lds-transpose (6 pages)

- [AMD CDNA4 Instruction Set Architecture Reference](../sources/docs/cdna4-isa.md) `[source-doc]` arch:cdna4
- [AMD Instinct MI350 Series Architecture Overview](../sources/docs/cdna4-whitepaper.md) `[source-doc]` arch:cdna4
- [LDS — Local Data Share](../wiki/hardware/lds.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Read-with-Transpose (CDNA4)](../wiki/hardware/lds-transpose.md) `[wiki-hardware]` arch:cdna4
- [CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide](../wiki/migration/cdna3-to-cdna4.md) `[wiki-migration]` arch:cdna3, cdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) `[source-pr]` arch:cdna4

## mfma (38 pages)

- [ROCm FlashAttention Performance Notes](../sources/blogs/flash-attention-rocm.md) `[source-blog]` arch:cdna2, cdna3, cdna4
- [Matrix Core Programming on CDNA](../sources/blogs/matrix-cores-cdna.md) `[source-blog]` arch:cdna2, cdna3, cdna4
- [AMD CDNA4 Instruction Set Architecture Reference](../sources/docs/cdna4-isa.md) `[source-doc]` arch:cdna4
- [AMD Instinct MI350 Series Architecture Overview](../sources/docs/cdna4-whitepaper.md) `[source-doc]` arch:cdna4
- [ROCm Flash Attention Repository](../sources/docs/flash-attention-rocm.md) `[source-doc]` arch:cdna2, cdna3, cdna4
- [MFMA Matrix Core (CDNA1–CDNA4)](../wiki/hardware/mfma-matrix-core.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [Scaled MFMA (CDNA4 Block-Scaled Matrix Operations)](../wiki/hardware/scaled-mfma.md) `[wiki-hardware]` arch:cdna4
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [GEMM Implementation on AMD CDNA](../wiki/kernels/gemm-rocm.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide](../wiki/migration/cdna3-to-cdna4.md) `[wiki-migration]` arch:cdna3, cdna4
- [Fix grouped conv bwd data wmma check](../sources/prs/composable_kernel/PR-3562.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Disable ActiveWorkgroupsPerCU for different arch in wmma kernels](../sources/prs/composable_kernel/PR-3566.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Remove code duplications in batched gemm wmma](../sources/prs/composable_kernel/PR-3580.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [WMMA grouped conv fwd large tensor extra flavors](../sources/prs/composable_kernel/PR-3582.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] Add reflection for wmma and bwd weight instances to ck builder reflection](../sources/prs/composable_kernel/PR-3592.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [WMMA grouped conv fwd large tensor bias bnorm clamp](../sources/prs/composable_kernel/PR-3595.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Add support to fp16 + compute fp16 and bf16 + compute bf16 contractions](../sources/prs/composable_kernel/PR-3598.md) `[source-pr]` arch:rdna4, rdna3
- [Remove code duplications in batched gemm (multi D) gemm (multi D) wmma](../sources/prs/composable_kernel/PR-3617.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [fix: AMD gfx1201 (RDNA4/ROCm) — INT8 Triton f32 MFMA, LTX Video device fix, validate_settings KeyError](../sources/prs/hipblaslt/PR-1822.md) `[source-pr]` arch:rdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) `[source-pr]` arch:cdna4
- [[CK Tile] Wavelet gemm pipeline for conv fwd](../sources/prs/hipblaslt/PR-7196.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[tensile] gfx12 assembly compatibility](../sources/prs/hipblaslt/PR-7655.md) `[source-pr]` arch:rdna4
- [rocWMMA: add gfx1032 (RDNA2) support with software WMMA fallback](../sources/prs/hipblaslt/PR-8209.md) `[source-pr]` arch:rdna2
- [[AMD] Restrict BlockPingPong scheduling for loop-variant masked loads](../sources/prs/hipblaslt/PR-10585.md) `[source-pr]` arch:cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [LDS Double Buffering](../wiki/techniques/double-buffering.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [Register Tiling for MFMA Kernels](../wiki/techniques/register-tiling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4

## scaled-mfma (25 pages)

- [AMD CDNA4 Instruction Set Architecture Reference](../sources/docs/cdna4-isa.md) `[source-doc]` arch:cdna4
- [AMD Instinct MI350 Series Architecture Overview](../sources/docs/cdna4-whitepaper.md) `[source-doc]` arch:cdna4
- [Scaled MFMA (CDNA4 Block-Scaled Matrix Operations)](../wiki/hardware/scaled-mfma.md) `[wiki-hardware]` arch:cdna4
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide](../wiki/migration/cdna3-to-cdna4.md) `[wiki-migration]` arch:cdna3, cdna4
- [[Triton] batched_gemm_a16wfp4 (gfx950): fuse dot_scaled accumulator; branchless mxfp4 quant; tune small-N configs](../sources/prs/hipblaslt/PR-3058.md) `[source-pr]` arch:cdna4
- [[FLYDSL MOE] mixed_moe + moe_gemm_2stage: fx internal-types cleanup (ASM-identical)](../sources/prs/hipblaslt/PR-3450.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[PERF] MXFP4 (a4w4) MoE backend for gfx950](../sources/prs/hipblaslt/PR-3470.md) `[source-pr]` arch:cdna4
- [Mx fp6 flatmm](../sources/prs/composable_kernel/PR-3601.md) `[source-pr]` arch:cdna2, cdna3, cdna4
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

## wavefront (10 pages)

- [AMD GCN Assembly Cross-Lane Operations](../sources/blogs/gcn-cross-lane.md) `[source-blog]` arch:cdna1, cdna2, cdna3, cdna4
- [Wavefront (64-thread execution unit)](../wiki/hardware/wavefront.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [Wavefront Reduction using DPP](../wiki/techniques/wave-reduction.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3
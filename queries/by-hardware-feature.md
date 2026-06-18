# Index: By Hardware Feature


## block-scale (24 pages)

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

## bpermute (5 pages)

- [AMD GCN Assembly Cross-Lane Operations](../sources/blogs/gcn-cross-lane.md) `[source-blog]` arch:cdna1, cdna2, cdna3, cdna4
- [DPP — Data-Parallel Primitives (Cross-Lane Operations)](../wiki/hardware/dpp-cross-lane.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [KV Cache Paged Attention on ROCm](../wiki/kernels/kv-cache-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Reduction Tree](../wiki/patterns/reduction-tree.md) `[wiki-pattern]` arch:cdna1, cdna2, cdna3, cdna4

## compute-unit (6 pages)

- [Compute Unit (CU) Microarchitecture](../wiki/hardware/compute-unit.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [XCD Chiplet Architecture](../wiki/hardware/xcd-chiplet.md) `[wiki-hardware]` arch:cdna3
- [Migration Guide: CDNA2 to CDNA3 Architecture](../wiki/migration/cdna2-to-cdna3.md) `[wiki-migration]` arch:cdna2, cdna3
- [Wavefront Occupancy Tuning](../wiki/techniques/occupancy-tuning.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [SGPR and Scalar Unit Optimization](../wiki/techniques/sgpr-scalar-unit.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [Multi-Wavefront Scheduling Strategies](../wiki/techniques/wavefront-scheduling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4

## dpp (9 pages)

- [AMD GCN Assembly Cross-Lane Operations](../sources/blogs/gcn-cross-lane.md) `[source-blog]` arch:cdna1, cdna2, cdna3, cdna4
- [DPP — Data-Parallel Primitives (Cross-Lane Operations)](../wiki/hardware/dpp-cross-lane.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Parallel Prefix Sum (Scan) on ROCm](../wiki/kernels/prefix-sum-scan.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3, cdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Reduction Tree](../wiki/patterns/reduction-tree.md) `[wiki-pattern]` arch:cdna1, cdna2, cdna3, cdna4
- [Cross-Lane Communication with DPP (Warp Shuffle Equivalent)](../wiki/techniques/warp-shuffle-dpp.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Wavefront Reduction using DPP](../wiki/techniques/wave-reduction.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3

## dual-cma (7 pages)

- [Matrix Core Programming on CDNA](../sources/blogs/matrix-cores-cdna.md) `[source-blog]` arch:cdna2, cdna3, cdna4
- [AMD CDNA4 Instruction Set Architecture Reference](../sources/docs/cdna4-isa.md) `[source-doc]` arch:cdna4
- [AMD Instinct MI350 Series Architecture Overview](../sources/docs/cdna4-whitepaper.md) `[source-doc]` arch:cdna4
- [Dual CMA (Compute Matrix Array) Engines in CDNA4](../wiki/hardware/dual-cma.md) `[wiki-hardware]` arch:cdna3, cdna4
- [MFMA Matrix Core (CDNA1–CDNA4)](../wiki/hardware/mfma-matrix-core.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [XCD Chiplet Architecture](../wiki/hardware/xcd-chiplet.md) `[wiki-hardware]` arch:cdna3
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4

## gws (4 pages)

- [GWS — Global Wave Sync](../wiki/hardware/gws.md) `[wiki-hardware]` arch:cdna2, cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Kernel Launch Overhead Optimization](../wiki/techniques/kernel-launch-overhead.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4

## lds (52 pages)

- [Compute Unit (CU) Microarchitecture](../wiki/hardware/compute-unit.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
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
- [Efficient Histogram Computation on ROCm](../wiki/kernels/histogram-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [KV Cache Paged Attention on ROCm](../wiki/kernels/kv-cache-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Parallel Prefix Sum (Scan) on ROCm](../wiki/kernels/prefix-sum-scan.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3, cdna4
- [Reduction Kernels on ROCm](../wiki/kernels/reduction-rocm.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Cooperative Loading](../wiki/patterns/cooperative-loading.md) `[wiki-pattern]` arch:cdna2, cdna3, cdna4
- [Reduction Tree](../wiki/patterns/reduction-tree.md) `[wiki-pattern]` arch:cdna1, cdna2, cdna3, cdna4
- [Wavefront Specialization (Warp Specialization)](../wiki/patterns/warp-specialization.md) `[wiki-pattern]` arch:cdna2, cdna3, cdna4
- [Refactor weight preshuffled pipeline and unify LDS buffer management API](../sources/prs/composable_kernel/PR-3493.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK TILE] Fix grouped conv kernels splitk and double lds](../sources/prs/composable_kernel/PR-3527.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Enable StoreSwapAddr and LDS > 64K for SPMM](../sources/prs/hipblaslt/PR-1968.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [fix lds padding vgpr alignment](../sources/prs/hipblaslt/PR-1981.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Enabling LDS>64K for LocalReadVALU](../sources/prs/hipblaslt/PR-2009.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [ LDS>64K enablement for LSU](../sources/prs/hipblaslt/PR-2015.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Fix device LDS size](../sources/prs/hipblaslt/PR-2035.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Remove an assertion related to LDS transpose](../sources/prs/hipblaslt/PR-2141.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Modify lds predicates](../sources/prs/rocWMMA/PR-298.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) `[source-pr]` arch:cdna4
- [# [TensileLite] Decouple MXFP8 scale DepthU from data DepthU (`ScaleDepthURatio`)](../sources/prs/hipblaslt/PR-7767.md) `[source-pr]` arch:cdna4
- [[AIROCMLIR-798] Add LDS usage estimate CAPI function](../sources/prs/hipblaslt/PR-2400.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Add option for larger LDS vecSize](../sources/prs/triton/PR-476.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [prune LDS usage for the new pipeliner](../sources/prs/triton/PR-600.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[tune_gemm] Update the filter for LDS usage for stream-pipelineV2](../sources/prs/triton/PR-661.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Query lds size for tune_gemm and tune_streamk](../sources/prs/triton/PR-680.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [MoE wvSplitK_int4: CU-count grid + skip duplicate MatA to LDS + gfx1151 N=1 K<1024 retune](../sources/prs/vllm/PR-920.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [triton_unified_attention: fix LDS overflow on gfx11 introduced by num_stages=3 retune](../sources/prs/vllm/PR-958.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)](../wiki/techniques/async-copy-lds.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [HIP Atomic Operations and Contention Reduction](../wiki/techniques/atomic-operations-hip.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [合并内存访问模式 (Coalesced Memory Access Patterns)](../wiki/techniques/coalesced-memory.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [LDS Double Buffering](../wiki/techniques/double-buffering.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Direct Read](../wiki/techniques/lds-direct-read.md) `[wiki-technique]` arch:cdna3, cdna4
- [Wavefront Occupancy Tuning](../wiki/techniques/occupancy-tuning.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [ROCm Profiling and Performance Analysis (rocprof, Omniperf)](../wiki/techniques/rocm-profiling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [LDS Address Swizzling](../wiki/techniques/swizzling.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3
- [Vectorized Global Memory Loads](../wiki/techniques/vectorized-loads.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4

## lds-transpose (6 pages)

- [AMD CDNA4 Instruction Set Architecture Reference](../sources/docs/cdna4-isa.md) `[source-doc]` arch:cdna4
- [AMD Instinct MI350 Series Architecture Overview](../sources/docs/cdna4-whitepaper.md) `[source-doc]` arch:cdna4
- [LDS — Local Data Share](../wiki/hardware/lds.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Read-with-Transpose (CDNA4)](../wiki/hardware/lds-transpose.md) `[wiki-hardware]` arch:cdna4
- [CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide](../wiki/migration/cdna3-to-cdna4.md) `[wiki-migration]` arch:cdna3, cdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) `[source-pr]` arch:cdna4

## mfma (57 pages)

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
- [GEMM Implementation on AMD CDNA](../wiki/kernels/gemm-rocm.md) `[wiki-kernel]` arch:cdna1, cdna2, cdna3
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [W8A8 Quantized GEMM](../wiki/kernels/quantized-gemm-w8a8.md) `[wiki-kernel]` arch:cdna2, cdna3
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [CDNA3 (MI300X) → CDNA4 (MI350X) Migration Guide](../wiki/migration/cdna3-to-cdna4.md) `[wiki-migration]` arch:cdna3, cdna4
- [Triton CUDA to ROCm Migration Guide](../wiki/migration/triton-cuda-to-rocm.md) `[wiki-migration]` arch:cdna2, cdna3, cdna4
- [Compute-Bound Optimization Patterns (算力密集优化模式)](../wiki/patterns/compute-bound-optimization.md) `[wiki-pattern]` arch:cdna2, cdna3, cdna4
- [add three I8 MFMA instructions into Stream-K I8 test for gfx950](../sources/prs/hipblaslt/PR-2105.md) `[source-pr]` arch:cdna4
- [issueLatency for MFMA](../sources/prs/hipblaslt/PR-2185.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Refine the method for caculating sparse mfma offset at tile section](../sources/prs/hipblaslt/PR-2195.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [update I8II gfx950 liblogic to use new gfx950 mfma instructions](../sources/prs/hipblaslt/PR-2237.md) `[source-pr]` arch:cdna4
- [update I8I8I liblogic to use new gfx950 mfma instructions](../sources/prs/hipblaslt/PR-2238.md) `[source-pr]` arch:cdna4
- [fix: AMD gfx1201 (RDNA4/ROCm) — INT8 Triton f32 MFMA, LTX Video device fix, validate_settings KeyError](../sources/prs/hipblaslt/PR-1822.md) `[source-pr]` arch:rdna4
- [Implement static unrolling in WMMA and MFMA](../sources/prs/rocWMMA/PR-497.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Fix f8 mfma implementation](../sources/prs/rocWMMA/PR-517.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[CK_TILE] Enable full transpose layout support for MX GEMM pipeline](../sources/prs/hipblaslt/PR-5813.md) `[source-pr]` arch:cdna4
- [[CK Tile] Wavelet gemm pipeline for conv fwd](../sources/prs/hipblaslt/PR-7196.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[tensile] gfx12 assembly compatibility](../sources/prs/hipblaslt/PR-7655.md) `[source-pr]` arch:rdna4
- [rocWMMA: add gfx1032 (RDNA2) support with software WMMA fallback](../sources/prs/hipblaslt/PR-8209.md) `[source-pr]` arch:rdna2
- [[AMD] Restrict BlockPingPong scheduling for loop-variant masked loads](../sources/prs/hipblaslt/PR-10585.md) `[source-pr]` arch:cdna4
- [enable layout conversion from mfma to dot_op for mfma16.](../sources/prs/triton/PR-453.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[MFMA] Support 64x4 and 4x64 tile size](../sources/prs/triton/PR-469.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[MFMA][Test] Add scripts generating mfma related lit tests](../sources/prs/triton/PR-472.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[MFMA] Move operand casts to AccelerateMatMul pass](../sources/prs/triton/PR-477.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[MFMA] Cleanup mfma pipeline](../sources/prs/triton/PR-479.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[MFMA] Remove redundant checks](../sources/prs/triton/PR-496.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Revert "[MFMA] Move operand casts to AccelerateMatMul pass"](../sources/prs/triton/PR-500.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [[ReductionOp][MFMA] fix reduction of mfma64x4 layout ](../sources/prs/triton/PR-526.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [Adding more support for machine model cycles per dot/mfma](../sources/prs/triton/PR-879.md) `[source-pr]` arch:cdna2, cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [LDS Double Buffering](../wiki/techniques/double-buffering.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Direct Read](../wiki/techniques/lds-direct-read.md) `[wiki-technique]` arch:cdna3, cdna4
- [CDNA4 FP8 Scaled MFMA](../wiki/techniques/mfma-fp8-cdna4.md) `[wiki-technique]` arch:cdna4
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [Mixed Precision Computing in HIP](../wiki/techniques/mixed-precision-hip.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Register Tiling for MFMA Kernels](../wiki/techniques/register-tiling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [ROCm Profiling and Performance Analysis (rocprof, Omniperf)](../wiki/techniques/rocm-profiling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)](../wiki/techniques/vgpr-pressure.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [XDLOPS 底层编程 (XDLOPS Low-level Programming)](../wiki/techniques/xdlops-programming.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3

## scaled-mfma (26 pages)

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
- [CDNA4 FP8 Scaled MFMA](../wiki/techniques/mfma-fp8-cdna4.md) `[wiki-technique]` arch:cdna4

## wavefront (20 pages)

- [AMD GCN Assembly Cross-Lane Operations](../sources/blogs/gcn-cross-lane.md) `[source-blog]` arch:cdna1, cdna2, cdna3, cdna4
- [Compute Unit (CU) Microarchitecture](../wiki/hardware/compute-unit.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [Wavefront (64-thread execution unit)](../wiki/hardware/wavefront.md) `[wiki-hardware]` arch:cdna1, cdna2, cdna3, cdna4
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [KV Cache Paged Attention on ROCm](../wiki/kernels/kv-cache-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) `[wiki-kernel]` arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) `[wiki-kernel]` arch:cdna3, cdna4
- [Wavefront Specialization (Warp Specialization)](../wiki/patterns/warp-specialization.md) `[wiki-pattern]` arch:cdna2, cdna3, cdna4
- [异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)](../wiki/techniques/async-copy-lds.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [合并内存访问模式 (Coalesced Memory Access Patterns)](../wiki/techniques/coalesced-memory.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Wavefront Occupancy Tuning](../wiki/techniques/occupancy-tuning.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [SGPR and Scalar Unit Optimization](../wiki/techniques/sgpr-scalar-unit.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3, cdna4
- [Cross-Lane Communication with DPP (Warp Shuffle Equivalent)](../wiki/techniques/warp-shuffle-dpp.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [Wavefront Reduction using DPP](../wiki/techniques/wave-reduction.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3
- [Multi-Wavefront Scheduling Strategies](../wiki/techniques/wavefront-scheduling.md) `[wiki-technique]` arch:cdna2, cdna3, cdna4
- [XDLOPS 底层编程 (XDLOPS Low-level Programming)](../wiki/techniques/xdlops-programming.md) `[wiki-technique]` arch:cdna1, cdna2, cdna3
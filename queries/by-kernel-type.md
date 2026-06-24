# Index: By Kernel Type


## activation (7 pages)

- [Activation Kernels (SiLU, GELU, SwiGLU)](../wiki/kernels/activation-kernels.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md) conf:source-reported arch:cdna3, cdna4
- [Grid-Stride Loop](../wiki/patterns/grid-stride-loop.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Memory-Bound Optimization Patterns](../wiki/patterns/memory-bound-optimization.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md) conf:source-reported arch:cdna4
- [[PR 4/7] Multi-arch ROCm kernel support with runtime optimization](../sources/prs/sglang/PR-27745.md) conf:source-reported arch:cdna3, cdna4
- [[AMD] Fuse shared-expert sigmoid + bf16->fp32 cast into the MoE append kernel (3 kernels -> 1)](../sources/prs/sglang/PR-28658.md) conf:source-reported arch:cdna3, cdna4

## attention (106 pages)

- [ROCm FlashAttention Performance Notes](../sources/blogs/flash-attention-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [ROCm Flash Attention Repository](../sources/docs/flash-attention-rocm.md) conf:verified arch:cdna2, cdna3, cdna4
- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Flash Decoding on ROCm](../wiki/kernels/flash-decoding-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md) conf:source-reported arch:cdna3, cdna4
- [Fused Attention Bias and Causal Masking](../wiki/kernels/fused-attention-bias.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [KV Cache Paged Attention on ROCm](../wiki/kernels/kv-cache-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Multi-Head Latent Attention (MLA) on ROCm](../wiki/kernels/mla-attention-rocm.md) conf:source-reported arch:cdna2, cdna3
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) conf:source-reported arch:cdna3, cdna4
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md) conf:source-reported arch:rdna3, rdna4
- [Speculative Decoding and Tree Attention on ROCm](../wiki/kernels/speculative-decoding-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) conf:source-reported arch:cdna3, cdna4
- [Cooperative Loading](../wiki/patterns/cooperative-loading.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [生产者-消费者流水线 (Producer-Consumer Pipeline)](../wiki/patterns/producer-consumer-pipeline.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Tile Quantization and Dequantization](../wiki/patterns/tile-quantize-dequant.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "[Triton] Declare triton>=3.6.0 dependency "](../sources/prs/hipblaslt/PR-3272.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[feat] FP8 (DeepSeek-V4 layout) sparse paged prefill attention](../sources/prs/hipblaslt/PR-3583.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add deterministic algorithm support to v3::flash::attn_options](../sources/prs/aotriton/PR-134.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add FP32 and Bias to fulfill the functionalities required by `torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION`](../sources/prs/aotriton/PR-22.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add varlen support to AOTriton's Flash Attention](../sources/prs/aotriton/PR-31.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Re-implement Causal Masks with Windowed Attention](../sources/prs/aotriton/PR-96.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Tile] Fix FMHA LSE calculation and potential division by zero](../sources/prs/composable_kernel/PR-3326.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [ Fp8 block scale quantization for fmha  fwd](../sources/prs/composable_kernel/PR-3330.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [fp8 fmha async pipeline](../sources/prs/composable_kernel/PR-3339.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Add sparse attention VSA](../sources/prs/composable_kernel/PR-3341.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Add logits soft-capping support for FAv3 (WIP)](../sources/prs/composable_kernel/PR-3355.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add attention sink support for FMHA FWD](../sources/prs/composable_kernel/PR-3368.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Fix Python 3.8 compatibility in fmha codegen](../sources/prs/composable_kernel/PR-3388.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix FMHA fp8 hdim=64 incorrect result in MI200](../sources/prs/composable_kernel/PR-3423.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Add FP8 support for batch_prefill kernel](../sources/prs/composable_kernel/PR-3425.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FMHA] Batch Prefill Support Improvements:  Change KV Cache Layout & Large Page Size Support](../sources/prs/composable_kernel/PR-3442.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] FMHA Ignore BWD Failed Cases in Smoke Test](../sources/prs/composable_kernel/PR-3480.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Align FMHA BWD Reference with Kernel Implementation](../sources/prs/composable_kernel/PR-3486.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Enable gpt-oss sink](../sources/prs/composable_kernel/PR-3490.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FMHA] Support page_size=1 (linear layout) in batch prefill pipeline](../sources/prs/composable_kernel/PR-3545.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FMHA] Enable page size 16 for batch prefill kernel](../sources/prs/composable_kernel/PR-3568.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Add new tile size for async](../sources/prs/composable_kernel/PR-3586.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA] Revert new tile size for async (#3586)"](../sources/prs/composable_kernel/PR-3613.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix Int32 Overflow in Deterministic FMHA BWD](../sources/prs/composable_kernel/PR-3615.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE][FMHA]Add new tile size for async](../sources/prs/composable_kernel/PR-3623.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert " Fp8 block scale quantization for fmha  fwd"](../sources/prs/composable_kernel/PR-3633.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "Revert " Fp8 block scale quantization for fmha  fwd""](../sources/prs/composable_kernel/PR-3635.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Add FP8 KV_BLOCKSCALE support for batch prefill](../sources/prs/composable_kernel/PR-3696.md) conf:source-reported arch:cdna3, cdna4
- [[CK_TILE] Sparge attention](../sources/prs/composable_kernel/PR-3727.md) conf:source-reported arch:cdna3, cdna4
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md) conf:source-reported arch:cdna4
- [[ck_tile/fmha] Fix sink un-mask under right-window and emit fp8bf16 batch_prefill sink kernels](../sources/prs/composable_kernel/PR-3732.md) conf:source-reported arch:cdna3, cdna4
- [[CK_TILE] fix(fmha): clamp paged KV lookups in batch prefill](../sources/prs/composable_kernel/PR-3733.md) conf:source-reported arch:cdna3, cdna4
- [[CK Tile] Prepare mixed batch-prefill FP8 KV contract](../sources/prs/composable_kernel/PR-3745.md) conf:source-reported arch:cdna3, cdna4
- [Support page attention in mha_varlen_fwd](../sources/prs/flash-attention/PR-103.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [head, seq, batch grid order for triton flash attention bwd.](../sources/prs/flash-attention/PR-141.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix NaN for FMHA BWD When seq_q=0](../sources/prs/flash-attention/PR-179.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Update CK and enable RDNA backward](../sources/prs/flash-attention/PR-184.md) conf:source-reported arch:cdna3, rdna3, rdna4
- [Ck tile/flash attention](../sources/prs/flash-attention/PR-61.md) conf:source-reported arch:cdna2, cdna3
- [Integrate ck tile backward](../sources/prs/flash-attention/PR-65.md) conf:source-reported arch:cdna2, cdna3
- [Use same python as build flash-attn to generate ck kernel](../sources/prs/flash-attention/PR-66.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Improve FMHA bwd](../sources/prs/flash-attention/PR-70.md) conf:source-reported arch:cdna2, cdna3
- [Ck tile/kvcache](../sources/prs/flash-attention/PR-74.md) conf:source-reported arch:cdna2, cdna3
- [[CK_TILE] Fix fmha fwd splitkv block table read out-of-bound](../sources/prs/flash-attention/PR-98.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[WIP] test: cut unit-test CI wall time](../sources/prs/hipblaslt/PR-3601.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [AMD - gpt-oss vllm mxfp4: AITER tuning + n-gram spec decode + server …](../sources/prs/hipblaslt/PR-1657.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [feat: shared-engine refactor + Nemotron-Nano-30B GB10 prefill/decode optimizations (~5×)](../sources/prs/hipblaslt/PR-20.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm] Add Triton MLA decode + prefill kernels for MI300X](../sources/prs/hipblaslt/PR-377.md) conf:source-reported arch:cdna3
- [Feat/fused ulysses qkv projection](../sources/prs/hipblaslt/PR-805.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[gfx1201] Mistral-3 + Qwen3-8B-FP8 on RDNA4 via native triton attention](../sources/prs/hipblaslt/PR-811.md) conf:source-reported arch:rdna4
- [vulkan: Intel Xe flash attention, GEMM optimizations, and optional weight compression (Xe-LPG Plus/Xe2/Xe3) [MEGA PR]](../sources/prs/hipblaslt/PR-24408.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm] Enable native AsyncTP](../sources/prs/hipblaslt/PR-177961.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Add Tile Engine -> Dispatcher bridge for GEMM](../sources/prs/hipblaslt/PR-8123.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md) conf:source-reported arch:rdna4
- [[AIROCMLIR-798] Add LDS usage estimate CAPI function](../sources/prs/hipblaslt/PR-2400.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AMD/gfx950] FlyDSL kgather diagnostic backend for DSv4 sparse FP8 MLA decode](../sources/prs/hipblaslt/PR-13.md) conf:source-reported arch:cdna4
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md) conf:source-reported arch:cdna4
- [[minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra](../sources/prs/sglang/PR-28715.md) conf:source-reported arch:cdna3, cdna4
- [[FA-qk-fp8] Add fp8 FA to 06-fused-attention-fwd-transV.py](../sources/prs/triton/PR-475.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add Paged Attention Decode Kernel](../sources/prs/triton/PR-718.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Tianxing/rope latent attention](../sources/prs/triton/PR-731.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix flash attention ops calculation](../sources/prs/triton/PR-758.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix the bwd Mode in flash-attention.py](../sources/prs/triton/PR-772.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [fix some syntax errors of the kernel in 06-fused-attention-transV.py](../sources/prs/triton/PR-876.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][Kernel] ViT prefill attention: split-D head_dim + re-tuned tile on gfx1151](../sources/prs/vllm/PR-1000.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm] Faster Custom Paged Attention kernels](../sources/prs/vllm/PR-12348.md) conf:source-reported arch:cdna3
- [[Bugfix][ROCm] Fix OOB query read in paged_attention_rocm for head_size < 128](../sources/prs/vllm/PR-40745.md) conf:source-reported arch:cdna3, cdna4
- [[Kernel][AMD] Optimize GatedDeltaNet FLA prefill kernels on MI300X](../sources/prs/hipblaslt/PR-41446.md) conf:source-reported arch:cdna3
- [[ROCm][AITER] Use pre-shuffled FP8 GEMM for Quark per-channel attention weights](../sources/prs/hipblaslt/PR-44626.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Attention Backend] add HPC-Ops Attention backend](../sources/prs/vllm/PR-46020.md) conf:source-reported arch:cdna3, cdna4
- [[Attention][DSA] support dcp for FLASHINFER_MLA_SPARSE](../sources/prs/vllm/PR-46076.md) conf:source-reported arch:cdna3, cdna4
- [[ROCm][CI] Only require q_scale==1.0 for fp8 query in RocmAttention](../sources/prs/vllm/PR-46148.md) conf:source-reported arch:cdna3, cdna4
- [Fix attention fp8 output fusion for split attention path in v1](../sources/prs/vllm/PR-569.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FEAT] Use `flash-attn` in ViT instead of `torch.sdpa`](../sources/prs/vllm/PR-610.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][Perf] Enable shuffle kv cache layout and assembly paged attention kernel for AiterFlashAttentionBackend](../sources/prs/vllm/PR-836.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Pass positions to attention function in llama.py](../sources/prs/vllm/PR-854.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [CI test for automated attention benchmarking suite](../sources/prs/vllm/PR-897.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix Triton attention shared memory overflow on Navi for head_size > 256](../sources/prs/vllm/PR-919.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[profiling] Capture attention call shapes with torch.profile](../sources/prs/vllm/PR-996.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)](../wiki/techniques/async-copy-lds.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) conf:verified arch:cdna1, cdna2, cdna3, cdna4
- [Non-Temporal Store (L2 Cache Bypass)](../wiki/techniques/buffer-store-nt.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [合并内存访问模式 (Coalesced Memory Access Patterns)](../wiki/techniques/coalesced-memory.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [LDS Double Buffering](../wiki/techniques/double-buffering.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [PR Insight: triton #475 - FP8 QK Flash Attention Integration](../wiki/techniques/pr-triton-475.md) conf:inferred arch:cdna2, cdna3, cdna4
- [Register Tiling for MFMA Kernels](../wiki/techniques/register-tiling.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [SGPR and Scalar Unit Optimization](../wiki/techniques/sgpr-scalar-unit.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Vectorized Global Memory Loads](../wiki/techniques/vectorized-loads.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)](../wiki/techniques/vgpr-pressure.md) conf:source-reported arch:cdna2, cdna3, cdna4

## cast-transpose (2 pages)

- [Full MXFP4 Training Recipe](../sources/prs/transformerengine/PR-537.md) conf:source-reported arch:cdna4
- [gfx1250 swizzle_xor changes for FP4](../sources/prs/transformerengine/PR-571.md) conf:source-reported arch:rdna4

## collective (1 pages)

- [Fix Qwen MoE precision issue with PP and all-reduce fusion](../sources/prs/sglang/PR-28619.md) conf:source-reported arch:cdna3, cdna4

## communication (1 pages)

- [RCCL Multi-GPU Communication](../wiki/techniques/multi-gpu-rccl.md) conf:source-reported arch:cdna2, cdna3, cdna4

## compute-bound (2 pages)

- [Flat vs Buffer Addressing Modes](../wiki/techniques/flat-addressing.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Scratch Memory Spill Management](../wiki/techniques/scratch-memory.md) conf:source-reported arch:cdna2, cdna3, cdna4

## conv (55 pages)

- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Compute-Bound Optimization Patterns (算力密集优化模式)](../wiki/patterns/compute-bound-optimization.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Cooperative Loading](../wiki/patterns/cooperative-loading.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Latency Hiding (延迟隐藏)](../wiki/patterns/latency-hiding.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[tests] Unit tests non-tunable conv asm solvers](../sources/prs/MIOpen/PR-3494.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[BUG][TESTS] Add unit tests for conv asm 1x1u bwdwrw3x3 3x3u 1x1uv2](../sources/prs/MIOpen/PR-3716.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Conv] group conv bias active ](../sources/prs/MIOpen/PR-3775.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[BUG] [CONV] Fix incorrect stride calculation when w=1/h=1 in MISA solvers](../sources/prs/MIOpen/PR-3786.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[GCBA] grouped conv + bias + activ NHWC/NDHWC 3d](../sources/prs/MIOpen/PR-3802.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add support for 2D group conv+activ fusion](../sources/prs/MIOpen/PR-3807.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add 3d support to group conv activ](../sources/prs/MIOpen/PR-3812.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Create samples directory with inital grouped conv + bias +activ sample](../sources/prs/MIOpen/PR-3820.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "[BUG] [CONV] Fix incorrect stride calculation when w=1/h=1 in…](../sources/prs/MIOpen/PR-3836.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[DRIVER] Fix inconsistent and incorrect conv stats prints](../sources/prs/MIOpen/PR-3854.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[BUG] [CONV] Fix incorrect stride calculation in MISA solvers](../sources/prs/MIOpen/PR-3867.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [3D conv heuristics (KTN part)](../sources/prs/MIOpen/PR-3918.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[TRITON] Conv Kernels First Commit to AITER](../sources/prs/hipblaslt/PR-2886.md) conf:source-reported arch:rdna4
- [[CK_TILE] Add indexing optimizations for conv bwd data](../sources/prs/composable_kernel/PR-3309.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Add indexing optimizations for conv bwd weight](../sources/prs/composable_kernel/PR-3321.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] fix enforcing fixed vectorsizes for ck tile conv](../sources/prs/composable_kernel/PR-3344.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Add splitk support to ck tile conv bwd data](../sources/prs/composable_kernel/PR-3353.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Minor splitk bugfix for gemms and conv](../sources/prs/composable_kernel/PR-3387.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Added large tensor support for grouped conv fwd wmma](../sources/prs/composable_kernel/PR-3437.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Split grouped conv fwd instances](../sources/prs/composable_kernel/PR-3449.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Adding support for scale and bilinear ops for WMMA grouped conv fwd ](../sources/prs/composable_kernel/PR-3450.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Improve XDL to WMMA porting for grouped conv fwd](../sources/prs/composable_kernel/PR-3456.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Post-merge cleanup for WMMA grouped conv fwd](../sources/prs/composable_kernel/PR-3468.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Replace grouped conv bwd wei wmmaV3 bilin/scale bf16f32bf16 support with bf16bf16bf16](../sources/prs/composable_kernel/PR-3470.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix jenkinsfile for large tensor conv test](../sources/prs/composable_kernel/PR-3478.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix grouped conv fwd wmma porting](../sources/prs/composable_kernel/PR-3479.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix grouped conv wrw kernels names](../sources/prs/composable_kernel/PR-3494.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] Instance traits for conv bwd weight algorithms ](../sources/prs/composable_kernel/PR-3498.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] Integrate reference conv with testing](../sources/prs/composable_kernel/PR-3511.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix large tensor grouped conv bwd data test](../sources/prs/composable_kernel/PR-3513.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] Add grouped conv fwd ck tile profiler](../sources/prs/composable_kernel/PR-3518.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE] Fix grouped conv kernels splitk and double lds](../sources/prs/composable_kernel/PR-3527.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Adding remaining conv, dynamic_op, and scaleadd_scaleadd_relu flavors for grouped conv fwd](../sources/prs/composable_kernel/PR-3529.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK tests] Extend conv GPU reference](../sources/prs/composable_kernel/PR-3539.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix grouped conv bwd data wmma check](../sources/prs/composable_kernel/PR-3562.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [WMMA grouped conv fwd large tensor extra flavors](../sources/prs/composable_kernel/PR-3582.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [WMMA grouped conv fwd large tensor bias bnorm clamp](../sources/prs/composable_kernel/PR-3595.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] Replace reference conv with old ck implementation](../sources/prs/composable_kernel/PR-3604.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_BUILDER] conv bwd weight testing](../sources/prs/composable_kernel/PR-3618.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE] Enable CK TILE Conv Fwd tests in CI and fix check_err](../sources/prs/composable_kernel/PR-3624.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Grouped conv fwd direct load vector=2](../sources/prs/composable_kernel/PR-3632.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Add new instances for merging multiple fwd conv groups into a single GEMM batch](../sources/prs/composable_kernel/PR-3639.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Grouped Conv Bwd Weight Direct Load](../sources/prs/composable_kernel/PR-3648.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Conv] Enable bwd weight splitk autodeduction with cap](../sources/prs/composable_kernel/PR-3656.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Enable Grouped Conv Tile Fwd Tests daily](../sources/prs/composable_kernel/PR-3680.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix path to ck tile conv fwd instance generator](../sources/prs/composable_kernel/PR-3699.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AIROCMLIR-71] Add gemm+gemm and conv+gemm support to quickTuningGen.py](../sources/prs/hipblaslt/PR-2262.md) conf:source-reported arch:cdna4
- [[CK Tile] Wavelet gemm pipeline for conv fwd](../sources/prs/hipblaslt/PR-7196.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Add Tile Engine -> Dispatcher bridge for GEMM](../sources/prs/hipblaslt/PR-8123.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AMD] Restrict BlockPingPong scheduling for loop-variant masked loads](../sources/prs/hipblaslt/PR-10585.md) conf:source-reported arch:cdna4
- [Vectorized Global Memory Loads](../wiki/techniques/vectorized-loads.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4

## convolution (4 pages)

- [[CK_TILE] Add support and tests for V6 pipeline in conv fwd](../sources/prs/composable_kernel/PR-3708.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Add support for large tensor index handling into conv bwd data WMMA](../sources/prs/hipblaslt/PR-8518.md) conf:source-reported arch:rdna3, rdna4
- [[CK DSL] conv heuristic: fix gemm_k_per_block, add K_per_C + log features, update all models to 101 features](../sources/prs/hipblaslt/PR-8620.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK][CK DSL] Pass vector sizes as arguments for implicit gemm](../sources/prs/hipblaslt/PR-8624.md) conf:source-reported arch:rdna3, rdna4, cdna3, cdna4

## custom-fusion (1 pages)

- [Tile Quantization and Dequantization](../wiki/patterns/tile-quantize-dequant.md) conf:source-reported arch:cdna2, cdna3, cdna4

## element-wise (2 pages)

- [Mixed Precision Computing in HIP](../wiki/techniques/mixed-precision-hip.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)](../wiki/techniques/vgpr-pressure.md) conf:source-reported arch:cdna2, cdna3, cdna4

## embedding (4 pages)

- [Embedding Lookup Kernel Optimization](../wiki/kernels/embedding-lookup.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Rotary Position Embedding (RoPE)](../wiki/kernels/rotary-embedding-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Memory-Bound Optimization Patterns](../wiki/patterns/memory-bound-optimization.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Scatter/Gather Memory Access Patterns](../wiki/patterns/scatter-gather.md) conf:source-reported arch:cdna2, cdna3, cdna4

## flash-attention (32 pages)

- [Flash Attention on ROCm](../wiki/kernels/flash-attention-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Flash Decoding on ROCm](../wiki/kernels/flash-decoding-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [FP8 FlashAttention on ROCm](../wiki/kernels/fp8-flash-attention-rocm.md) conf:source-reported arch:cdna3, cdna4
- [Fused Attention Bias and Causal Masking](../wiki/kernels/fused-attention-bias.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Paged Prefill Attention on ROCm](../wiki/kernels/paged-prefill-attention-rocm.md) conf:source-reported arch:cdna3, cdna4
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md) conf:source-reported arch:rdna3, rdna4
- [Triton FlashAttention on ROCm](../wiki/kernels/triton-flash-attention-rocm.md) conf:source-reported arch:cdna3, cdna4
- [Cooperative Loading](../wiki/patterns/cooperative-loading.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Latency Hiding (延迟隐藏)](../wiki/patterns/latency-hiding.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [生产者-消费者流水线 (Producer-Consumer Pipeline)](../wiki/patterns/producer-consumer-pipeline.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Wavefront Specialization (Warp Specialization)](../wiki/patterns/warp-specialization.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Add FP8 KV_BLOCKSCALE support for batch prefill](../sources/prs/composable_kernel/PR-3696.md) conf:source-reported arch:cdna3, cdna4
- [[CK_TILE] async trload for fmha 192/128 in mi355](../sources/prs/composable_kernel/PR-3729.md) conf:source-reported arch:cdna4
- [[ck_tile/fmha] Fix sink un-mask under right-window and emit fp8bf16 batch_prefill sink kernels](../sources/prs/composable_kernel/PR-3732.md) conf:source-reported arch:cdna3, cdna4
- [[CK_TILE] fix(fmha): clamp paged KV lookups in batch prefill](../sources/prs/composable_kernel/PR-3733.md) conf:source-reported arch:cdna3, cdna4
- [[CK Tile] Prepare mixed batch-prefill FP8 KV contract](../sources/prs/composable_kernel/PR-3745.md) conf:source-reported arch:cdna3, cdna4
- [[CK_TILE] Update CK and enable RDNA backward](../sources/prs/flash-attention/PR-184.md) conf:source-reported arch:cdna3, rdna3, rdna4
- [Ck tile/flash attention](../sources/prs/flash-attention/PR-61.md) conf:source-reported arch:cdna2, cdna3
- [Integrate ck tile backward](../sources/prs/flash-attention/PR-65.md) conf:source-reported arch:cdna2, cdna3
- [Improve FMHA bwd](../sources/prs/flash-attention/PR-70.md) conf:source-reported arch:cdna2, cdna3
- [Ck tile/kvcache](../sources/prs/flash-attention/PR-74.md) conf:source-reported arch:cdna2, cdna3
- [[CK_TILE] Use Unified Workspace for FMHA BWD](../sources/prs/flash-attention/PR-182.md) conf:? arch:cdna3
- [[CK_TILE] FMHA BWD: stream-async workspace prepare](../sources/prs/flash-attention/PR-183.md) conf:? arch:cdna3
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md) conf:source-reported arch:rdna4
- [[ROCm] Faster Custom Paged Attention kernels](../sources/prs/vllm/PR-12348.md) conf:source-reported arch:cdna3
- [Non-Temporal Store (L2 Cache Bypass)](../wiki/techniques/buffer-store-nt.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Unified Workspace Allocation for Flash Attention Backward](../wiki/techniques/pr-flash-attention-rocm-182.md) conf:verified arch:cdna3
- [LDS Direct Read](../wiki/techniques/lds-direct-read.md) conf:source-reported arch:cdna3, cdna4
- [PR Insight: triton #475 - FP8 QK Flash Attention Integration](../wiki/techniques/pr-triton-475.md) conf:inferred arch:cdna2, cdna3, cdna4
- [SGPR and Scalar Unit Optimization](../wiki/techniques/sgpr-scalar-unit.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Stream-Async Workspace Preparation via Host Callbacks](../wiki/techniques/pr-flash-attention-rocm-183.md) conf:verified arch:cdna3
- [Multi-Wavefront Scheduling Strategies](../wiki/techniques/wavefront-scheduling.md) conf:source-reported arch:cdna2, cdna3, cdna4

## gemm (237 pages)

- [Batched GEMM on ROCm](../wiki/kernels/batched-gemm-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [CK Tile GEMM on ROCm](../wiki/kernels/ck-tile-gemm-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [FP8 and Block-Scale GEMM on ROCm](../wiki/kernels/fp8-blockscale-gemm-rocm.md) conf:source-reported arch:cdna3, cdna4
- [MFMA GEMM on ROCm](../wiki/kernels/gemm-mfma-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [GEMM Implementation on AMD CDNA](../wiki/kernels/gemm-rocm.md) conf:verified arch:cdna1, cdna2, cdna3, rdna3, rdna4
- [hipBLASLt Fused GEMM and Quantization on ROCm](../wiki/kernels/hipblaslt-fused-gemm-rocm.md) conf:source-reported arch:cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [W4A16 Quantized GEMM on ROCm](../wiki/kernels/quantized-gemm-w4a16.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [W8A8 Quantized GEMM](../wiki/kernels/quantized-gemm-w8a8.md) conf:source-reported arch:cdna2, cdna3
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md) conf:source-reported arch:rdna3, rdna4
- [Sparse GEMM (SpMM) on ROCm](../wiki/kernels/sparse-gemm-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Stream-K and Split-K GEMM on ROCm](../wiki/kernels/streamk-splitk-gemm-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Compute-Bound Optimization Patterns (算力密集优化模式)](../wiki/patterns/compute-bound-optimization.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Cooperative Loading](../wiki/patterns/cooperative-loading.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Latency Hiding (延迟隐藏)](../wiki/patterns/latency-hiding.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [生产者-消费者流水线 (Producer-Consumer Pipeline)](../wiki/patterns/producer-consumer-pipeline.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Tile Quantization and Dequantization](../wiki/patterns/tile-quantize-dequant.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Wavefront Specialization (Warp Specialization)](../wiki/patterns/warp-specialization.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add unit tests for implicit gemm bwd/fwd xdlops solver](../sources/prs/MIOpen/PR-3521.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Remove CK's gemm and reduction libraries from MIOpen](../sources/prs/MIOpen/PR-3739.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[TRITON] Conv Kernels First Commit to AITER](../sources/prs/hipblaslt/PR-2886.md) conf:source-reported arch:rdna4
- [[Triton] batched_gemm_a16wfp4 (gfx950): fuse dot_scaled accumulator; branchless mxfp4 quant; tune small-N configs](../sources/prs/hipblaslt/PR-3058.md) conf:source-reported arch:cdna4
- [[TRITON] gfx1201: gemm_a8w8 tuning configs (Mistral-3 / Qwen3 shapes)](../sources/prs/hipblaslt/PR-3168.md) conf:source-reported arch:rdna4
- [Revert "[Triton] Declare triton>=3.6.0 dependency "](../sources/prs/hipblaslt/PR-3272.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add GLM-4.7-FP8 tuned/untuned BF16 GEMM configs (gfx950)](../sources/prs/hipblaslt/PR-3285.md) conf:source-reported arch:cdna4
- [gfx1201 gemm_a8w8: blockscale HIP→triton fallback + tuning configs (plain + blockscale_preshuffled)](../sources/prs/hipblaslt/PR-3343.md) conf:source-reported arch:rdna4
- [[gluon gemm_afp4wfp4] Fix data access pattern to remove redundant data loads](../sources/prs/hipblaslt/PR-3355.md) conf:source-reported arch:cdna4
- [Add GLM-5.1 FP8 blockscale GEMM/FMoE tunings for gfx942 (MI300X/MI325)](../sources/prs/hipblaslt/PR-3422.md) conf:source-reported arch:cdna3
- [[FLYDSL MOE] mixed_moe + moe_gemm_2stage: fx internal-types cleanup (ASM-identical)](../sources/prs/hipblaslt/PR-3450.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fused SplitK zero-init for FP8 a8w8 blockscale GEMMs (y_is_zeroed) + re-enable CKTile SplitK](../sources/prs/hipblaslt/PR-3457.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[PERF] MXFP4 (a4w4) MoE backend for gfx950](../sources/prs/hipblaslt/PR-3470.md) conf:source-reported arch:cdna4
- [Tune fused GEMM AFP4WFP4 A16W16 gfx950 config and add benchmark](../sources/prs/hipblaslt/PR-3642.md) conf:source-reported arch:cdna4
- [[CK_Tile] Enable PreshuffleB for 2d block scale Gemm](../sources/prs/composable_kernel/PR-3298.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Implement grouped gemm fastgelu for RDNA4](../sources/prs/composable_kernel/PR-3303.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Implement grouped gemm tile loop for RDNA4](../sources/prs/composable_kernel/PR-3304.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix Quant GEMM build](../sources/prs/composable_kernel/PR-3320.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Make CK TILE GEMM Aquant support block tile 128x128x128](../sources/prs/composable_kernel/PR-3325.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Disable the gfx90a on CK Tile Grouped GEMM](../sources/prs/composable_kernel/PR-3336.md) conf:source-reported arch:cdna2
- [[CK Tile] Grouped GEMM aquant mode and non-persistent kernel](../sources/prs/composable_kernel/PR-3337.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Support A/B Quantization in Blockscale GEMM](../sources/prs/composable_kernel/PR-3343.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE GEMM STREAMK] update identifier names according to the new code style](../sources/prs/composable_kernel/PR-3348.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Support more layouts for BQuant GEMM](../sources/prs/composable_kernel/PR-3349.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK-Tile] fixup codegen for tile engine ops gemm multid and gemm preshuffle](../sources/prs/composable_kernel/PR-3383.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Implement batched gemm add relu gemm add for rdna4](../sources/prs/composable_kernel/PR-3391.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Fix some inconsistencies with OverrideBDatatype in BQuant GEMM](../sources/prs/composable_kernel/PR-3394.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Grouped Gemm] Disable split-k kernel for split-k > 1 with non-contiguous strides](../sources/prs/composable_kernel/PR-3405.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Grouped Gemm] Fix workspace stride in two stage kernel](../sources/prs/composable_kernel/PR-3412.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Grouped gemm quant tensor layouts](../sources/prs/composable_kernel/PR-3414.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[TILE ENGINE] Restructure to Base class of GEMM](../sources/prs/composable_kernel/PR-3434.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_Tile] Support for various group sizes Preshuffle quant for 2d block scale gemm](../sources/prs/composable_kernel/PR-3445.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CKTILE] Support A/B Quantization in Blockscale Grouped Gemm](../sources/prs/composable_kernel/PR-3452.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_Tile]  Support for group size 128 for Preshuffle quant for 2d block scale gemm](../sources/prs/composable_kernel/PR-3462.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK grouped gemm] Fix grouped gemm two stage HasMainK0BlockLoop](../sources/prs/composable_kernel/PR-3466.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] add preshuffleB mode for ABQuant GEMM](../sources/prs/composable_kernel/PR-3495.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [fix mxfp8-gemm example failure](../sources/prs/composable_kernel/PR-3531.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Implement batched gemm bias permute for RDNA4](../sources/prs/composable_kernel/PR-3534.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE GEMM] Add bf8 support to tile engine streamk generator](../sources/prs/composable_kernel/PR-3543.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Batched gemm softmax gemm descriptor fix](../sources/prs/composable_kernel/PR-3564.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Remove code duplications in batched gemm wmma](../sources/prs/composable_kernel/PR-3580.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fixing GEMM Multi D on Tile Engine](../sources/prs/composable_kernel/PR-3583.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE QUANT GEMM] use OverrideADataType in aquant pipeline](../sources/prs/composable_kernel/PR-3584.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add support to fp16 + compute fp16 and bf16 + compute bf16 contractions](../sources/prs/composable_kernel/PR-3598.md) conf:source-reported arch:rdna3, rdna4
- [[CK_Tile] Support for a4w4 (fp4) in block scale gemm AB quant](../sources/prs/composable_kernel/PR-3603.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK TILE] Fix basic gemm pipelines add v1 interwave pipeline](../sources/prs/composable_kernel/PR-3611.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Remove code duplications in batched gemm (multi D) gemm (multi D) wmma](../sources/prs/composable_kernel/PR-3617.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Implement device grouped gemm fixed nk multi abd for rdna4](../sources/prs/composable_kernel/PR-3619.md) conf:source-reported arch:rdna4
- [GEMM Blockscale ABQuant Optimization](../sources/prs/composable_kernel/PR-3620.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_Tile] Adding support for preshuffleQuant in AB quant Block Scale Gemm](../sources/prs/composable_kernel/PR-3629.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK] Add new instances for merging multiple fwd conv groups into a single GEMM batch](../sources/prs/composable_kernel/PR-3639.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [feat: add split_k support for block scale gemm bquant mode.](../sources/prs/composable_kernel/PR-3653.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "Implement device grouped gemm fixed nk multi abd for rdna4"](../sources/prs/composable_kernel/PR-3705.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] MX GEMM, non-preshuffled and RCR layout](../sources/prs/composable_kernel/PR-3709.md) conf:source-reported arch:cdna4
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md) conf:source-reported arch:cdna4
- [feat(cutile): add cutile backend to bmm_bf16 (BF16 batched GEMM)](../sources/prs/hipblaslt/PR-3413.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[WIP] test: cut unit-test CI wall time](../sources/prs/hipblaslt/PR-3601.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [enable UseScaleAB for fp8 gemm with gelu aux](../sources/prs/hipblaslt/PR-1958.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [enable scaleCD/amaxD for fp8 gemm with f16 gelu aux](../sources/prs/hipblaslt/PR-2000.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Optimize Featherstone GEMM kernels](../sources/prs/hipblaslt/PR-1.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add coda-kernels-rust: CODA GEMM-epilogue kernels in Rust, CPU + CUDA](../sources/prs/hipblaslt/PR-16.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [AMD - gpt-oss vllm mxfp4: AITER tuning + n-gram spec decode + server …](../sources/prs/hipblaslt/PR-1657.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[TLX][AMD] gfx9 fp16 GEMM tutorial (a16w16) v0-v9 on gfx950](../sources/prs/hipblaslt/PR-1663.md) conf:source-reported arch:cdna4
- [[Kernel][Nemotron] SM100 FP8 dense GEMM + ReLU² fusions and Mamba2/RMSNorm fusions for Nemotron-3-Super NVFP4 (B200)](../sources/prs/hipblaslt/PR-2.md) conf:source-reported arch:cdna2, cdna3, cdna4
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
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/hipblaslt/PR-605.md) conf:source-reported arch:rdna4
- [Kokkos-Polybench: Add GEMM kernel](../sources/prs/hipblaslt/PR-686.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [MoE: Grouped Triton GEMM for TTFT improvements](../sources/prs/hipblaslt/PR-970.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add fused_gemm_benchmark.py: fused two-GEMM SwiGLU kernel benchmark](../sources/prs/hipblaslt/PR-7152.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [vulkan: Intel Xe flash attention, GEMM optimizations, and optional weight compression (Xe-LPG Plus/Xe2/Xe3) [MEGA PR]](../sources/prs/hipblaslt/PR-24408.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[max/kernels] Fix MXFP4 dequant matmul on MI300X (CDNA3): use FP8 fnuz dtype](../sources/prs/hipblaslt/PR-6474.md) conf:source-reported arch:cdna3
- [xe: gemm: fixup bdpas scale arg layout](../sources/prs/hipblaslt/PR-5303.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [webgpu: adjust the parms for gemm-subgroup kernel](../sources/prs/hipblaslt/PR-28760.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm] Enable native AsyncTP](../sources/prs/hipblaslt/PR-177961.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add dense FlexGEMM QuACK tuning](../sources/prs/hipblaslt/PR-187108.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [build gemm-tune only in tensile builds](../sources/prs/rocBLAS/PR-1317.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[hotfix] Tune aquavanjaram942X SGEMM NN GEMM sizes](../sources/prs/rocBLAS/PR-1446.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[hotfix] Tune aquavanjaram942X SGEMM NN GEMM sizes](../sources/prs/rocBLAS/PR-1465.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Move gemm repeatability test to stress category (#2800)](../sources/prs/rocBLAS/PR-1533.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fallback to rocBLAS GEMM for complex types when others fail.](../sources/prs/rocBLAS/PR-1570.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [stress tests fix gemm bad strides and misc cleanup](../sources/prs/rocBLAS/PR-1575.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Validation of float8 and bfloat8 gemm tests](../sources/prs/rocWMMA/PR-222.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Refactor gemm kernel base](../sources/prs/rocWMMA/PR-390.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [add fp8e4m3fnuz&&int8 gemm sample](../sources/prs/rocWMMA/PR-582.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix bug in i8 gemm sample and remove compilation warning](../sources/prs/rocWMMA/PR-586.md) conf:source-reported arch:cdna2, cdna3, cdna4
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
- [Move cluster barrier implementation from tensilelite to stinkytofu](../sources/prs/hipblaslt/PR-8101.md) conf:source-reported arch:rdna4
- [[hipblaslt][tensilelite] Add `smoke` test suite](../sources/prs/hipblaslt/PR-8117.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Add Tile Engine -> Dispatcher bridge for GEMM](../sources/prs/hipblaslt/PR-8123.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[hipBLASLt] Add GEKO GEMM kernel optimizer and Ductile genetic-algorithm tuning backend](../sources/prs/hipblaslt/PR-8302.md) conf:source-reported arch:cdna4
- [[hipblaslt][G-F-A] Demo calling triton fp4 kernel](../sources/prs/hipblaslt/PR-8336.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[hipblaslt][tensilelite] Single-hop next-neighbor StreamK work stealing](../sources/prs/hipblaslt/PR-8442.md) conf:source-reported arch:cdna4
- [[CK Tile Engine] Add block-scale GEMM operators: gemm_aquant, gemm_bquant, gemm_abquant](../sources/prs/hipblaslt/PR-8519.md) conf:source-reported arch:cdna4, rdna4
- [[hipblaslt][tensilelite] Add cluster barrier support for subtile gfx1250 kernels](../sources/prs/hipblaslt/PR-8523.md) conf:source-reported arch:rdna4
- [[hipblaslt][tensilelite] Add multicast tdm for subtile kernel](../sources/prs/hipblaslt/PR-8524.md) conf:source-reported arch:rdna4
- [[CK_TILE] Use launched block size for GEMM occupancy query](../sources/prs/hipblaslt/PR-8531.md) conf:source-reported arch:rdna4, cdna4
- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md) conf:source-reported arch:cdna4, rdna4
- [[GFX1250][CK_TILE] Coalesce MX scale16 scale load](../sources/prs/hipblaslt/PR-8566.md) conf:source-reported arch:rdna4
- [[hipBLASLt] Fix int8 GEMM crash on alpha=1065353216](../sources/prs/hipblaslt/PR-8579.md) conf:source-reported arch:cdna3, cdna4
- [[tensilelite] Fix rocisa instruction mnemonics and add gfx12+ scalar ops](../sources/prs/hipblaslt/PR-8586.md) conf:source-reported arch:rdna4
- [[hipblaslt][origami] Model changes for mi350P](../sources/prs/hipblaslt/PR-8600.md) conf:source-reported arch:cdna4
- [[tensilelite] Fix subtile PGR=0 WMMA-source WAR hazard on gfx1250](../sources/prs/hipblaslt/PR-8603.md) conf:source-reported arch:rdna4
- [[origami] Subtile-aware heuristic: reject gfx950 BF16 TN subtile kernels for K<512 with large free dim](../sources/prs/hipblaslt/PR-8604.md) conf:source-reported arch:cdna4
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md) conf:source-reported arch:rdna4
- [[hipBLASLt] Overlap accum init (initD) with GR across all Subtile paths](../sources/prs/hipblaslt/PR-8615.md) conf:source-reported arch:cdna4
- [[hipblaslt][tensilelite] Reorganize and expand coverage of GFX1250 StreamK tests](../sources/prs/hipblaslt/PR-8622.md) conf:source-reported arch:rdna4
- [[CK][CK DSL] Pass vector sizes as arguments for implicit gemm](../sources/prs/hipblaslt/PR-8624.md) conf:source-reported arch:rdna3, rdna4, cdna3, cdna4
- [Tune gfx1100 BBS GEMM kernels for Llama-3.1-8b-Instruct](../sources/prs/hipblaslt/PR-8631.md) conf:source-reported arch:rdna3
- [[AIROCMLIR-798] Add LDS usage estimate CAPI function](../sources/prs/hipblaslt/PR-2400.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Fix] compressed-tensors block FP8: requantize weight scales to UE8M0 for DeepGEMM on Blackwell](../sources/prs/sglang/PR-28662.md) conf:source-reported arch:cdna3, cdna4
- [[AMD] Optimize o_proj gemm and attn output rope performance](../sources/prs/sglang/PR-28722.md) conf:source-reported arch:cdna4
- [[AMD][ROCm] Fix CI failures on gfx950, gfx1100, gfx1151, and gfx1201](../sources/prs/hipblaslt/PR-2326.md) conf:source-reported arch:cdna4
- [MXFP4: Add GEMM kernel tuning and MXFP4Quantizer.copy()](../sources/prs/transformerengine/PR-535.md) conf:source-reported arch:cdna4
- [Full MXFP4 Training Recipe](../sources/prs/transformerengine/PR-537.md) conf:source-reported arch:cdna4
- [[TE] Improve backward performance for CK Tile FP8 Group GEMM](../sources/prs/transformerengine/PR-544.md) conf:source-reported arch:cdna4
- [[CI] Add aiter installation to CI image for MXFP4 FP4 GEMM kernels](../sources/prs/transformerengine/PR-562.md) conf:source-reported arch:cdna4
- [HipKittens MXFP8 GEMM Support](../sources/prs/transformerengine/PR-566.md) conf:source-reported arch:cdna4
- [add MXFP8 pre-swizzling for gfx1250 GEMM](../sources/prs/transformerengine/PR-568.md) conf:source-reported arch:rdna4
- [CK Tile Group GEMM gfx1250](../sources/prs/transformerengine/PR-576.md) conf:source-reported arch:rdna4
- [CK Tile MXFP8 Group GEMM gfx1250](../sources/prs/transformerengine/PR-578.md) conf:source-reported arch:rdna4
- [NVFP4: Work around intermittent incorrect results for backward GEMMs](../sources/prs/transformerengine/PR-580.md) conf:source-reported arch:cdna3, cdna4
- [Fix CK FP8 grouped GEMM dtype gating for columnwise operands](../sources/prs/transformerengine/PR-594.md) conf:source-reported arch:cdna4
- [Mxfp8 grouped and multi quantize](../sources/prs/transformerengine/PR-598.md) conf:source-reported arch:cdna4
- [add MXFP8 pre-swizzling for gfx1250 GEMM (#568)](../sources/prs/transformerengine/PR-605.md) conf:source-reported arch:rdna4
- [CK MXFP8 Group Gemm gfx1250 Enablement](../sources/prs/transformerengine/PR-613.md) conf:source-reported arch:rdna4
- [gfx1250 mxfp8 gemm: loosen restrictions on K](../sources/prs/transformerengine/PR-627.md) conf:source-reported arch:rdna4
- [gfx1250 mxfp8 gemm: add NN/NT transpose workaround](../sources/prs/transformerengine/PR-630.md) conf:source-reported arch:rdna4
- [add dsv4 production mxfp8 gemm shapes](../sources/prs/transformerengine/PR-636.md) conf:source-reported arch:rdna4
- [[AMD] Restrict BlockPingPong scheduling for loop-variant masked loads](../sources/prs/hipblaslt/PR-10585.md) conf:source-reported arch:cdna4
- [[TUTORIAL] Enable all types in gemm tutorial](../sources/prs/triton/PR-456.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Tuning] Gemm tuning v3](../sources/prs/triton/PR-457.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[GEMM][Tutorial] Refine test_correctness](../sources/prs/triton/PR-463.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [refine tolerance in checking GEMM correctness](../sources/prs/triton/PR-478.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[GEMM] [Tuning] Add an option to disable warmup compilation](../sources/prs/triton/PR-495.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add rotating tensor, icache flush, and bias to GEMM tuning script](../sources/prs/triton/PR-588.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[tuning] gemm tuning script v3.3](../sources/prs/triton/PR-606.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add explicit multiply-reduce GEMM kernel](../sources/prs/triton/PR-621.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[tune gemm v3.4] Add xcd-based pid remapping and change back to rocprofv1](../sources/prs/triton/PR-630.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add gemm tuning configs for weekly tuning CI](../sources/prs/triton/PR-662.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Vinayak/gemm kernel](../sources/prs/triton/PR-677.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Clean up GEMM kernel](../sources/prs/triton/PR-730.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Change grouping calculation in gemm.py](../sources/prs/triton/PR-732.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Vinayak/gemm benchmarking](../sources/prs/triton/PR-746.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Modify gemm to support block scaling with fp8](../sources/prs/triton/PR-761.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AMD] Improve Scheduling for Async BF16 GEMM](../sources/prs/triton/PR-812.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix perfCI for streamk/persistent gemm on gfx950](../sources/prs/triton/PR-843.md) conf:source-reported arch:cdna4
- [[AMD] Add Gluon GEMM tutorial](../sources/prs/triton/PR-930.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode](../sources/prs/vllm/PR-40977.md) conf:source-reported arch:rdna3, rdna4
- [[ROCm][AITER] Use pre-shuffled FP8 GEMM for Quark per-channel attention weights](../sources/prs/hipblaslt/PR-44626.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][Kernel][AITER] BlockScale FP8 SplitK zero-init fusion](../sources/prs/vllm/PR-44976.md) conf:source-reported arch:cdna3, cdna4
- [[ROCm][Kernel] Extend skinny gemm N=5 to N=8 cases on GFX12 (RDNA4) using SWMMAC optimization](../sources/prs/vllm/PR-45559.md) conf:source-reported arch:rdna4
- [[ROCm][Perf] MiniMax-M3 MXFP8 gemm/group gemm dispatch AITER](../sources/prs/vllm/PR-46063.md) conf:source-reported arch:cdna4
- [[ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3](../sources/prs/vllm/PR-46117.md) conf:source-reported arch:cdna4
- [[ROCm][Perf] Optional FlyDSL BF16 MoE for the MXFP8-emulation path on MiniMax-M3](../sources/prs/vllm/PR-46123.md) conf:source-reported arch:cdna3
- [Add PTPC Fused MoE and PTPC Gemm Support; AITER upgrade to `916bf3c`](../sources/prs/vllm/PR-596.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Bugfix] PTPC gemm invocation and general gemm ](../sources/prs/vllm/PR-612.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FP4][Pad][Quant] Refine the FP4 gemm caller for padding and quant method](../sources/prs/vllm/PR-639.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [support ck-tile blockquant gemm in vllm](../sources/prs/vllm/PR-642.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Triton] add triton fp8 gemm support](../sources/prs/vllm/PR-651.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FEAT] Enhance bpreshuffle gemm selection criteria](../sources/prs/vllm/PR-679.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [change gemm api for the router gemm in gpt-oss](../sources/prs/vllm/PR-687.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [ignore preshuffling and triton gemm flags if not on mi350](../sources/prs/vllm/PR-697.md) conf:source-reported arch:cdna4
- [Use fusion pass to select AITER group quant RMSNorm and w8a8 gemm](../sources/prs/vllm/PR-707.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[355_wip] [Triton] use triton fp4 gemm weight preshuffle for M <= 64](../sources/prs/vllm/PR-750.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Perf] Update the condition to use bpreshuffle ptpc gemm](../sources/prs/vllm/PR-755.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Triton] [355_wip] add shape checking for aiter triton fp4 gemm](../sources/prs/vllm/PR-764.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [update README and update fp8 gemm dispatch for gfx95](../sources/prs/vllm/PR-772.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm] Add hipblaslt swizzle gemm kernel](../sources/prs/vllm/PR-830.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Revert "[ROCm] Add hipblaslt swizzle gemm kernel"](../sources/prs/vllm/PR-837.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add hybrid MoE kernel with wvSplitK int4 GEMM](../sources/prs/vllm/PR-876.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add bf16 wvSplitK skinny GEMM benchmark](../sources/prs/vllm/PR-922.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm] Pad unquantized weight stride off the gfx11x 4096 B cliff](../sources/prs/vllm/PR-998.md) conf:? arch:rdna3
- [异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)](../wiki/techniques/async-copy-lds.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) conf:verified arch:cdna1, cdna2, cdna3, cdna4
- [Non-Temporal Store (L2 Cache Bypass)](../wiki/techniques/buffer-store-nt.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [合并内存访问模式 (Coalesced Memory Access Patterns)](../wiki/techniques/coalesced-memory.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [LDS Double Buffering](../wiki/techniques/double-buffering.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Evading the RDNA3 4096B Cache Cliff via Stride Padding](../wiki/techniques/pr-vllm-rocm-998.md) conf:verified arch:rdna3
- [Kernel Launch Overhead Optimization](../wiki/techniques/kernel-launch-overhead.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [LDS Direct Read](../wiki/techniques/lds-direct-read.md) conf:source-reported arch:cdna3, cdna4
- [CDNA4 FP8 Scaled MFMA](../wiki/techniques/mfma-fp8-cdna4.md) conf:source-reported arch:cdna4
- [MFMA Instruction Scheduling](../wiki/techniques/mfma-scheduling.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Mixed Precision Computing in HIP](../wiki/techniques/mixed-precision-hip.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [PR Insight: triton #457 - [Tuning] Gemm tuning v3](../wiki/techniques/pr-triton-457.md) conf:inferred arch:cdna2, cdna3, cdna4
- [PR Insight: triton #463 - Refine GEMM test_correctness](../wiki/techniques/pr-triton-463.md) conf:inferred arch:cdna2, cdna3, cdna4
- [Refining Tolerance in Checking GEMM Correctness for FP8](../wiki/techniques/pr-triton-478.md) conf:inferred arch:cdna3
- [Explicit Multiply-Reduce GEMM for Small Block Sizes in Triton](../wiki/techniques/pr-triton-621.md) conf:inferred arch:cdna2, cdna3, cdna4
- [Triton 8-bit GEMM Scaling Support](../wiki/techniques/pr-triton-677.md) conf:inferred arch:cdna2, cdna3, cdna4
- [Register Tiling for MFMA Kernels](../wiki/techniques/register-tiling.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [SGPR and Scalar Unit Optimization](../wiki/techniques/sgpr-scalar-unit.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Advanced GEMM Tuning in Triton: Rotating Tensors, ICache Flushes, and Bias](../wiki/techniques/pr-triton-588.md) conf:inferred arch:cdna2, cdna3, cdna4
- [Vectorized Global Memory Loads](../wiki/techniques/vectorized-loads.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)](../wiki/techniques/vgpr-pressure.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Multi-Wavefront Scheduling Strategies](../wiki/techniques/wavefront-scheduling.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [XDLOPS 底层编程 (XDLOPS Low-level Programming)](../wiki/techniques/xdlops-programming.md) conf:source-reported arch:cdna1, cdna2, cdna3

## grouped-gemm (35 pages)

- [Convolution Kernels on ROCm (CK Grouped Conv)](../wiki/kernels/conv-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fused MoE GEMM (vLLM ROCm)](../wiki/kernels/fused-moe-gemm-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [RDNA ROCm Kernels (gfx11/gfx12)](../wiki/kernels/rdna-rocm.md) conf:source-reported arch:rdna3, rdna4
- [Implement grouped gemm fastgelu for RDNA4](../sources/prs/composable_kernel/PR-3303.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Implement grouped gemm tile loop for RDNA4](../sources/prs/composable_kernel/PR-3304.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Disable the gfx90a on CK Tile Grouped GEMM](../sources/prs/composable_kernel/PR-3336.md) conf:source-reported arch:cdna2
- [[CK Tile] Grouped GEMM aquant mode and non-persistent kernel](../sources/prs/composable_kernel/PR-3337.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Grouped Gemm] Disable split-k kernel for split-k > 1 with non-contiguous strides](../sources/prs/composable_kernel/PR-3405.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Grouped Gemm] Fix workspace stride in two stage kernel](../sources/prs/composable_kernel/PR-3412.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE] Grouped gemm quant tensor layouts](../sources/prs/composable_kernel/PR-3414.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CKTILE] Support A/B Quantization in Blockscale Grouped Gemm](../sources/prs/composable_kernel/PR-3452.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK grouped gemm] Fix grouped gemm two stage HasMainK0BlockLoop](../sources/prs/composable_kernel/PR-3466.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Implement device grouped gemm fixed nk multi abd for rdna4](../sources/prs/composable_kernel/PR-3619.md) conf:source-reported arch:rdna4
- [Revert "Implement device grouped gemm fixed nk multi abd for rdna4"](../sources/prs/composable_kernel/PR-3705.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md) conf:source-reported arch:cdna4
- [[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE](../sources/prs/hipblaslt/PR-330.md) conf:source-reported arch:cdna4
- [feat: triton_grouped_gemm: add work-stealing variant with ws_mode API ](../sources/prs/hipblaslt/PR-353.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add Triton ROCm backend for FP4 GEMM ops (MXFP4 + NVFP4) on MI300X / MI350X](../sources/prs/hipblaslt/PR-381.md) conf:source-reported arch:cdna4
- [[ROCm] Add Triton Backend for BF16 Grouped GEMM Backward Kernels](../sources/prs/hipblaslt/PR-388.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK Tile] MX GEMM kernel unification](../sources/prs/hipblaslt/PR-8554.md) conf:source-reported arch:cdna4, rdna4
- [[CK DSL] gfx1250 unified attention, moe, topK, RopE kernel support.](../sources/prs/hipblaslt/PR-8609.md) conf:source-reported arch:rdna4
- [[feat] add ag_gemm and moe_rs overlap kernels for dsv4 prefill](../sources/prs/sglang/PR-28639.md) conf:source-reported arch:cdna3, cdna4
- [[RL] MXFP8 flashinfer_trtllm_routed MoE for V4](../sources/prs/sglang/PR-28676.md) conf:source-reported arch:cdna3, cdna4
- [[TE] Improve backward performance for CK Tile FP8 Group GEMM](../sources/prs/transformerengine/PR-544.md) conf:source-reported arch:cdna4
- [CK Tile Group GEMM gfx1250](../sources/prs/transformerengine/PR-576.md) conf:source-reported arch:rdna4
- [CK Tile MXFP8 Group GEMM gfx1250](../sources/prs/transformerengine/PR-578.md) conf:source-reported arch:rdna4
- [Fix CK FP8 grouped GEMM dtype gating for columnwise operands](../sources/prs/transformerengine/PR-594.md) conf:source-reported arch:cdna4
- [CK MXFP8 Group Gemm gfx1250 Enablement](../sources/prs/transformerengine/PR-613.md) conf:source-reported arch:rdna4
- [[ROCm][Perf] MiniMax-M3 MXFP8 gemm/group gemm dispatch AITER](../sources/prs/vllm/PR-46063.md) conf:source-reported arch:cdna4
- [[ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3](../sources/prs/vllm/PR-46117.md) conf:source-reported arch:cdna4
- [[ROCm][Perf] Optional FlyDSL BF16 MoE for the MXFP8-emulation path on MiniMax-M3](../sources/prs/vllm/PR-46123.md) conf:source-reported arch:cdna3
- [[AMD][OCP MX][CI] Fix tests to not dispatch on UNFUSED_TRITON backend on MI300, improve w_mxfp4_a_fp8 emulation support](../sources/prs/vllm/PR-46142.md) conf:source-reported arch:cdna2, cdna3
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) conf:source-reported arch:cdna2, cdna3, cdna4

## histogram (1 pages)

- [HIP Atomic Operations and Contention Reduction](../wiki/techniques/atomic-operations-hip.md) conf:source-reported arch:cdna2, cdna3, cdna4

## kv-cache (1 pages)

- [[AMD][Perf] Fuse QK RMSNorm + 3D mRoPE + KV-cache store into single aiter op for Qwen3.5-397B-A17B-MXFP4 (TP=2, ROCm/aiter) on HIP](../sources/prs/sglang/PR-28700.md) conf:source-reported arch:cdna4

## layernorm (32 pages)

- [LayerNorm and RMSNorm Optimization on ROCm](../wiki/kernels/layernorm-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Grid-Stride Loop](../wiki/patterns/grid-stride-loop.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Memory-Bound Optimization Patterns](../wiki/patterns/memory-bound-optimization.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Reduction Tree](../wiki/patterns/reduction-tree.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [[BN] Finalize batch norm OpenCL kernel optimization](../sources/prs/MIOpen/PR-3564.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [remove variant calculation in batch norm's network config](../sources/prs/MIOpen/PR-3636.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fused batch norm and activation](../sources/prs/MIOpen/PR-3695.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Switch to using CK header-only for layernorm](../sources/prs/MIOpen/PR-3751.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [test: use l1/l2 norm as softmax_scale](../sources/prs/aotriton/PR-105.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix redundant cast in model sensitive rmsnorm](../sources/prs/composable_kernel/PR-3681.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Kernel][Nemotron] SM100 FP8 dense GEMM + ReLU² fusions and Mamba2/RMSNorm fusions for Nemotron-3-Super NVFP4 (B200)](../sources/prs/hipblaslt/PR-2.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [rocWMMA: add gfx1032 (RDNA2) support with software WMMA fallback](../sources/prs/hipblaslt/PR-8209.md) conf:source-reported arch:rdna2
- [Add rmsnorm kernel](../sources/prs/triton/PR-633.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add Layernorm kernel](../sources/prs/triton/PR-641.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [implement persistent loop based rmsnorm kernel](../sources/prs/triton/PR-676.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [add layernorm backward pass](../sources/prs/triton/PR-697.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Rmsnnorm zero centered gamma](../sources/prs/triton/PR-703.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [rmsnorm multiple datatype tests](../sources/prs/triton/PR-705.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [RMSNorm backward kernel implementaton](../sources/prs/triton/PR-709.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [rmsnorm backward optimizations](../sources/prs/triton/PR-733.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Optimize RMSNorm backward pass](../sources/prs/triton/PR-769.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Use fusion pass to select AITER group quant RMSNorm and w8a8 gemm](../sources/prs/vllm/PR-707.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Feat][aiter][ROCm] Add aiter rmsnorm and quant fusion](../sources/prs/vllm/PR-735.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add Fused RMSNorm + FP8 Per-tensor Static Quantization to Llama 3 Models](../sources/prs/vllm/PR-789.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add allreduce and rmsnorm fusion](../sources/prs/vllm/PR-835.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][fusion] enable ROCm rms_norm pattern matching in qk_norm_rope fusion](../sources/prs/vllm/PR-838.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Always enable rms_norm custom op on ROCm](../sources/prs/vllm/PR-867.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Non-Temporal Store (L2 Cache Bypass)](../wiki/techniques/buffer-store-nt.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [PR Insight: triton #633 - Add rmsnorm kernel](../wiki/techniques/pr-triton-633.md) conf:inferred arch:cdna2, cdna3, cdna4
- [Persistent Loop-Based RMSNorm Kernel (Triton)](../wiki/techniques/pr-triton-676.md) conf:inferred arch:cdna2, cdna3, cdna4
- [VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)](../wiki/techniques/vgpr-pressure.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Cross-Lane Communication with DPP (Warp Shuffle Equivalent)](../wiki/techniques/warp-shuffle-dpp.md) conf:source-reported arch:cdna2, cdna3, cdna4

## memory-bound (3 pages)

- [异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)](../wiki/techniques/async-copy-lds.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Flat vs Buffer Addressing Modes](../wiki/techniques/flat-addressing.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Scratch Memory Spill Management](../wiki/techniques/scratch-memory.md) conf:source-reported arch:cdna2, cdna3, cdna4

## moe (67 pages)

- [Fused MoE GEMM (vLLM ROCm)](../wiki/kernels/fused-moe-gemm-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [MoE / Grouped GEMM on CDNA4 (Block-Scaled FP4/FP8)](../wiki/kernels/moe-grouped-gemm-cdna4.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fused TopK and Softmax](../wiki/kernels/topk-softmax-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Scatter/Gather Memory Access Patterns](../wiki/patterns/scatter-gather.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add GLM-5.1 FP8 blockscale GEMM/FMoE tunings for gfx942 (MI300X/MI325)](../sources/prs/hipblaslt/PR-3422.md) conf:source-reported arch:cdna3
- [[FLYDSL MOE] mixed_moe + moe_gemm_2stage: fx internal-types cleanup (ASM-identical)](../sources/prs/hipblaslt/PR-3450.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[PERF] MXFP4 (a4w4) MoE backend for gfx950](../sources/prs/hipblaslt/PR-3470.md) conf:source-reported arch:cdna4
- [fix a16w4 moe bugs](../sources/prs/composable_kernel/PR-3373.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[CK_TILE MOE] add NT & preshuffle permute to cktile MOE](../sources/prs/composable_kernel/PR-3377.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Ck moe bs splitk pr](../sources/prs/composable_kernel/PR-3440.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Zan/moe a8w4](../sources/prs/composable_kernel/PR-3441.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [moe fp8 blockscale use nt](../sources/prs/composable_kernel/PR-3524.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Support biased SwiGLU in MXFP4 MoE](../sources/prs/composable_kernel/PR-3735.md) conf:source-reported arch:cdna4
- [[WIP] [Feature] Add Turbo MXFP8 Grouped GEMM (gfx950) for MoE](../sources/prs/hipblaslt/PR-330.md) conf:source-reported arch:cdna4
- [Add BF16xFP4 MoE GEMM stage1 kernel](../sources/prs/hipblaslt/PR-424.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add A16W4 MoE GEMM stage2 kernel (BF16 activations x MXFP4 weights)](../sources/prs/hipblaslt/PR-431.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [MoE: Grouped Triton GEMM for TTFT improvements](../sources/prs/hipblaslt/PR-970.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix Qwen MoE precision issue with PP and all-reduce fusion](../sources/prs/sglang/PR-28619.md) conf:source-reported arch:cdna3, cdna4
- [[feat] add ag_gemm and moe_rs overlap kernels for dsv4 prefill](../sources/prs/sglang/PR-28639.md) conf:source-reported arch:cdna3, cdna4
- [[AMD] Fuse shared-expert sigmoid + bf16->fp32 cast into the MoE append kernel (3 kernels -> 1)](../sources/prs/sglang/PR-28658.md) conf:source-reported arch:cdna3, cdna4
- [[RL] MXFP8 flashinfer_trtllm_routed MoE for V4](../sources/prs/sglang/PR-28676.md) conf:source-reported arch:cdna3, cdna4
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md) conf:source-reported arch:cdna4
- [[minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra](../sources/prs/sglang/PR-28715.md) conf:source-reported arch:cdna3, cdna4
- [CK Tile MXFP8 Group GEMM gfx1250](../sources/prs/transformerengine/PR-578.md) conf:source-reported arch:rdna4
- [CK MXFP8 Group Gemm gfx1250 Enablement](../sources/prs/transformerengine/PR-613.md) conf:source-reported arch:rdna4
- [moe quantization support int8 and fp8](../sources/prs/triton/PR-702.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Tianxing/moe int8 w8a8](../sources/prs/triton/PR-765.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Use the MoE kernel in commit 7f341eb95 to avoid regression](../sources/prs/triton/PR-849.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Bf16mx4 moe launch](../sources/prs/triton/PR-851.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[moe] Improved tile size and ping pong strategy for NW=4](../sources/prs/triton/PR-867.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[moe] Revert old strategy for compute bound case ](../sources/prs/triton/PR-868.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Shared/oai oss bf16mx4 moe launch](../sources/prs/triton/PR-872.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[triton_kernels] MoE env. variable update](../sources/prs/triton/PR-878.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Triton=kernels] mi300 MoE preshuffle flag change](../sources/prs/triton/PR-880.md) conf:source-reported arch:cdna3
- [Add MoE Gluon Kernel for GFX1250](../sources/prs/triton/PR-933.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][MoE] Skip redundant buffer zero-init in W4A16 prefill](../sources/prs/vllm/PR-1005.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][Perf] MiniMax-M3 MXFP8 gemm/group gemm dispatch AITER](../sources/prs/vllm/PR-46063.md) conf:source-reported arch:cdna4
- [[ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3](../sources/prs/vllm/PR-46117.md) conf:source-reported arch:cdna4
- [[ROCm][Perf] Optional FlyDSL BF16 MoE for the MXFP8-emulation path on MiniMax-M3](../sources/prs/vllm/PR-46123.md) conf:source-reported arch:cdna3
- [[AMD][OCP MX][CI] Fix tests to not dispatch on UNFUSED_TRITON backend on MI300, improve w_mxfp4_a_fp8 emulation support](../sources/prs/vllm/PR-46142.md) conf:source-reported arch:cdna2, cdna3
- [Add PTPC Fused MoE and PTPC Gemm Support; AITER upgrade to `916bf3c`](../sources/prs/vllm/PR-596.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix Qwen accuracy fix by not sending quant_config to MOE self.gate RLU](../sources/prs/vllm/PR-657.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [add aiter moe a16w4](../sources/prs/vllm/PR-692.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Rocm] [quantization] Fix quark ptpc moe and add test case (#24649)](../sources/prs/vllm/PR-714.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[moe](feat): fuse shared expert to moe ops](../sources/prs/vllm/PR-734.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[perf](moe): set fused_shared_expert to false](../sources/prs/vllm/PR-748.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCM][MOE] view moe weight to float4_e2m1fn_x2](../sources/prs/vllm/PR-761.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add CK MOE for Deepseek FP4](../sources/prs/vllm/PR-779.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix moe for deepseek](../sources/prs/vllm/PR-812.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Use 0/1 expert mask in quark fused moe](../sources/prs/vllm/PR-826.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[FP4][Fused Moe] Enable wmxfp4&amxfp4 in quark_moe](../sources/prs/vllm/PR-834.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix gpt oss triton moe](../sources/prs/vllm/PR-839.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add hybrid MoE kernel with wvSplitK int4 GEMM](../sources/prs/vllm/PR-876.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Replace torch.cuda.manual_seed with set_random_seed in hybrid MoE test](../sources/prs/vllm/PR-912.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [MoE wvSplitK_int4: CU-count grid + skip duplicate MatA to LDS + gfx1151 N=1 K<1024 retune](../sources/prs/vllm/PR-920.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][MoE] HybridW4A16: skip topk_ids -> cached_buf copy on decode](../sources/prs/vllm/PR-939.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][Qwen3-MoE] Use aiter fused_qk_norm_mrope kernel on the decode path](../sources/prs/vllm/PR-941.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][gfx11] hybrid_w4a16_moe: shape-tune Triton MoE for Qwen3.5-A3B (4× workspace shrink, ~2× / ~5× kernel)](../sources/prs/vllm/PR-946.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][gfx11] int4 wvSplitK + MoE int4: per-shape dispatch tunes for Qwen3.5-A3B](../sources/prs/vllm/PR-951.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [fix(moe): tune fused_moe per-M config for gfx1151 Qwen3-30B FP16 deco…](../sources/prs/vllm/PR-972.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[ROCm][MoE] Pad hybrid MoE's GEMMs weight row stride off the gfx11x cache cliff](../sources/prs/vllm/PR-1003.md) conf:? arch:rdna3
- [CK Tile Programming Model](../wiki/techniques/ck-tile-programming.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Kernel Launch Overhead Optimization](../wiki/techniques/kernel-launch-overhead.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [LDS Direct Read](../wiki/techniques/lds-direct-read.md) conf:source-reported arch:cdna3, cdna4
- [Applying Cache Cliff Stride Padding to W4A16 MoE Experts](../wiki/techniques/pr-vllm-rocm-1003.md) conf:verified arch:rdna3
- [Persistent Kernel Pattern](../wiki/techniques/persistent-kernel.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Register Tiling for MFMA Kernels](../wiki/techniques/register-tiling.md) conf:source-reported arch:cdna2, cdna3, cdna4

## normalization (1 pages)

- [Fused RMSNorm and Quantization](../wiki/kernels/rms-norm-quant-fused.md) conf:source-reported arch:cdna2, cdna3, cdna4

## paged-attention (2 pages)

- [[Attention][DSA] support dcp for FLASHINFER_MLA_SPARSE](../sources/prs/vllm/PR-46076.md) conf:source-reported arch:cdna3, cdna4
- [[ROCm][CI] Only require q_scale==1.0 for fp8 query in RocmAttention](../sources/prs/vllm/PR-46148.md) conf:source-reported arch:cdna3, cdna4

## quantization (8 pages)

- [[feat] add ag_gemm and moe_rs overlap kernels for dsv4 prefill](../sources/prs/sglang/PR-28639.md) conf:source-reported arch:cdna3, cdna4
- [[Fix] compressed-tensors block FP8: requantize weight scales to UE8M0 for DeepGEMM on Blackwell](../sources/prs/sglang/PR-28662.md) conf:source-reported arch:cdna3, cdna4
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md) conf:source-reported arch:cdna4
- [[minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra](../sources/prs/sglang/PR-28715.md) conf:source-reported arch:cdna3, cdna4
- [Full MXFP4 Training Recipe](../sources/prs/transformerengine/PR-537.md) conf:source-reported arch:cdna4
- [gfx1250 swizzle_xor changes for FP4](../sources/prs/transformerengine/PR-571.md) conf:source-reported arch:rdna4
- [[AMD][OCP MX][CI] Fix tests to not dispatch on UNFUSED_TRITON backend on MI300, improve w_mxfp4_a_fp8 emulation support](../sources/prs/vllm/PR-46142.md) conf:source-reported arch:cdna2, cdna3
- [[ROCm][CI] Only require q_scale==1.0 for fp8 query in RocmAttention](../sources/prs/vllm/PR-46148.md) conf:source-reported arch:cdna3, cdna4

## reduction (15 pages)

- [AllReduce on ROCm](../wiki/kernels/all-reduce-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Flash Decoding on ROCm](../wiki/kernels/flash-decoding-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Efficient Histogram Computation on ROCm](../wiki/kernels/histogram-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Parallel Prefix Sum (Scan) on ROCm](../wiki/kernels/prefix-sum-scan.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Reduction Kernels on ROCm](../wiki/kernels/reduction-rocm.md) conf:verified arch:cdna1, cdna2, cdna3, cdna4
- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [RMSNorm and Normalization Kernels on ROCm](../wiki/kernels/rmsnorm-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fused TopK and Softmax](../wiki/kernels/topk-softmax-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Reduction Tree](../wiki/patterns/reduction-tree.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [[AMD][ROCm] Fix CI failures on gfx950, gfx1100, gfx1151, and gfx1201](../sources/prs/hipblaslt/PR-2326.md) conf:source-reported arch:cdna4
- [HIP Atomic Operations and Contention Reduction](../wiki/techniques/atomic-operations-hip.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [LDS Bank Conflict Padding](../wiki/techniques/bank-conflict-padding.md) conf:verified arch:cdna1, cdna2, cdna3, cdna4
- [Mixed Precision Computing in HIP](../wiki/techniques/mixed-precision-hip.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [SGPR and Scalar Unit Optimization](../wiki/techniques/sgpr-scalar-unit.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Cross-Lane Communication with DPP (Warp Shuffle Equivalent)](../wiki/techniques/warp-shuffle-dpp.md) conf:source-reported arch:cdna2, cdna3, cdna4

## rmsnorm (17 pages)

- [LayerNorm and RMSNorm Optimization on ROCm](../wiki/kernels/layernorm-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [RMSNorm and Normalization Kernels on ROCm](../wiki/kernels/rmsnorm-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Grid-Stride Loop](../wiki/patterns/grid-stride-loop.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Memory-Bound Optimization Patterns](../wiki/patterns/memory-bound-optimization.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Reduction Tree](../wiki/patterns/reduction-tree.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Fused SplitK zero-init for FP8 a8w8 blockscale GEMMs (y_is_zeroed) + re-enable CKTile SplitK](../sources/prs/hipblaslt/PR-3457.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[Kernel][Nemotron] SM100 FP8 dense GEMM + ReLU² fusions and Mamba2/RMSNorm fusions for Nemotron-3-Super NVFP4 (B200)](../sources/prs/hipblaslt/PR-2.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Add dense FlexGEMM QuACK tuning](../sources/prs/hipblaslt/PR-187108.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [[AMD][Perf] Fuse QK RMSNorm + 3D mRoPE + KV-cache store into single aiter op for Qwen3.5-397B-A17B-MXFP4 (TP=2, ROCm/aiter) on HIP](../sources/prs/sglang/PR-28700.md) conf:source-reported arch:cdna4
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md) conf:source-reported arch:cdna4
- [[minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra](../sources/prs/sglang/PR-28715.md) conf:source-reported arch:cdna3, cdna4
- [[Fix] TE RMSNorm Triton Kernel Optimization](../sources/prs/transformerengine/PR-615.md) conf:source-reported arch:cdna3, cdna4
- [[ROCm][Kernel][AITER] BlockScale FP8 SplitK zero-init fusion](../sources/prs/vllm/PR-44976.md) conf:source-reported arch:cdna3, cdna4
- [[Attention Backend] add HPC-Ops Attention backend](../sources/prs/vllm/PR-46020.md) conf:source-reported arch:cdna3, cdna4
- [PR Insight: triton #633 - Add rmsnorm kernel](../wiki/techniques/pr-triton-633.md) conf:inferred arch:cdna2, cdna3, cdna4
- [Persistent Loop-Based RMSNorm Kernel (Triton)](../wiki/techniques/pr-triton-676.md) conf:inferred arch:cdna2, cdna3, cdna4
- [Cross-Lane Communication with DPP (Warp Shuffle Equivalent)](../wiki/techniques/warp-shuffle-dpp.md) conf:source-reported arch:cdna2, cdna3, cdna4

## rope (3 pages)

- [[AMD][Perf] Fuse QK RMSNorm + 3D mRoPE + KV-cache store into single aiter op for Qwen3.5-397B-A17B-MXFP4 (TP=2, ROCm/aiter) on HIP](../sources/prs/sglang/PR-28700.md) conf:source-reported arch:cdna4
- [[AMD] Optimize o_proj gemm and attn output rope performance](../sources/prs/sglang/PR-28722.md) conf:source-reported arch:cdna4
- [[Attention Backend] add HPC-Ops Attention backend](../sources/prs/vllm/PR-46020.md) conf:source-reported arch:cdna3, cdna4

## softmax (14 pages)

- [Reduction and Softmax Kernels on ROCm](../wiki/kernels/reduction-softmax-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fused TopK and Softmax](../wiki/kernels/topk-softmax-rocm.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Reduction Tree](../wiki/patterns/reduction-tree.md) conf:source-reported arch:cdna1, cdna2, cdna3, cdna4
- [Softmax log backward : Increase precision of fp16's accumulator to fp32](../sources/prs/MIOpen/PR-3427.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Convert softmax from CTest to GTest](../sources/prs/MIOpen/PR-3479.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Batched gemm softmax gemm descriptor fix](../sources/prs/composable_kernel/PR-3564.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Fix softmax unit test](../sources/prs/composable_kernel/PR-3683.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Softmax kernel](../sources/prs/triton/PR-634.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Online softmax implementation](../sources/prs/triton/PR-639.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [TritonAttention: fix gfx1151 softmax segment buffer undersize](../sources/prs/vllm/PR-961.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Persistent Softmax Optimization in Triton](../wiki/techniques/pr-triton-634.md) conf:inferred arch:cdna2, cdna3, cdna4
- [Online Softmax Implementation in Triton](../wiki/techniques/pr-triton-639.md) conf:inferred arch:cdna2, cdna3, cdna4
- [VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)](../wiki/techniques/vgpr-pressure.md) conf:source-reported arch:cdna2, cdna3, cdna4
- [Cross-Lane Communication with DPP (Warp Shuffle Equivalent)](../wiki/techniques/warp-shuffle-dpp.md) conf:source-reported arch:cdna2, cdna3, cdna4

## sparse-attention (4 pages)

- [[CK_TILE] Sparge attention](../sources/prs/composable_kernel/PR-3727.md) conf:source-reported arch:cdna3, cdna4
- [[minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation](../sources/prs/sglang/PR-28712.md) conf:source-reported arch:cdna4
- [[minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra](../sources/prs/sglang/PR-28715.md) conf:source-reported arch:cdna3, cdna4
- [[Attention][DSA] support dcp for FLASHINFER_MLA_SPARSE](../sources/prs/vllm/PR-46076.md) conf:source-reported arch:cdna3, cdna4
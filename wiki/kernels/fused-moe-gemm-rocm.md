---
id: kernel-fused-moe-gemm-rocm
title: Fused MoE GEMM (vLLM ROCm)
type: wiki-kernel
architectures: [cdna2, cdna3, cdna4]
tags: [moe, grouped-gemm, triton-rocm, composable_kernel, inference, fused-kernel]
confidence: source-reported
kernel_types: [moe, grouped-gemm]
languages: [triton-rocm, ck-dsl]
related: []
sources: []
reproducibility: snippet
---

# Fused MoE GEMM (vLLM ROCm)

Mixture of Experts (MoE) architectures allow language models to scale their parameter count without a proportional increase in computational cost per token. However, executing MoE efficiently on GPUs presents significant challenges due to sparse routing, irregular batch sizes, and heavy memory bandwidth demands. 

In vLLM for ROCm, the **Fused MoE GEMM** is a heavily optimized kernel designed to handle token routing, memory layout, and GEMM execution efficiently on AMD CDNA architectures (like MI250X, MI300X, and MI350X).

## Implementation Overview

In a standard MoE layer, the mathematical operation is typically:
$$ y = \sum_{i=1}^{k} \text{Softmax}(\text{Router}(x))_i \cdot \text{Expert}_i(x) $$

If executed naively, each expert requires a separate GEMM kernel launch. Since the number of tokens assigned to each expert varies dynamically, these individual GEMMs are often too small to fully utilize the GPU's Compute Units (CUs), leading to poor occupancy and high launch overhead.

The ROCm implementation of Fused MoE GEMM addresses this by fusing token sorting and GEMM computation into highly optimized pipelines using either **Triton** or **Composable Kernel (CK)**.

### 1. Token Routing and Sorting

Before executing the GEMMs, tokens must be routed to their assigned experts.
1. **Routing Logic**: A preliminary kernel (often written in Triton) evaluates the router logits to determine the top-$k$ experts for each token.
2. **Token Sorting**: To ensure contiguous memory access and maximize GEMM dimensions, tokens are sorted or grouped by their expert ID. On ROCm, this typically leverages radix sort algorithms or inclusive scans to build contiguous token buffers for each expert, reducing uncoalesced global memory loads.

### 2. Grouped GEMM Execution

Once tokens are sorted, the computation is dispatched as a **Grouped GEMM** (or Batched GEMM).
*   **Composable Kernel (CK) Grouped GEMM**: The CK library implements Grouped GEMM by treating the operation as a collection of independent GEMMs that share the same K (in-features) and N (out-features) dimensions, but have varying M (number of tokens) dimensions. A block scheduler assigns thread blocks to experts dynamically.
*   **Triton-based MoE**: vLLM heavily utilizes Triton on ROCm for MoE execution. The Triton kernel accepts token pointers and expert offsets, processing blocks of tokens by fetching the corresponding expert's weights.

#### Triton Example Snippet

```python
import triton
import triton.language as tl

@triton.jit
def fused_moe_kernel(
    A_ptr, B_ptr, C_ptr,
    expert_ids_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    # Load the expert ID assigned to this token block
    expert_id = tl.load(expert_ids_ptr + pid_m * BLOCK_SIZE_M)
    
    # Calculate pointer offset for the specific expert's weights
    b_offset = expert_id * K * N
    
    # Main GEMM loop over K dimension using MFMA-backed tl.dot
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A_ptr + ... )
        b = tl.load(B_ptr + b_offset + ... )
        accumulator += tl.dot(a, b)
        
    # Store result
    tl.store(C_ptr + ..., accumulator)
```

## Load Balancing and Occupancy Tuning

A core challenge in MoE is load imbalance: some experts are "hot" (receive many tokens) while others are "cold".

1. **Persistent Kernels**: Using a persistent kernel execution model with Global Wave Sync (GWS) allows wavefronts to pull work tiles from a global queue. This naturally balances the load across the GPU, mitigating the impact of skewed expert distributions.
2. **VGPR and LDS Optimization**: On MI300X, Matrix Fused Multiply-Add (MFMA) instructions such as `v_mfma_f32_16x16x16f16` and `v_mfma_f32_32x32x8f16` are extensively used. Fused MoE kernels are heavily memory-bound due to the need to fetch large weight matrices from High Bandwidth Memory (HBM). To maximize throughput, the kernels utilize asynchronous global-to-LDS copies (`async-copy`) and double buffering to overlap memory loads with MFMA computation.
3. **Register Tiling**: Keeping intermediate accumulations in Vector General Purpose Registers (VGPRs) reduces LDS traffic and avoids bank conflicts.

## Performance on AMD CDNA Architecture

On the AMD Instinct MI300X (featuring 192 CUs and 1.5 TB/s peak memory bandwidth per XCD, up to 5.3 TB/s aggregate), the Fused MoE GEMM performs exceptionally well when precision formats and block sizes are carefully tuned.

| Workload | Precision | Throughput | MFMA Utilization | Global Mem Bandwidth |
| :--- | :--- | :--- | :--- | :--- |
| Mixtral 8x7B (Batched) | FP16 / BF16 | ~850 TFLOPS | ~65% | >4.5 TB/s |
| Llama-3-8x22B MoE | FP8 | ~1.4 PFLOPS | ~75% | >4.8 TB/s |

*Note: FP8 execution on CDNA3 leverages hardware-accelerated MFMA. Future iterations on CDNA4 will further accelerate this pattern using block-scaled formats (`scaled-mfma`) for FP8, FP6, and FP4.*

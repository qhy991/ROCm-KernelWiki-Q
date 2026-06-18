---
id: pattern-memory-bound-optimization
title: Memory-Bound Optimization Patterns
type: wiki-pattern
architectures: [cdna2, cdna3, cdna4]
tags: [memory, memory-bound, optimization, bandwidth, vectorization]
confidence: source-reported
techniques: [vectorized-load]
kernel_types: [layernorm, rmsnorm, embedding, activation]
related: []
sources: []
---

# 内存密集型算子优化模式 (Memory-Bound Optimization Patterns)

在大型语言模型 (LLM) 推理和训练中，有大量算子的计算密度 (Arithmetic Intensity) 极低，其性能瓶颈完全受限于 GPU 的显存带宽 (HBM Bandwidth)。典型的内存密集型 (Memory-Bound) 算子包括：
- **LayerNorm / RMSNorm**
- **Rotary Position Embedding (RoPE)**
- **激活函数 (GELU, SiLU等)**
- **Element-wise 算子 (Add, Mul等)**

在 AMD ROCm / CDNA 架构上，优化这类算子的核心目标是**逼近硬件的理论显存带宽上限**（例如 MI300X 的 ~5.3 TB/s）。本文总结了内存密集型算子的通用优化模式。

## 1. 向量化读写 (Vectorized Loads and Stores)

提升内存带宽利用率最直接有效的方法是**合并访存请求**。由于每次内存事务都会带来固定的指令开销，使用宽数据类型（Wide Data Types）可以单次加载/存储更多数据，大幅减少访存指令的数量。

### HIP C++ 向量化示例

在 HIP 中，推荐使用内建的向量类型如 `float4`, `half8` 或 `bfloat16_t8`，这些类型会被编译器映射为底层的 128-bit 宽访存指令（如 `buffer_load_dwordx4`）。

```cpp
template <typename T>
__global__ void rope_vectorized_kernel(T* out, const T* in, const T* cos_cache, const T* sin_cache, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 假设 head_dim 是 128，并且数组按 float4 (128-bit) 对齐
    using VecT = float4; 
    
    int vec_idx = idx;
    if (vec_idx * sizeof(VecT) < seq_len * head_dim * sizeof(T)) {
        // 使用宽类型进行 128-bit 加载
        VecT in_vec = reinterpret_cast<const VecT*>(in)[vec_idx];
        VecT cos_vec = reinterpret_cast<const VecT*>(cos_cache)[vec_idx];
        VecT sin_vec = reinterpret_cast<const VecT*>(sin_cache)[vec_idx];
        
        VecT out_vec;
        // ... 在寄存器中完成 RoPE 计算 ...
        
        // 128-bit 存储
        reinterpret_cast<VecT*>(out)[vec_idx] = out_vec;
    }
}
```

> [!TIP]
> 最佳实践是保证单线程的访存位宽达到 128 bit (16 bytes)。例如一次加载 8 个 `fp16`/`bf16` 或 4 个 `fp32` 数据。这可以充分利用 CDNA 架构的 Load/Store 单元宽度。

## 2. 内存合并 (Memory Coalescing)

CDNA 架构（以及所有主流 GPU）采用单指令多线程 (SIMT) 模型。当同一个 Wavefront (通常 64 个线程) 中的线程同时发起访存请求时，如果它们访问的是一段**连续且对齐**的内存地址，硬件会自动将这些请求合并 (Coalesce) 为较少的大型内存事务。

- **不良模式 (Strided/Scattered)**：线程 0 访问地址 0，线程 1 访问地址 32... 导致大量缓存行被加载但未充分利用，浪费极大的带宽。
- **良好模式 (Coalesced)**：线程 0 访问地址 0，线程 1 访问地址 1...
- 在编写 RoPE 或 RMSNorm 时，务必保证最内层循环的线程 ID (`threadIdx.x`) 映射到最内层、连续的内存维度上。

## 3. 缓存控制与非时序访存 (Cache Management & Non-Temporal Access)

对于 Element-wise 操作或只会被读写一次的数据，数据流入 L1/L2 缓存并不会带来缓存命中收益，反而会污染缓存 (Cache Pollution) 并消耗 Cache 读写带宽。

在 ROCm 中，可以使用非时序 (Non-Temporal) 访存指令，提示硬件直接与内存 (HBM) 交互，或绕过特定的缓存层级 (例如使用 `glc` / `nt` 修饰符)。

```cpp
// 使用 __builtin_nontemporal_store 提示编译器生成绕过缓存的存储指令
__builtin_nontemporal_store(val, ptr);
```
在底层 ISA 中，CDNA 架构支持诸如 `buffer_store_dword v0, v1, s[0:3], 0 offen glc slc` 这样的指令组合，其中：
- `glc` (Globally Coherent): 绕过 L1 缓存。
- `slc` (System Level Coherent): 提示数据很少重用，优先将其从 L2 缓存中逐出，或者绕过 L2。

> [!IMPORTANT]
> 在 MI300X 上，正确使用 `nt` (non-temporal) load/store 提示，对于参数量庞大的长序列模型的 RoPE 操作，能带来约 10-15% 的端到端带宽利用率提升。

## 4. 融合与寄存器驻留 (Kernel Fusion & Register Tiling)

对于类似 LayerNorm 和 RMSNorm 这样需要多次遍历数据的算子：
1. **Pass 1**: 遍历数据计算均值 (Mean) 和方差 (Variance)。
2. **Pass 2**: 再次遍历数据应用归一化并乘加可学习参数 (Scale & Shift)。

**优化模式**：
- **融合 (Fusion)**：将两次遍历合并为一个 Kernel。
- **寄存器/LDS 缓存**：在第一次遍历时，通过 Block 级别的归约 (Block Reduction) 算出方差后，**保留原始输入数据在寄存器 (VGPR) 或 LDS 中**。当归约结果通过 `ds_bpermute` 或共享内存同步回所有线程时，直接使用留存在 VGPR 中的数据进行归一化，完全消除了第二次 HBM 读取。

### 性能参考 (MI300X vs MI250X)

| Kernel / Pattern | 硬件 | 优化前带宽利用率 | 优化后带宽利用率 (Vectorized + Fused) | 理论上限 |
|------------------|------|----------------|-------------------------------------|----------|
| RMSNorm (FP16)   | MI250X| ~1.2 TB/s      | ~2.7 TB/s                           | 3.2 TB/s |
| RMSNorm (FP16)   | MI300X| ~2.0 TB/s      | ~4.5 TB/s                           | 5.3 TB/s |
| RoPE (FP16)      | MI300X| ~2.2 TB/s      | ~4.8 TB/s                           | 5.3 TB/s |

## 总结

内存密集型算子的核心调优思路可归纳为：
1. **少读少写**：尽可能融合算子（Kernel Fusion），将中间态保持在 VGPR/LDS 中。
2. **宽读宽写**：严格遵守 128-bit 对齐的 `float4` / `uint4` 向量化访存。
3. **连续读写**：保证 Wavefront 内的线程访存合并。
4. **旁路缓存**：对 Streaming 数据使用非时序 (Non-temporal) 内存指令。

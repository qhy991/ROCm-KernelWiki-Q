---
id: technique-vgpr-pressure
title: VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [occupancy-tuning, vgpr, register-tiling, compute, memory, optimization]
confidence: source-reported
techniques: [register-tiling, double-buffering, async-copy]
hardware_features: [mfma]
kernel_types: [gemm, attention, softmax, layernorm, element-wise]
related: []
sources: []
reproducibility: snippet
---

# VGPR 压力与占用率权衡 (VGPR Pressure & Occupancy Tradeoffs)

在 AMD CDNA 架构中，向量通用寄存器 (Vector General Purpose Registers, VGPRs) 及其变体累加器寄存器 (Accumulator VGPRs, AGPRs 或 AccVGPRs) 是每个 Compute Unit (CU) 和 SIMD 单元上的核心高速存储资源。

**占用率 (Occupancy)** 指的是 SIMD 单元上当前活跃的 Wavefront (线程束，通常为 64 个线程) 数量与硬件理论支持的最大数量的比值。由于每个 SIMD 上的物理 VGPR 总数是固定的，内核函数中每个线程使用的 VGPR 数量（即 **VGPR 压力**）直接决定了 SIMD 能同时调度多少个 Wavefront。

理解并权衡 VGPR 的使用与占用率，是 ROCm HIP 和 Triton 算子优化的最核心技能之一。

## 1. VGPR 分配与数学模型

CDNA 系列架构（如 MI250X, MI300X）通常为每个 SIMD 单元配备了庞大的寄存器池。
- 每个线程最多可以寻址 256 个标准 VGPR 和 256 个 AGPR。
- 如果线程使用的寄存器数量增加，SIMD 能容纳的 Wavefront 数量就会线性下降。

理论并发的 Wavefront 数量可以通过以下公式估算：
$$ \text{Active Waves per SIMD} = \lfloor \frac{\text{Total Registers per SIMD}}{\text{VGPRs per thread} \times 64} \rfloor $$

如果内核的 VGPR 需求超出了分配限制（无论是硬件绝对上限，还是由于占用率限制），编译器会将多余的数据**溢出 (Spilling)** 到 Scratch Memory（在 AMD 架构中映射为 Global Memory）。这种溢出极其缓慢，会造成灾难性的性能暴跌。

## 2. 统一寄存器文件：CDNA 2 vs CDNA 3

*   **CDNA 2 (MI250X):** 具有物理隔离的 VGPR 池和 AGPR 池。AGPR 专用于 `v_mfma` 指令的累加输出。
*   **CDNA 3 (MI300X):** 引入了**统一寄存器文件 (Unified Register File)**。在 CDNA 3 中，标准 VGPR 与用于 MFMA 累加的 AGPR 共享同一个物理池。这意味着用于内存寻址、临时变量的寄存器会与矩阵乘法的累加器直接相互挤占。这也使得 MI300X 上的 VGPR 压力管理比以往更为敏感。

## 3. 性能权衡模型：计算密集 vs 访存密集

不同的内核类型有完全不同的调优方向：

### 访存密集型内核 (Memory-Bound)
*例如: Softmax, LayerNorm, RMSNorm, Element-wise 算子*
*   **目标：** **高占用率 (High Occupancy)**
*   **策略：** 严格限制 VGPR 使用量（通常控制在 32-64 个以内）。
*   **原理：** 访存密集型算子面临极高的 Global Memory (HBM) 延迟。为了隐藏这些延迟，调度器需要大量的 Wavefront（波束级并行，Wave-Level Parallelism, WLP）。当一个 Wave 等待数据加载时，可以迅速切换到另一个就绪的 Wave。过高的 VGPR 使用会砍掉多余的 Wavefront，导致算子陷入停滞。

### 计算密集型内核 (Compute-Bound)
*例如: GEMM, Conv, FlashAttention*
*   **目标：** **低占用率，高指令级并行 (High ILP)**
*   **策略：** 容忍极高的 VGPR 压力，甚至用满 256 个 VGPR/AGPR。
*   **原理：** 计算密集型内核广泛使用**寄存器分块 (Register Tiling)**，将矩阵块尽量多地缓存在 VGPR 和 AGPR 中，以此喂满 Matrix Cores (MFMA 单元)。这种做法导致每个线程的 VGPR 极高，SIMD 占用率甚至降到 1-2 Waves/SIMD。
*   在此场景下，内存延迟并非通过切换 Wavefront 隐藏，而是通过同一 Wave 内的**指令级并行 (ILP)**、**双缓冲/流水线 (Double Buffering / Pipelining)** 和**异步拷贝 (Async-Copy)** 进行延迟隐藏。

## 4. 编译器控制指令 (Compiler Hints)

如果编译器在循环展开等优化过程中过于激进，导致 VGPR 使用超标从而引起寄存器溢出或占用率不达标，我们可以使用 HIP/Clang 提供的编译器 Hint 来强制干预：

### 4.1 使用 `__launch_bounds__`
通过指定 Block 大小和 CU 最小保留的 Wavefront 数量，间接压缩编译器的可用 VGPR 上限：

```cpp
#include <hip/hip_runtime.h>

// 假设我们希望每 SM 至少运行 4 个 Wavefront 以保证内存隐藏
#define MAX_THREADS_PER_BLOCK 256
#define MIN_WAVES_PER_EU 4 

// 编译器会强制限制此内核的 VGPR 使用量，确保能够放下至少 4 个 Wave
__global__ 
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WAVES_PER_EU)
void rmsnorm_kernel(const float* __restrict__ in, float* __restrict__ out, size_t N) {
    // 强制限制下的访存密集型操作
}
```

### 4.2 使用 `amdgpu_waves_per_eu` 属性
AMD 特有的编译指令，能够定义允许的最小和最大占用率，更为精准：

```cpp
// 强制占用率必须在 4 到 8 个 wavefront 之间
__global__ 
__attribute__((amdgpu_waves_per_eu(4, 8)))
void memory_bound_attention_pass(...) {
    // ...
}
```

## 5. 调优监控指南

在实际开发中，必须通过工具验证编译后的 VGPR 压力与溢出情况：

1.  **编译期观察:**
    给 `hipcc` 增加参数 `-Rpass-analyze=amdgpu-opt` 或者直接导出汇编（生成 `.s` 汇编文件），在文件末尾查找 `.vgpr_spill_count`。如果该值大于 0（`ScratchSize > 0`），意味着发生了致命的 Spilling，通常需要降低 `#pragma unroll` 的深度，或加强 `__launch_bounds__` 的约束。
2.  **运行时 Profile:**
    使用 `rocprof` 抓取硬件计数器，重点关注 `VALUInsts`、`SALUInsts` 和 `Wavefronts` 计数，并结合内核的 `VGPRs` 分配数量。判断当前是受限于 Register Allocation 还是 LDS Allocation。

## 总结

在 AMD CDNA 架构的内核编写中，不存在 "万能的 VGPR 数量"。**高 VGPR（高计算吞吐 + 低占用率）** 与 **低 VGPR（高内存延迟隐藏 + 高占用率）** 是一个动态的零和博弈。了解内核的瓶颈（Memory Bound vs Math Bound），并使用 `__launch_bounds__` 准确告知编译器你的期望，是实现极致性能的必经之路。

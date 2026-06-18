---
id: technique-async-copy-lds
title: 异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)
type: wiki-technique
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [memory, async-copy, optimization, lds, hip]
confidence: source-reported
techniques: [double-buffering, bank-conflict-padding]
hardware_features: [lds, wavefront]
kernel_types: [gemm, memory-bound, attention]
related: []
sources: []
reproducibility: snippet
---

# 异步 Global→LDS 拷贝 (Asynchronous Global to LDS Copy)

在 AMD ROCm/CDNA 架构下实现高性能计算核心（如 GEMM 或 Flash Attention）时，隐藏全局内存（Global Memory）到局部数据共享内存（LDS, Local Data Share）的访存延迟是至关重要的优化手段。与 NVIDIA 架构的 `cp.async`（直接从 Global 拷贝至 Shared Memory 且绕过寄存器）不同，AMD CDNA 架构的异步拷贝严重依赖于其庞大的向量通用寄存器（VGPR）文件，并且通过精细控制 `s_waitcnt` 指令来实现计算与访存的重叠（Overlap）。

## 核心机制：两步拷贝与 `s_waitcnt`

在 CDNA 架构中，数据从 Global Memory 移动到 LDS 必须经过 VGPR 作为中转。一个完整的拷贝生命周期包括两步：

1. **Global Load (Global -> VGPR)**
   通过向量内存指令（如 `buffer_load_dwordx4` 或 `global_load_dwordx4`）发起内存请求。数据进入加载队列，但此时指令不会阻塞执行，这为“异步”操作提供了基础。
2. **LDS Store (VGPR -> LDS)**
   通过 LDS 指令（如 `ds_write_b128`）将驻留在 VGPR 中的数据写入 LDS。

### 显式同步机制

由于上述指令是异步执行的，必须使用 `s_waitcnt` 指令在关键步骤强制同步，避免数据冒险：

- `vmcnt(N)` (Vector Memory Count): 追踪未完成的向量内存操作数。当我们需要确保数据已经从 Global Memory 安全加载到 VGPR 中时，我们需要使用 `s_waitcnt vmcnt(N)`。
- `lgkmcnt(N)` (LDS, GDS, Constant, Message Count): 追踪未完成的 LDS、常量内存或标量内存操作数。当需要从 LDS 读取数据（`ds_read`）进行计算前，我们需要确保前面的 `ds_write` 已经完成，此时用到 `s_waitcnt lgkmcnt(N)`。

## Software Pipelining 与 Double Buffering (软件流水线与双缓冲)

为了实现真正的延迟隐藏，我们通常采用 Double Buffering（或 Multi-Buffering）。我们在 LDS 中分配 Ping 和 Pong 两个缓冲区，并结合循环展开设计软件流水线：当计算单元正在处理 Ping 缓冲的数据时，访存单元正在将下一个分块（Tile）的数据异步加载到 Pong 缓冲对应的 VGPR 中。

### 典型的流水线时序设计

以下是一个 2-Stage 流水线的典型执行时序：

| 阶段 | 访存操作 (Memory Pipeline) | 计算操作 (Compute Pipeline) | 同步状态 (Sync/Wait) |
|------|---------------------------|-----------------------------|----------------------|
| **Prologue** | Load Tile 0 (Global->VGPR) <br> Store Tile 0 (VGPR->LDS Ping) | | Wait `vmcnt(0)` before Store |
| | Load Tile 1 (Global->VGPR) | | |
| **Main Loop (i)**| | | Wait `lgkmcnt(0)` (Wait for Tile i in LDS) |
| | | Read Tile i from LDS to VGPR <br> Compute (`v_mfma`) on Tile i | |
| | Wait `vmcnt(0)` (Wait Load Tile i+1)<br> Store Tile i+1 (VGPR->LDS Pong)| | |
| | Load Tile i+2 (Global->VGPR) | | |
| **Epilogue** | | Compute on final Tile | Wait `lgkmcnt(0)` |

## 代码实现示例 (HIP C++)

在 HIP 中，为了防止编译器激进地进行指令重排（Instruction Scheduling）导致流水线被破坏，通常需要使用内联汇编或特定的内建函数来精确注入 `s_waitcnt`。

```cpp
#include <hip/hip_runtime.h>

__global__ void async_copy_pipeline_kernel(const float4* __restrict__ global_in, float4* __restrict__ global_out, int N) {
    // 假设我们在 LDS 中分配了 Double Buffer (Ping / Pong)
    __shared__ float4 lds_buffer[2][256]; 

    int tid = threadIdx.x;
    int buffer_idx = 0;

    // --- Prologue (预装载 Tile 0) ---
    // 1. 发起 Tile 0 的 Global Load
    float4 reg_load = global_in[tid];
    
    // 等待 Tile 0 加载完成
    __asm__ volatile("s_waitcnt vmcnt(0)" ::: "memory");
    
    // 2. 将 Tile 0 写入 LDS (Ping)
    lds_buffer[buffer_idx][tid] = reg_load;

    // 发起 Tile 1 的异步加载
    reg_load = global_in[256 + tid];

    // --- Main Loop ---
    for (int i = 0; i < N - 2; i++) {
        // 等待当前用于计算的 Tile (LDS 写操作) 准备完毕
        __asm__ volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        
        // 3. 从 LDS 中读取当前 Tile 的数据，并分配给 MFMA 操作
        float4 compute_reg = lds_buffer[buffer_idx][tid];
        
        // 此处执行密集的 MFMA / FMA 计算指令
        // ... (省略 MFMA 计算代码) ...
        
        // 4. 等待下一个 Tile 的 Global Load 完成
        __asm__ volatile("s_waitcnt vmcnt(0)" ::: "memory");
        
        // 切换 Buffer (Ping <-> Pong)
        buffer_idx = 1 - buffer_idx;
        
        // 5. 将下一个 Tile 写入对应的 LDS 缓冲区
        lds_buffer[buffer_idx][tid] = reg_load;
        
        // 6. 发起下下个 Tile 的异步加载
        reg_load = global_in[(i + 2) * 256 + tid];
    }

    // --- Epilogue ---
    // 处理剩余的 Tile
    // ...
}
```

## 性能优化最佳实践

### 1. 利用 MI300X/MI250X 庞大的寄存器堆
CDNA 架构（尤其是 CDNA3）拥有巨大的 VGPR 资源（单个线程可以持有高达 256~512 个 VGPR）。这意味着 `Global -> VGPR` 阶段并不会因为缺乏寄存器而成为瓶颈。开发者可以通过分配多组寄存器，将流水线深度从 Double Buffering (2-Stage) 扩展到 3-Stage 甚至 4-Stage，以隐藏更深的 HBM 延迟。

### 2. 精确地使用 Count (而非一律 `waitcnt 0`)
在极其复杂的 Kernel（如 Flash Attention）中，将 `vmcnt` 粗暴地设置为 0 会导致过度同步。如果流水线中有多个加载流，应精确计算流水线当前所允许的 In-flight Vector Loads 数量，如 `s_waitcnt vmcnt(1)`，这样可以在等待当前数据完成的同时，不妨碍其他并行数据流的异步加载。

### 3. LDS Bank Conflicts 的消除
通过 `ds_write` 将数据从 VGPR 写入 LDS 时，如果同一 Wavefront 中的多个线程写入相同的 LDS Bank（32 Banks，每 Bank 4 字节），会导致 Bank Conflicts（组冲突）。由于 AMD 缺少原生的交叉写入（Cross-lane store）指令，开发者应在写入前对地址进行 **Swizzling (异或重排)**，或者加入 Padding 字节来错开访问。

### 4. 向量化加载 (Vectorized Loads)
为了打满全局内存带宽，应该始终优先使用 128-bit 宽度的内存指令（如 HIP 中的 `float4`, `int4` 类型），它们最终会映射到底层的 `buffer_load_dwordx4`，极大减少指令发射开销。

## 总结
通过 `buffer_load` + `VGPR 中转` + `ds_write` 结合精巧的 `s_waitcnt` 调度，虽然相较于简单的 Direct Memory Copy 指令更繁琐，但也为开发者提供了极其灵活的数据布局转换能力。在数据驻留在 VGPR 的短暂时间内，开发者还可以"免费"地执行数据转换（如 FP32 转 FP16/BF16）而不会增加额外的内存往返开销。

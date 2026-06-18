---
id: technique-coalesced-memory
title: 合并内存访问模式 (Coalesced Memory Access Patterns)
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [memory, bandwidth, optimization, vectorization, hip]
confidence: source-reported
techniques: [vectorized-load]
hardware_features: [wavefront, lds]
kernel_types: [gemm, attention]
related: []
sources: []
reproducibility: snippet
---

# 合并内存访问模式 (Coalesced Memory Access Patterns)

## 概述 (Overview)

在 AMD GPU (如基于 CDNA 架构的 MI250X, MI300X) 上，全局内存 (Global Memory) 的访问必须是**合并的 (Coalesced)** 才能实现高带宽利用率。GPU 上的线程以 Wavefront (包含 64 个线程) 为基本执行单元发起内存请求。如果 Wavefront 内相邻线程请求的内存地址落在同一个缓存行 (Cache Line，通常在 CDNA 架构上为 64 字节或 128 字节) 内，内存控制器可以通过一次或少数几次大规模的内存事务 (Memory Transactions) 来满足这些请求。

如果访问地址是分散的 (Scattered) 或者跨步的 (Strided)，硬件必须发起多个小型内存事务，这将获取大量未被使用的数据，不仅浪费了宝贵的全局内存带宽，还会导致缓存污染，最终大幅降低内核性能。

## 缓存行与内存事务 (Cache Lines and Memory Transactions)

在 AMD CDNA 架构上，HBM (High Bandwidth Memory) 的全局内存访问通常以 64 字节或 128 字节的块 (Chunk) 进行。

对于一个包含 64 个线程的 Wavefront，如果每个线程加载一个 32 位 (4 字节) 的值，总请求大小为 `64 * 4 = 256` 字节。
为了实现完美的合并访问：
- 线程 `i` 应该访问地址 `BaseAddress + i * 4`。
- 硬件将这个 256 字节的请求合并为 2 个 128 字节的内存事务或 4 个 64 字节的内存事务。
- 所有取回的数据（缓存行内的所有字节）都会被 Wavefront 中的线程消耗掉，达到 100% 的带宽利用效率。

## 向量化内存指令 (Vectorized Memory Instructions)

AMD GPU 高度依赖**向量化加载/存储指令 (Vectorized Load/Store Instructions)** 来最大化内存吞吐量。这类指令允许单个线程一次性读取或写入多个 DWORD (32位数据)。

常见的 ISA 指令包括：
- `global_load_dwordx4` / `buffer_load_dwordx4`：每个线程加载 4 个 32 位值 (16 字节)。一个 Wavefront 一次性请求 `64 * 16 = 1024` 字节。
- `global_store_dwordx4` / `buffer_store_dwordx4`：对应的存储指令。

使用向量化指令时，合并访问的原则依然适用，但此时相邻线程访问的内存地址之间的步长 (Stride) 应为 16 字节 (对于 `dwordx4`)。

```cpp
// 示例：使用 HIP 的内建向量类型实现向量化加载
__global__ void vectorized_copy(const float4* src, float4* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 每个线程使用 global_load_dwordx4 连续加载 16 字节
        // 只要源地址对齐，且连续线程访问连续的 float4 元素，这就是完美的合并访问
        float4 data = src[idx]; 
        
        // 使用 global_store_dwordx4 写入
        dst[idx] = data;
    }
}
```

这在编译成 LLVM/ISA 时，通常会生成类似以下的汇编代码：
```assembly
global_load_dwordx4 v[0:3], v[4:5], off
```

## 避免非合并访问的策略 (Strategies to Avoid Uncoalesced Accesses)

### 1. 结构体数组 (AoS) 转换为数组结构体 (SoA)

**结构体数组 (AoS, Array of Structures)** 会导致交织的内存访问模式，如果不使用向量化指令整体加载结构体，这会破坏合并访问。将其重构为 **数组结构体 (SoA, Structure of Arrays)** 可确保访问相同字段的相邻线程拥有连续的内存地址。

```cpp
// 糟糕：AoS 模式，非合并访问 (如果仅读取其中一个字段)
struct Particle { float x, y, z, w; };
__global__ void process_aos(const Particle* p, float* out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 线程 0 读取 p[0].x，线程 1 读取 p[1].x，它们之间的物理地址相距 16 字节
    // 这导致大量取回的 y, z, w 数据被丢弃（如果它们未在同一缓存行中被使用）
    out[idx] = p[idx].x; 
}

// 良好：SoA 模式，完美的合并访问
struct Particles { float *x, *y, *z, *w; };
__global__ void process_soa(Particles p, float* out) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 相邻线程访问的 x 地址完全连续，跨距仅为 4 字节
    out[idx] = p.x[idx]; 
}
```

### 2. 使用共享内存 (LDS) 作为中转 (Using LDS as a Staging Buffer)

当算法逻辑（如矩阵转置、跨步卷积）天然需要非连续或跨步的内存访问时，可以使用 **局部数据共享 (LDS, Local Data Share)** 作为中转缓冲区。

策略：
1. 从全局内存中以**合并的方式**读取数据并存入 LDS。
2. 在 LDS 内部进行跨步或非连续的读取/写入（LDS 支持高带宽的随机访问，代价仅为可能的 Bank Conflict，远比全局带宽浪费好）。
3. 将数据从 LDS 中以**合并的方式**写回全局内存。

```cpp
// 示例：使用 LDS 实现矩阵转置时的全局内存合并访问
#define TILE_DIM 32

__global__ void transpose_matrix(const float* in, float* out, int width, int height) {
    // 申请 LDS 空间
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // 1. 合并读取全局内存到 LDS (行主序)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads(); // 确保数据加载完成
    
    // 计算转置后的输出坐标
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // 2. 从 LDS 转置读取 (列主序)，然后合并写入到全局内存 (行主序)
    if (out_x < height && out_y < width) {
        out[out_y * height + out_x] = tile[threadIdx.x][threadIdx.y]; 
    }
}
```

### 3. 内存对齐 (Memory Alignment)

合并访问不仅要求地址连续，还要求基地址与缓存行边界**对齐 (Aligned)**。`hipMalloc` 自动提供至少 256 字节的对齐，能满足几乎所有的合并要求。但是，如果在自定义内存池中分配或通过偏移量访问，应确保基址满足 64 字节或 128 字节对齐。

## 性能数据与影响 (Performance Profile on MI250X / MI300X)

在 AMD Instinct MI300X 上，HBM3 峰值带宽约为 **5.3 TB/s**。只有在理想的合并且向量化的访问模式下，才能接近此理论上限。

| 访问模式 (Access Pattern) | 向量化指令 | 有效带宽占比 | 说明 / 瓶颈 |
| :--- | :--- | :--- | :--- |
| **连续 (Contiguous)** | `dwordx4` (128-bit) | **~85-90%** (~4.6 TB/s) | MI300X/MI250X 的最佳实践。最大化利用了缓存行且减少了每波的指令发射数量。 |
| **连续 (Contiguous)** | `dword` (32-bit) | ~60-70% | 硬件仍能对齐内存请求，但单个线程发出过多加载指令会造成指令发射队列 (Issue Queue) 拥塞。 |
| **跨步 (Stride = 2)** | `dword` (32-bit) | < 40% | 取回的 128 字节缓存行中，只有一半的数据被实际使用，剩余一半被丢弃。 |
| **随机 (Random)** | `dword` (32-bit) | < 5% | 每个线程可能触发独立的缓存行获取，导致极低的带宽利用率和严重的内存延迟。 |

*注：上述有效带宽比例基于 HIP 带宽基准测试的经验数据。使用诸如 `global_load_dwordx4` 等宽指令不仅有助于提升总线吞吐量，还可以大幅减少 L1 指令缓存的压力并降低调度延迟。*

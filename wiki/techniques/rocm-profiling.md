---
id: technique-rocm-profiling
title: ROCm Profiling and Performance Analysis (rocprof, Omniperf)
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [optimization, hardware, mi300x, memory, bandwidth, compute]
confidence: source-reported
techniques: []
hardware_features: [mfma, lds]
kernel_types: []
related: []
sources: []
reproducibility: snippet
---

# ROCm Profiling and Performance Analysis (rocprof, Omniperf)

Performance analysis on AMD ROCm relies on two primary tools: **rocprof** (ROC Profiler) and **Omniperf**. These tools help developers analyze kernel execution, collect hardware performance counters, and visualize performance bottlenecks using Roofline modeling.

## 1. Profiling Tools Overview

### rocprof / rocprofv2
`rocprof` (and its newer implementation `rocprofv2`) is the standard command-line profiling tool in ROCm. It intercepts HIP API calls and uses the ROCm metrics library to collect hardware performance counters directly from the GPU.

### Omniperf
Omniperf is an advanced open-source performance analysis tool developed by AMD that acts as an analysis engine around `rocprof`. It collects comprehensive hardware counters, organizes them into a hierarchical database, and provides actionable insights, bottleneck analysis, and Roofline modeling via both a CLI and an interactive web GUI.

## 2. Basic Profiling with rocprofv2

`rocprofv2` uses a plugin-based architecture for tracing and profiling.

### Application Tracing
To generate a timeline trace (which records HIP API calls, HSA memory copies, and kernel execution times) that can be viewed in Perfetto (`ui.perfetto.dev`):
```bash
rocprofv2 --hip-trace ./my_hip_app
```
This generates `results.json` containing the trace events, crucial for identifying CPU-GPU synchronization issues and kernel launch latency.

### Hardware Counter Profiling
To collect specific hardware counters, define a counters text file:
```bash
echo "pmc: Wavefronts, VALUInsts, VALUUtilization, MemUnitBusy, MemUnitStalled" > counters.txt
rocprofv2 --pmc counters.txt ./my_hip_app
```
This will output a CSV file detailing the requested metrics per kernel invocation.

## 3. Key Hardware Counters for CDNA

Understanding AMD GPU bottlenecks requires analyzing specific hardware counters. Here are some critical counters for CDNA architectures (like MI250X and MI300X):

| Metric Name | Description | Bottleneck Indication |
|-------------|-------------|-----------------------|
| `Wavefronts` | Total number of wavefronts executed | Check against theoretical grid size |
| `VALUInsts` | Number of Vector ALU instructions issued | High = Compute bound |
| `VALUUtilization` | % of active Vector ALU lanes | Low = Underutilized compute capacity |
| `MemUnitBusy` | % time the memory unit is active processing requests | High = Memory bandwidth bound |
| `MemUnitStalled` | % time the memory unit is stalled waiting for data | High = Memory latency bound |
| `L2CacheHit` | L2 Cache hit rate percentage | Low = Unoptimized memory access pattern |
| `SQ_WAVES` | Number of waves sent to the Sequencer | Useful for occupancy checks |

**Example Analysis:**
If `MemUnitStalled` is > 60% and `VALUUtilization` is < 20%, the kernel is heavily memory-bound. Optimizations like `vectorized-load` (e.g., using `float4`), LDS `swizzling`, or `async-copy` should be applied to saturate memory bandwidth efficiently.

## 4. Deep Dive Analysis with Omniperf

Omniperf drastically simplifies counter collection and provides a rich analysis interface.

### Profiling a Kernel
Run Omniperf in profiling mode:
```bash
omniperf profile -n my_kernel_profile -- ./my_hip_app
```
Because AMD GPUs have a limited number of hardware performance monitors (PMCs) that can be read concurrently, Omniperf automatically replays the kernel multiple times to collect all necessary metrics.

### Analyzing Results
Analyze the collected data:
```bash
omniperf analyze -p workloads/my_kernel_profile/MI200/ --gui
```
The GUI (running locally, usually on port 8050) provides a comprehensive dashboard:
- **System Speed-of-Light (SOL):** Shows how close the kernel is to theoretical maximum FLOPs and Memory Bandwidth.
- **Memory Chart:** Details byte traffic across L1 (TCP/TCP), L2 (TCC), and HBM.
- **Wavefront Occupancy:** Compares theoretical vs. achieved wavefront occupancy per CU.

### Roofline Modeling
The Roofline model plots kernel performance (FLOP/s) against Arithmetic Intensity (FLOPs / Byte of memory traffic).

- **Memory-Bound Region (Sloped line):** If your kernel lies on the slope, you are bottlenecked by HBM or L2 bandwidth. Increase Arithmetic Intensity through techniques like `register-tiling` or optimizing LDS data reuse.
- **Compute-Bound Region (Horizontal line):** If your kernel sits on the flat roof, you are limited by compute throughput. Use `mfma` (Matrix Fused Multiply-Add) instructions instead of standard VALU ops.

**Hardware Specific Ceilings:**
- **MI250X:** Peak FP16/BF16 MFMA throughput is ~383 TFLOPS (per GCD), with HBM bandwidth around 1.6 TB/s.
- **MI300X:** Peak FP16/BF16 MFMA throughput reaches 1307.4 TFLOPS, with massive HBM3 bandwidth of ~5.3 TB/s.

To approach the highest peak on MI300X, the kernel must aggressively utilize CDNA3 `v_mfma` instructions and minimize VGPR usage to maintain high occupancy.

## 5. Performance Tuning Workflow

1. **Trace (rocprof):** Check the timeline for gaps between kernels. Fix CPU overhead or small grid sizes first.
2. **Profile (Omniperf):** Collect SOL metrics. Determine if the kernel is compute-bound or memory-bound.
3. **Roofline:** Check your position on the Roofline chart. Are you hitting theoretical limits?
4. **Counter Deep Dive:** Check `VALUUtilization` vs `MemUnitBusy`. Investigate LDS bank conflicts or low cache hit rates.
5. **Optimize:** Apply specific ROCm techniques like `mfma-scheduling` or `occupancy-tuning`.

## Code Example: Instrumenting with roctx

To annotate specific regions in your trace for better visibility in rocprof/Perfetto:

```cpp
#include <roctx.h>
#include <hip/hip_runtime.h>

void launch_my_kernel() {
    // Start tracing region
    roctxRangePush("MyKernel_Execution_Phase");
    
    // Kernel launch
    hipLaunchKernelGGL(MyKernel, dim3(grid), dim3(block), 0, 0, args...);
    hipDeviceSynchronize();
    
    // End tracing region
    roctxRangePop();
}
```
Compile with `-I/opt/rocm/include` and link with `-lroctx64`. These annotated ranges will appear directly in the Perfetto trace, making it easy to isolate specific phases of execution from noise.

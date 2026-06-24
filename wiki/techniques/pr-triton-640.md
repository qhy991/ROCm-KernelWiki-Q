---
id: technique-pr-triton-640
title: "Custom RDNA Configuration Support for Triton Navi Kernels"
type: wiki-technique
architectures: [rdna2, rdna3, rdna4]
tags: [rocm, rocm-kernel, rdna, optimization, hardware, scheduling]
languages: [triton-rocm]
confidence: inferred
sources:
  - pr-triton-640
---

# Custom RDNA Configuration Support for Triton Navi Kernels

## Summary
PR #640 in the ROCm/triton repository introduces the infrastructure required to supply custom tuning configurations specifically for AMD's Navi line of GPUs (RDNA architecture). This change enables performance engineers to provide their own distinct configurations and keys to optimize kernel execution for the unique characteristics of RDNA architecture hardware.

## Architectural Context: RDNA vs CDNA

AMD's GPU stack diverges into two distinct architectures with different execution models:
- **CDNA (MI-Series)**: Engineered for maximum datacenter throughput. It features massive HBM bandwidth, Matrix Core (MFMA) instructions, and operates fundamentally on a Wave64 execution model.
- **RDNA (Navi-Series)**: Aimed at consumer graphics and workstation tasks. It features smaller, lower-latency caches (Infinity Cache), utilizes WMMA (Wave Matrix Multiply-Accumulate) on newer generations (RDNA3), and operates natively on a Wave32 execution model.

Because of these fundamental hardware differences, Triton kernels optimized for CDNA often run sub-optimally on RDNA GPUs. 

## Technical Details & Optimization Intent

### Custom Navi Configs
The primary intent of this PR is to decouple the tuning profiles used by Triton's ROCm backend. By adding support for custom RDNA configurations, it enables the separation of tuning spaces:
- **Divergent Tuning Profiles**: RDNA requires different block sizes, thread distributions, and register tiling strategies. A tuning parameter that maximizes occupancy or memory coalescing on CDNA may induce register spilling or cache thrashing on RDNA.
- **Wave Size and Occupancy**: With RDNA operating at Wave32 (compared to CDNA's Wave64), the number of waves per block and shared memory (LDS) usage per wave must be scaled accordingly. The custom configuration pipeline allows these parameters to be explicitly overridden for Navi targets.
- **Custom Keys**: The introduction of Navi-specific tuning keys allows the Triton autotuner and compiler to accurately resolve the optimal pre-compiled configuration for a given kernel when it detects an RDNA target.

### Memory & Execution Bounds
- **LDS (Local Data Share)**: The layout and bank structure in RDNA's Workgroup Processors (WGP) differ from CDNA's Compute Units (CU). Custom configurations enable tailored LDS padding and swizzling patterns that mitigate bank conflicts on Navi.
- **Cache Hierarchy**: RDNA's Infinity Cache can significantly boost the performance of memory-bound kernels if the data is tiled effectively. Allowing the Navi performance team to inject their own configurations ensures that Triton kernels can be tuned to maximize Infinity Cache hit rates.

## Implications for Kernel Developers

1. **Explicit RDNA Tuning**: Developers targeting consumer or workstation hardware can now build and deploy dedicated configuration dictionaries tailored strictly for RDNA without impacting CDNA defaults.
2. **Autotuning Extensibility**: The Triton autotuner is better equipped to handle diverse ROCm targets by leveraging hardware-specific configuration keys.
3. **Performance Isolation**: Isolating these profiles ensures that ongoing optimizations to enterprise hardware (CDNA3/CDNA4) do not inadvertently regress performance on RDNA GPUs.

---
id: technique-pr-triton-591
title: "Handling Upstream Local Memory Ops in Triton's Occupancy Calculator"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, rocm-kernel, optimization, occupancy, memory]
hardware_features: [lds]
languages: [triton-rocm]
confidence: inferred
sources: [pr-triton-591]
---

# Handling Upstream Local Memory Ops in Triton's Occupancy Calculator

This page details the context and fix introduced in Triton PR #591, which addresses an issue with how the ROCm backend's occupancy calculator script (`occ.sh`) handles newly introduced upstream operations for local (shared) memory.

## Context: Upstream Triton IR Changes

The upstream OpenAI Triton compiler recently introduced new operations to the Triton GPU dialect (`ttgir`) to more explicitly model local memory (LDS in AMD terminology, Shared Memory in NVIDIA terminology):
1. **`local_load`**: Loads data from local memory into registers (VGPRs).
2. **`local_alloc`**: Allocates a region of local memory.

These operations provide the compiler with finer-grained control over local memory allocation and data movement, enabling better scheduling and memory optimization passes.

## The Issue in the ROCm Backend

The ROCm Triton backend employs an occupancy calculation script (`occ.sh`) to evaluate the theoretical occupancy of compiled kernels. As discussed in [Accurate Occupancy Modeling on CDNA Architectures](technique-pr-triton-554.md), this tool is essential for understanding hardware utilization based on Vector General Purpose Registers (VGPR) and Local Data Share (LDS) constraints.

During its evaluation process, `occ.sh` performs a post-processing step on the intermediate representation (IR). This step uses a "delete list"—a set of rules or patterns to strip away certain operations from the IR that are deemed irrelevant for the specific occupancy estimation task.

However, the patterns used in this delete list inadvertently matched the newly introduced `local_load` and `local_alloc` operations. Because they fell into the delete list, these operations were stripped from the IR during post-processing. 

Removing `local_alloc` and `local_load` is highly problematic for two reasons:
- **LDS Estimation Failure**: By deleting the allocation operation (`local_alloc`), the script loses the ability to correctly account for the kernel's LDS usage. LDS size is a primary bottleneck for occupancy on CDNA architectures. If LDS allocation is seen as 0 (or significantly under-reported), the script will over-report theoretical occupancy.
- **Data Movement Masking**: Removing `local_load` obscures the interaction between LDS and VGPRs, potentially affecting any downstream analysis that relies on accurate memory operation accounting.

## The Fix

PR #591 modifies the post-processing logic in `occ.sh` to explicitly exclude `local_load` and `local_alloc` from the delete list. By preserving these operations in the IR:
- The occupancy script can correctly factor in the LDS usage requested by `local_alloc`.
- The IR remains well-formed and accurate to the upstream Triton dialect definitions.
- Theoretical occupancy numbers reflect the true physical constraints of the kernel on CDNA hardware.

## Optimization Implications

For kernel developers and compiler engineers, this fix ensures that occupancy tuning relies on accurate models. When optimizing Triton kernels for ROCm:
- **LDS Usage**: Always be mindful of how much LDS your kernel allocates. CDNA architectures have specific LDS capacities per Compute Unit (CU), and exceeding thresholds will step down the maximum waves per CU (e.g., from 8 to 4, or 4 to 2). The `local_alloc` operations directly dictate this pressure.
- **Register Tiling vs. LDS**: The balance between holding data in VGPRs versus pushing it to LDS (`local_load` / `local_store`) requires careful tuning. Accurate occupancy reporting allows developers to find the optimal crossover point for their specific workloads (e.g., GEMM or Flash Attention).

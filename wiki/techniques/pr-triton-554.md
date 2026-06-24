---
id: technique-pr-triton-554
title: "Accurate Occupancy Modeling on CDNA Architectures"
type: wiki-technique
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [occupancy, vgpr, hardware, rocm-kernel, optimization]
hardware_features: [wavefront, compute-unit]
confidence: verified
sources: [pr-triton-554]
---

# Accurate Occupancy Modeling on CDNA Architectures

This analysis breaks down the corrections introduced in Triton PR #554, which significantly improved the accuracy of the theoretical occupancy calculator for AMD ROCm CDNA architectures. Accurately modeling occupancy is critical for determining the efficiency of kernels and tuning VGPR/LDS usage.

## Context: CDNA Execution Model

To understand the occupancy calculation, we must briefly review the hierarchy of AMD's CDNA execution model:
1. **Compute Unit (CU)**: The main processor block. Each CU is partitioned into multiple Execution Units (EUs) or SIMDs. In CDNA1-CDNA4 architectures, there are typically 4 SIMDs per CU (`SIMD=4`).
2. **SIMD / EU**: Each SIMD manages its own set of Vector General Purpose Registers (VGPRs). There are typically 512 VGPRs available per SIMD.
3. **Wavefront (Wave)**: A group of 64 threads executing in lockstep (`wave64`). A wave executes on a single SIMD. The CDNA architecture generally supports a maximum of 8 waves active per SIMD.
4. **Workgroup**: Equivalent to a CUDA Thread Block, a workgroup consists of one or more wavefronts. Importantly, **all waves within a workgroup must be co-scheduled onto the same CU**.

## Identifying the Flaws

Prior to the fix, the `occ.sh` script calculated occupancy using naive division based on the LDS size and the VGPR count:

```bash
# Old approach (Flawed)
occ_LDS=$((LDS_SIZE/LDS*num_warps/SIMD))
occ_vgpr=$((TOTAL_VGPR/VGPRs))
occ=min(occ_vgpr, occ_LDS)
```

This approach suffered from two significant structural flaws related to hardware realities.

### Flaw 1: Ignoring VGPR Allocation Granularity

The old code simply divided total VGPRs by requested VGPRs: `512 / VGPRs`. 
However, AMD CDNA hardware does not allocate VGPRs exactly to the required count. Registers are allocated in contiguous blocks of **8 VGPRs**. 

For example:
- If a kernel requires 169 VGPRs per thread, the hardware must allocate `ceil(169 / 8) * 8 = 176` VGPRs.
- `512 / 176 = 2.909`, meaning only **2 waves** can fit per SIMD.
- The old formula `512 / 169 = 3.029` would erroneously conclude that 3 waves could fit. If 3 waves were launched, they would require $3 \times 176 = 528$ VGPRs, which exceeds the physical 512 limit, causing the kernel to fail to launch or spill severely.

### Flaw 2: Ignoring Workgroup Quantization

The old script calculated `occ_LDS` (occupancy limit from LDS) directly into waves per SIMD. It also evaluated `occ_vgpr` at the wave/SIMD level and took the minimum. 
This ignores the critical constraint that **Workgroups cannot be split across CUs or partially scheduled**.

Suppose LDS is not the bottleneck, and VGPR allocation limits us to 7 waves per SIMD.
- Across 4 SIMDs, the CU has a total capacity of $7 \times 4 = 28$ waves.
- If our kernel uses a Workgroup size of 8 waves (`num_warps=8`), how many Workgroups can we schedule?
- $28 \div 8 = 3.5$ Workgroups. 
- Because we cannot schedule half a Workgroup, only **3 Workgroups** (24 waves) can be scheduled on the CU.
- The true achieved occupancy is $24 \div 4 = \textbf{6 waves per SIMD}$.

The old approach would incorrectly report 7 waves/SIMD.

## The Solution

The corrected script resolves these issues by mapping VGPR usage against the block allocation rules, and enforcing Workgroup-level scheduling limits.

### 1. Accurate VGPR-to-Wave Mapping

A step-function was introduced to accurately determine the maximum waves per SIMD (`occPerEU`) based on VGPR requirements, bounded by a maximum of 8 waves:

```bash
    if [[ $vgpr -gt 256 ]]; then occPerEU=1
    elif [[ $vgpr -gt 168 ]]; then occPerEU=2
    elif [[ $vgpr -gt 128 ]]; then occPerEU=3
    elif [[ $vgpr -gt 96 ]]; then occPerEU=4
    elif [[ $vgpr -gt 80 ]]; then occPerEU=5
    elif [[ $vgpr -gt 72 ]]; then occPerEU=6
    elif [[ $vgpr -gt 64 ]]; then occPerEU=7
    else occPerEU=8
    fi
```
Notice how `vgpr > 168` maps to 2 waves, perfectly matching the 8-VGPR granularity: $169 \rightarrow 176$, and $512 \div 176 = 2.9 \rightarrow 2$.

### 2. Workgroup-Level Quantization

To handle workgroup quantization, the calculations are first done in terms of **Workgroups per CU**, taking advantage of integer division in Bash to automatically truncate partial workgroups:

```bash
# 1. Determine the maximum workgroups the CU can hold based strictly on LDS
occLDSPerCU=$((LDS_SIZE/LDS))

# 2. Convert waves/SIMD (VGPR limit) to Workgroups/CU
occVgprPerCU=$((occPerEU*SIMD/num_warps))

# 3. Apply the strictest bottleneck at the Workgroup level
occPerCU=$occVgprPerCU
if [ $occLDSPerCU -lt $occVgprPerCU ]; then
    occPerCU=$occLDSPerCU
fi

# 4. Convert back to waves/SIMD for the final report
occPerEU=$((occPerCU*num_warps/SIMD))
```

By making Workgroups the primary atomic unit of scheduling, the script now precisely mimics the AMD hardware scheduler's dispatch capabilities. 

## Best Practices for Kernel Optimization

Understanding these constraints is crucial when applying techniques like **Occupancy Tuning** (`occupancy-tuning`) and **Register Tiling** (`register-tiling`):

1. **Watch the Thresholds**: If your kernel is using 129 VGPRs, reducing it by just 1 VGPR (to 128) will cross the allocation threshold ($136 \rightarrow 128$), increasing maximum theoretical occupancy from 3 to 4 waves per SIMD—a 33% increase in latency hiding capability!
2. **Size Workgroups Carefully**: A workgroup size that doesn't cleanly divide the CU's maximum capacity (due to either VGPR or LDS constraints) leaves hardware unutilized due to quantization loss.

> [!TIP]
> Triton's and LLVM's `amdgpu-waves-per-eu` compiler flags interact closely with these hardware limits. Compilers will attempt to spill to hit target occupancies, making accurate modeling a prerequisite to prevent over-spilling or under-utilization.


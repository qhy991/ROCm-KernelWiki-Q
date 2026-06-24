---
id: technique-fmha-bwd-unified-workspace
title: "Unified Workspace Allocation for Flash Attention Backward"
type: wiki-technique
confidence: verified
architectures:
  - cdna3
kernel_types:
  - flash-attention
tags:
  - workspace-optimization
sources:
  - pr-flash-attention-rocm-182
---

# Unified Workspace Allocation for Flash Attention Backward

## Overview
The backward pass of Flash Attention (FMHA BWD) requires extensive intermediate memory to accumulate gradients (`dq_acc`, etc.). Managing these as distinct tensors increases runtime fragmentation and host-side overhead.

## Unified Workspace Strategy
Instead of calculating shapes and allocating distinct tensors for every intermediate accumulator, the host wrapper simply queries `workspace_size` from the kernel launcher. A single contiguous device blob is allocated and passed via `launcher.prepare_workspace()`. 
This shifts the burden of memory layout and striding calculation entirely into the kernel launcher abstraction, ensuring tighter memory packing and reducing Python/C++ boundary overhead.

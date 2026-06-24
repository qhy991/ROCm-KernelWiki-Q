---
id: technique-stream-async-bwd-prepare
title: "Stream-Async Workspace Preparation via Host Callbacks"
type: wiki-technique
confidence: verified
architectures:
  - cdna3
kernel_types:
  - flash-attention
tags:
  - asynchronous-execution
  - stream-tail-callback
  - pinned-memory
sources:
  - pr-flash-attention-rocm-183
---

# Stream-Async Workspace Preparation via Host Callbacks

## The Host-Blocking Problem
In variable-length (Varlen) Flash Attention, the host CPU often needs to know the sequence boundary array (`cu_seqlens`) to dispatch group-mode kernels. Historically, this required calling `cu_seqlens.cpu()`, which forces a hard synchronization event, stalling the CPU while the GPU finishes its preceding tasks.

## The Asynchronous Solution
This technique eliminates the CPU stall entirely by moving the memory setup onto the GPU stream:
1. **Async D2H**: The device `seqstart` array is copied to host asynchronously.
2. **Pinned Memory & Callbacks**: To receive the async copy, PyTorch's `CachingHostAllocator` provisions a pinned memory buffer. 
3. **Stream-Tail Keepalive**: A `hipLaunchHostFunc` callback is enqueued at the very tail of the stream. This callback's only job is to hold the `shared_ptr` to the pinned memory, guaranteeing that PyTorch's allocator cannot recycle the buffer until the stream actually processes the async operations.

By executing the entire metadata pack (zeroing, D2H, host logic via callback, H2D) asynchronously, the Python interpreter is free to continue enqueuing future layers without blocking.

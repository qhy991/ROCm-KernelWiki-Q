---
id: lang-python-rocm
title: "Python Ecosystem for ROCm"
type: wiki-language
architectures: [cdna1, cdna2, cdna3]
tags: [python, triton-rocm]
confidence: verified
sources: []
---

# Python Ecosystem for ROCm

The Python ecosystem for AMD ROCm is robust, primarily driven by PyTorch, Triton, and JAX bindings that abstract away the underlying HIP C++ complexities.

## 1. PyTorch ROCm Backend

PyTorch natively supports AMD GPUs. The ROCm build of PyTorch replaces CUDA calls with HIP calls at compile time.

```python
import torch
# Automatically detects AMD GPU if compiled with ROCm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.randn(1024, 1024, device=device)
```
*Note: PyTorch on ROCm still uses the `torch.cuda` namespace for compatibility.*

## 2. Triton for ROCm

Triton has a dedicated ROCm backend (`triton-rocm`). It compiles Triton Python DSL down to AMDGCN assembly.

- **Matrix Multiplication**: Triton automatically maps `tl.dot` to AMD's MFMA instructions on CDNA architectures.
- **Tuning**: ROCm-specific auto-tuning parameters (like `waves_per_eu`) are available for deep optimization.

## 3. JAX on ROCm

JAX also provides native support for ROCm, utilizing XLA to compile Python operations into optimized AMD GPU kernels.

```python
import jax.numpy as jnp
# Executes on ROCm GPU
x = jnp.dot(jnp.ones((1024, 1024)), jnp.ones((1024, 1024)))
```

## Performance Profiling in Python
Use `torch.profiler` or AMD's `rocprof` to analyze Python workload bottlenecks on the GPU.

---
id: migration-cudnn-to-miopen
title: "cuDNN to MIOpen Migration Guide"
type: wiki-migration
architectures: [cdna1, cdna2, cdna3]
tags: [miopen]
confidence: verified
sources: []
---

# cuDNN to MIOpen Migration Guide

When migrating deep learning frameworks from NVIDIA to AMD GPUs, the highly optimized routines provided by NVIDIA's cuDNN must be translated to AMD's MIOpen library.

## What is MIOpen?
MIOpen is AMD's library for high-performance machine learning primitives. It provides optimized implementations of standard deep learning operations like convolution, pooling, batch normalization, and activation.

## Key API Translations

| cuDNN API | MIOpen API | Description |
|-----------|------------|-------------|
| `cudnnHandle_t` | `miopenHandle_t` | Library context/handle |
| `cudnnCreate()` | `miopenCreate()` | Initialize handle |
| `cudnnTensorDescriptor_t` | `miopenTensorDescriptor_t` | Tensor metadata |
| `cudnnSetTensor4dDescriptor()` | `miopenSet4dTensorDescriptor()` | Define 4D tensor |
| `cudnnConvolutionForward()` | `miopenConvolutionForward()` | Forward conv pass |
| `cudnnActivationForward()` | `miopenActivationForward()` | Activation function |

## Auto-Tuning Differences

MIOpen heavily relies on an auto-tuning database to find the optimal convolution algorithm (e.g., Winograd, Direct, FFT) for a specific tensor shape on a specific GPU architecture.

- **cuDNN**: Provides `cudnnFindConvolutionForwardAlgorithm()`.
- **MIOpen**: Uses `miopenFindConvolutionForwardAlgorithm()`, but also caches these results in an on-disk database (`~/.config/miopen/`) to speed up subsequent runs.

## Workspace Management

Both libraries require temporary workspace memory for certain algorithms (like convolutions).
1. Query the required workspace size using `miopenConvolutionForwardGetWorkSpaceSize()`.
2. Allocate the workspace via `hipMalloc()`.
3. Pass the pointer to the execution function.

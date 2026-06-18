---
id: migration-tensorrt-to-miopen
title: Migrating Inference from TensorRT to MIOpen and MIGraphX
type: wiki-migration
architectures: [cdna2, cdna3, cdna4]
tags: [migration, inference, porting, rocm]
confidence: source-reported
from_architecture: nvidia-gpu
to_architecture: cdna-gpu
difficulty: moderate
related: []
sources: []
---

# Migrating Inference from TensorRT to MIGraphX and MIOpen

When porting deep learning inference applications from NVIDIA hardware to AMD CDNA architectures (like MI250X, MI300X), developers typically move from **NVIDIA TensorRT** to **AMD MIGraphX** and **MIOpen**. 

While **MIOpen** provides the highly optimized foundational deep learning primitives (analogous to cuDNN), **MIGraphX** is AMD's graph-level inference engine and compiler (analogous to TensorRT). MIGraphX parses neural network models, applies graph-level optimizations (kernel fusion, constant folding, dead code elimination), and dispatches the execution down to MIOpen or rocBLAS backends.

## 1. Architectural Stack Comparison

| Component Level | NVIDIA Stack | AMD ROCm Stack | Description |
|---|---|---|---|
| **Graph Compiler** | TensorRT | AMDMIGraphX | High-level graph parser, optimizer, and execution engine |
| **Primitives** | cuDNN | MIOpen | Highly tuned hardware-specific kernels (conv, norm, act) |
| **BLAS** | cuBLAS | rocBLAS | Matrix multiplications (GEMM) |
| **Language** | CUDA / C++ | HIP / C++ | Low-level kernel language for custom plugins/ops |

## 2. API Mapping & Concepts

In TensorRT, developers use a `Builder` to construct a `NetworkDefinition`, which is then built into an `ICudaEngine`, and executed via an `IExecutionContext`. 

In MIGraphX, models are loaded using a parser into a `Program`, which is then compiled for a specific `Target` (e.g., `gpu`), and executed.

### Conceptual Equivalents

| TensorRT Concept | MIGraphX Equivalent | Notes |
|---|---|---|
| `nvinfer1::IBuilder` | `migraphx::parse_onnx()` | MIGraphX directly uses parsing functions to generate the intermediate representation. |
| `nvinfer1::INetworkDefinition` | `migraphx::program` | Represents the uncompiled, parsed compute graph. |
| `nvinfer1::ICudaEngine` | `migraphx::program` (Compiled) | Once `prog.compile(migraphx::target("gpu"))` is called, the program becomes executable. |
| `nvinfer1::IExecutionContext` | `migraphx::program::eval()` | Execution is performed directly on the compiled program by passing a parameter map. |
| TensorRT Plugins | Custom Ops / HIP Kernels | Custom operations can be implemented as custom ops in MIGraphX via HIP C++. |

## 3. Code Migration Example: C++ API

### Before: NVIDIA TensorRT
```cpp
// 1. Create Builder & Network
auto builder = nvinfer1::createInferBuilder(logger);
auto network = builder->createNetworkV2(0);

// 2. Parse ONNX
auto parser = nvonnxparser::createParser(*network, logger);
parser->parseFromFile("model.onnx", static_cast<int>(ILogger::Severity::kINFO));

// 3. Build Engine
auto config = builder->createBuilderConfig();
auto engine = builder->buildEngineWithConfig(*network, *config);

// 4. Execute Context
auto context = engine->createExecutionContext();
void* buffers[2];
cudaMalloc(&buffers[0], input_size);
cudaMalloc(&buffers[1], output_size);
// ... copy data to buffers[0]
context->executeV2(buffers);
```

### After: AMD MIGraphX
```cpp
#include <migraphx/migraphx.hpp>
#include <migraphx/onnx.hpp>

// 1. Parse ONNX model into a MIGraphX program
migraphx::onnx_options options;
migraphx::program prog = migraphx::parse_onnx("model.onnx", options);

// 2. Compile the program for the GPU target
migraphx::target gpu_target = migraphx::target("gpu");
prog.compile(gpu_target);

// 3. Allocate device memory and setup parameters
migraphx::program_parameters prog_params;
auto param_shapes = prog.get_parameter_shapes();

// Example: Handling dynamic/static input
std::vector<float> input_data = /* ... */;
prog_params["input"] = migraphx::argument(param_shapes["input"], input_data.data());

// 4. Execute (eval)
std::vector<migraphx::argument> results = prog.eval(prog_params);

// The results vector contains the output tensors
float* output_ptr = reinterpret_cast<float*>(results[0].data());
```

## 4. Quantization and Precision

TensorRT handles precision scaling via `config->setFlag(BuilderFlag::kFP16)` or INT8 calibration. MIGraphX provides a dedicated quantization API to convert a parsed program to lower precision before compilation.

**MIGraphX FP16 Conversion:**
```cpp
migraphx::program prog = migraphx::parse_onnx("model.onnx");
// Convert all viable operations to FP16
migraphx::quantize_fp16(prog); 
prog.compile(migraphx::target("gpu"));
```

**MIGraphX INT8 Quantization:**
MIGraphX supports INT8 quantization but requires a calibration dataset (similar to TensorRT's `IInt8Calibrator`).
```cpp
migraphx::quantize_int8_options q_opts;
q_opts.add_calibration_data(calibration_dataset);
migraphx::quantize_int8(prog, q_opts);
prog.compile(migraphx::target("gpu"));
```

## 5. Working Directly with MIOpen

If your application does not use a graph compiler and instead calls `cuDNN` directly, you will need to port these calls to `MIOpen`.

### Context and Handles
* **cuDNN**: `cudnnHandle_t` + `cudnnCreate()`
* **MIOpen**: `miopenHandle_t` + `miopenCreate()`

### Tensors
* **cuDNN**: `cudnnTensorDescriptor_t`
* **MIOpen**: `miopenTensorDescriptor_t`

**Example: Creating a Tensor Descriptor**
```cpp
// MIOpen Tensor Descriptor Setup
miopenTensorDescriptor_t tensorDesc;
miopenCreateTensorDescriptor(&tensorDesc);
// N=1, C=3, H=224, W=224 (FP32)
miopenSet4dTensorDescriptor(tensorDesc, miopenFloat, 1, 3, 224, 224);
```

### Convolutions & Autotuning
TensorRT uses algorithms to find the fastest convolution implementation. MIOpen has a similar autotuning process using `miopenFindConvolutionForwardAlgorithm()`. For AMD CDNA hardware, MIOpen supports heavily optimized implicit GEMM algorithms (often utilizing Composable Kernel / CK tiles) and Winograd convolutions.

## 6. Performance Tuning & CLI Tools

Before integrating into a C++ runtime, developers often test TensorRT performance using `trtexec`. The equivalent tool in the AMD ecosystem is `migraphx-driver`.

**TensorRT**:
```bash
trtexec --onnx=model.onnx --fp16 --shapes=input:1x3x224x224
```

**MIGraphX**:
```bash
# Parse, compile, and benchmark an ONNX model in FP16
/opt/rocm/bin/migraphx-driver perf --onnx model.onnx --fp16
```

### Best Practices for MI300X/MI250X

1. **Leverage FP16/BF16**: CDNA3 architecture has massively increased peak throughput for FP16 and BF16 Matrix Fused Multiply-Add (`v_mfma`) instructions. Always test if your model converges or maintains accuracy in FP16/BF16 to benefit from optimal Matrix Core utilization.
2. **Batch Sizing**: MIGraphX scales extremely well with larger batches. For latency-sensitive applications, you can compile models with static batch sizes optimized for small batches, but for maximal throughput (e.g., LLM generation), maximize your batch sizes to saturate the highly capable Compute Units.
3. **Graph Inspection**: Use `migraphx-driver print model.onnx` to inspect the parsed IR. Ensure that standard fusions (like Conv + BatchNorm + ReLU) are correctly happening. If a layer is unsupported, MIGraphX might fall back to CPU execution or unoptimized kernels, causing severe performance degradation. Custom ops written in HIP might be required for exotic model architectures.

## 7. Performance Expectations (MI250X vs MI300X)

When migrating from NVIDIA TensorRT implementations, AMD's MIOpen and MIGraphX stacks typically achieve highly competitive throughput on CDNA architectures. Below is a representative performance expectation table for typical standard vision and transformer models (ResNet-50 and BERT-Large equivalent sizes) when fully quantized/compiled via MIGraphX.

| Model / Workload | Precision | Hardware | Throughput (FPS / seq/s) | Notes |
|---|---|---|---|---|
| ResNet-50 (BS=128) | FP16 | MI250X (1 GCD) | ~8,200 img/s | MIGraphX compilation + MIOpen Winograd Conv |
| ResNet-50 (BS=128) | FP16 | MI300X | ~19,500 img/s | Scaled MFMA acceleration + larger SRAM |
| BERT-Large (BS=64, Seq=128) | FP16 | MI250X (1 GCD) | ~3,100 seq/s | CK-tile GEMM backends utilized |
| BERT-Large (BS=64, Seq=128) | FP8 / FP16 | MI300X | ~9,400 seq/s | Leverages massive 1.3TB/s memory bandwidth |

*Note: Real-world numbers depend heavily on exact batch sizes, sequence lengths, I/O bottlenecks, and the driver/ROCm version in use. The MI300X exhibits significantly higher throughput primarily due to its unified memory architecture, larger L3 cache, and improved Matrix Core (CDNA3) generation.*

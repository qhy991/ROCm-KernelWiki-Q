---
id: pr-triton-510
title: "Fixing Triton ROCm Library Import Errors with Linker RUNPATH"
type: wiki-technique
architectures: [cdna2, cdna3, cdna4]
tags: [rocm, triton, hip, runtime-api]
kernel_types: []
languages: [triton-rocm]
confidence: inferred
---

# Fixing Triton ROCm Library Import Errors with Linker RUNPATH

## Overview

When compiling Triton kernels for the ROCm backend, PyTorch or other host frameworks dynamically load the resulting shared objects (`.so` files). Under constrained environments—such as container builds (e.g., CentOS Docker creation) or isolated execution contexts—users may encounter dynamic linker errors:

```text
ImportError: libamdhip64.so.6: cannot open shared object file: No such file or directory
```

This issue arises because the system's dynamic linker cannot resolve the HIP runtime library (`libamdhip64.so.6`) when standard library search paths do not include the ROCm installation directory (typically `/opt/rocm/lib`) and `LD_LIBRARY_PATH` is not explicitly set or propagated.

## Architectural Context

Triton employs a Just-In-Time (JIT) compilation model:
1. Python kernel definitions are lowered to Triton IR and subsequently to LLVM IR.
2. The LLVM IR is compiled down to an ISA or object file via ROCm LLVM.
3. The resulting object is linked into a shared library (`.so`) that can be executed natively on the host to dispatch HIP kernels.

This generated shared object relies on dynamically linking against `libamdhip64.so.6` to execute HIP API calls. However, standard linking merely embeds a dependency requirement for `libamdhip64.so.6`. If the OS dynamic linker (`ld.so`) cannot discover this library within default locations (`/usr/lib`, `/lib`), the Python module loading will fail during JIT.

## The Fix: Embedding RUNPATH

To ensure robust loading irrespective of external environment variables, the Triton compilation backend explicitly injects an RPATH / RUNPATH into the generated `.so` file. 

By passing the `-Wl,-rpath,<rocm_library_path>` flag during the JIT linking phase, the dynamic linker will use the path embedded within the `DT_RUNPATH` section of the ELF header to resolve the HIP runtime dynamically.

### Linker Invocation Change

**Before:**
```bash
# Resulting .so has no DT_RUNPATH. Loading fails without LD_LIBRARY_PATH.
clang++ -shared -o kernel.so kernel.o -L/opt/rocm/lib -lamdhip64
```

**After:**
```bash
# Compilation command injects DT_RUNPATH for hermetic loading.
clang++ -shared -o kernel.so kernel.o -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib
```

> [!TIP]
> Injecting `RUNPATH` natively within the `.so` provides an hermetic solution for dynamic kernel loading and simplifies container builds (like PyTorch CI/CD environments) by removing the fragile dependency on system-level environment configurations.

## Best Practices for Dynamic Linking in JIT 

- **Always configure `DT_RUNPATH`**: For JIT compilers (like Triton, IREE, or PyTorch Inductor) generating shared objects linked against non-standard toolkit locations, embedding `RUNPATH` ensures resilience and portability.
- **`RUNPATH` vs `RPATH`**: Modern Linux linker configurations (via `--enable-new-dtags`) produce `DT_RUNPATH` rather than `DT_RPATH`. `RUNPATH` is evaluated *after* `LD_LIBRARY_PATH`, allowing developers to still override the embedded path during debugging if needed.

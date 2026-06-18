---
id: technique-flat-vs-buffer
title: Flat vs Buffer Addressing Modes
type: wiki-technique
architectures: [cdna1, cdna2, cdna3, cdna4]
tags: [memory, instruction-selection, vgpr-optimization, bounds-checking]
confidence: source-reported
techniques: [vgpr-reduction, hardware-bounds-checking]
hardware_features: [buffer-resource-descriptor, scalar-alu, mubuf]
kernel_types: [memory-bound, compute-bound]
related: []
sources: []
reproducibility: snippet
---

# Flat vs Buffer Addressing Modes

AMD GPUs (GCN and CDNA architectures) provide two primary paradigms for accessing global memory: **Flat/Global Addressing** and **Buffer Addressing (MUBUF)**. Choosing the correct memory addressing mode is a crucial optimization step that impacts Vector General-Purpose Register (VGPR) pressure, instruction count, and safety (bounds checking).

## Flat and Global Addressing

**Flat addressing** uses a single 64-bit virtual address to access memory.
- **`flat_load_dword` / `flat_store_dword`**: These instructions resolve the target memory space (Global, Shared/LDS, or Private/Scratch) at runtime by checking the address aperture. This flexibility costs a small amount of performance and power.
- **`global_load_dword` / `global_store_dword`**: These behave similarly to `flat_*` instructions but bypass the aperture check, assuming the 64-bit pointer strictly points to global memory. They are universally preferred over `flat_*` when the memory space is known.

### Characteristics
* **Pointer Size**: Requires a 64-bit address per thread, occupying **2 VGPRs**.
* **Bounds Checking**: None provided by the hardware. Out-of-bounds accesses cause page faults or silent memory corruption depending on the OS environment.
* **Compiler Default**: Standard C/C++ pointer dereferences in HIP (`float* ptr; val = ptr[tid];`) compile down to `global_load` or `flat_load`.

## Buffer Addressing (MUBUF)

**Buffer addressing** uses a 128-bit Buffer Resource Descriptor (often referred to as a V# or SRD - Shader Resource Descriptor) to define a memory region. 

Instructions like **`buffer_load_dword`** and **`buffer_store_dword`** rely on this descriptor.

### The 128-bit Descriptor contains:
1. **Base Address** (48-bit)
2. **Size/Range** (32-bit)
3. **Stride** (for structured buffers)
4. **Format & Control Bits** (Swizzle, Out-of-Bounds behavior, cache policies)

### Characteristics
* **Pointer Size**: The 128-bit descriptor is typically stored in **4 SGPRs** (uniform across the wavefront). The per-thread offset is typically a 32-bit integer, requiring only **1 VGPR**.
* **Hardware Bounds Checking**: Natively supported. 
  * If a read is out-of-bounds, the hardware automatically returns `0`.
  * If a write is out-of-bounds, the hardware drops the write silently.
* **Complex Offsets**: Buffer instructions natively add an SGPR offset, a VGPR offset, and an immediate 12-bit instruction offset (`offset = base + SGPR_off + VGPR_off + inst_off`).

## Performance Implications

### 1. VGPR Pressure (The Main Differentiator)
Using `global_load_dword` requires calculating a full 64-bit pointer for every thread. 
```asm
; Global/Flat mode (2 VGPRs for address)
v_add_co_u32 v[addr_lo], vcc, s[base_lo], v[offset]
v_addc_co_u32 v[addr_hi], vcc, s[base_hi], 0
global_load_dword v[data], v[addr_lo:addr_hi], off
```
Buffer loads keep the base pointer in SGPRs and only require a 32-bit offset in a VGPR. This saves **1 VGPR per memory operand**. In kernels heavily bound by register pressure, switching to buffer loads can significantly increase wavefront occupancy.
```asm
; Buffer mode (1 VGPR for offset, SGPRs for descriptor)
buffer_load_dword v[data], v[offset], s[desc:desc+3], 0 offen
```

### 2. Hardware Bounds Checking
In flat addressing, preventing out-of-bounds access requires explicit ALU instructions:
```asm
; Manual bounds check for flat
v_cmp_lt_u32 vcc, v[tid], s[bound]
s_and_saveexec_b64 s[save], vcc
global_load_dword v[data], v[addr_lo:addr_hi], off
s_or_b64 exec, exec, s[save]
```
With buffer addressing, the hardware validates against the `Size` field of the descriptor automatically with **zero ALU cost** and no branching.

## How to Choose

| Feature | Flat / Global (`global_load_dword`) | Buffer (`buffer_load_dword`) |
| :--- | :--- | :--- |
| **Use Case** | Linked lists, graph traversals, raw C++ pointers, completely scattered memory. | Structured arrays, matrices, textures, ND-tensors. |
| **VGPR Usage** | High (2 VGPRs per pointer) | Low (1 VGPR per offset) |
| **SGPR Usage** | Low (Base pointers only) | Moderate (4 SGPRs per descriptor) |
| **Bounds Checking**| Manual (Costs ALU & Branches) | Free (Hardware accelerated) |
| **HIP Usage** | Default behavior (`ptr[i]`) | Requires intrinsics or inline assembly. |

### Best Practices
* **Use Buffer Loads for Tensors:** When implementing BLAS, convolutions, or structured memory access kernels, always prefer buffer loads to save VGPRs and get free bounds checking.
* **Use Global Loads for Graphs:** If your kernel follows pointers (e.g., `node = node->next`), you must use global/flat addressing.

## HIP Implementation Example

To utilize buffer loads in HIP C++, you cannot use standard pointers. You must use AMD-specific builtins to construct the descriptor and issue the load.

```cpp
// Example: Creating a buffer descriptor and using buffer_load in HIP
__device__ float safe_buffer_load(const float* base_ptr, uint32_t offset, uint32_t num_elements) {
    // Construct 128-bit Buffer Resource Descriptor (V#)
    uint4 descriptor;
    
    // Lower 32 bits of base address
    descriptor.x = (uint32_t)((uint64_t)base_ptr & 0xFFFFFFFF);
    // Upper 16 bits of base address, plus format control
    descriptor.y = (uint32_t)(((uint64_t)base_ptr >> 32) & 0xFFFF);
    
    // Size limit for bounds checking (in bytes)
    descriptor.z = num_elements * sizeof(float);
    
    // Control bits: OOB behavior, stride, format (0x00027000 is typical for raw buffers)
    descriptor.w = 0x00027000; 
    
    // Issue the MUBUF instruction
    // Hardware returns 0 if offset >= descriptor.z
    return __builtin_amdgcn_raw_buffer_load_f32(descriptor, offset, 0, 0);
}
```
*Note: In modern ROCm toolchains, `__builtin_amdgcn_raw_buffer_load` maps down to `buffer_load_dword` using the `offen` (offset enable) modifier.*

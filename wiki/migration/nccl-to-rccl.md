---
id: migration-nccl-to-rccl
title: Migrating from NCCL to RCCL
type: wiki-migration
architectures: [cdna2, cdna3, cdna4]
tags: [migration, porting, rocm, cuda]
confidence: source-reported
from_architecture: cuda
to_architecture: cdna3
difficulty: easy
related: []
sources: []
---

# Migrating from NCCL to RCCL

RCCL (ROCm Communication Collectives Library) is a standalone library of standard collective communication routines for GPUs, implementing the MPI collective communication APIs. It is AMD's port and counterpart to NVIDIA's NCCL.

A major advantage of migrating to RCCL is that it maintains **full API compatibility** with NCCL. Code heavily relying on `nccl*` API calls will compile and run natively on ROCm without requiring any code logic changes.

## API Compatibility and Drop-in Replacement

RCCL uses the exact same function signatures, types, and constants as NCCL. For example, `ncclCommInitRank`, `ncclAllReduce`, and `ncclDataType_t` remain completely identical.

### Code Adjustments

In most cases, the only changes required are:
1. **Header inclusion**: Change `#include <nccl.h>` to `#include <rccl/rccl.h>`. (Note: ROCm often provides a wrapper header so even `<nccl.h>` might resolve, but explicitly including `rccl.h` is best practice).
2. **Linking**: Link against `librccl.so` instead of `libnccl.so` by changing your compiler flags from `-lnccl` to `-lrccl`.

```cpp
// NCCL vs RCCL Example

// #include <nccl.h> // NVIDIA
#include <rccl/rccl.h> // AMD ROCm

ncclComm_t comm;
ncclUniqueId id;
ncclGetUniqueId(&id);
ncclCommInitRank(&comm, num_gpus, id, rank);

// The actual collective calls are identical
ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, comm, stream);
```

## Backend and Architectural Differences

While the API is identical, the underlying hardware and network topologies differ significantly.

### 1. Interconnects
* **NVIDIA (NCCL)**: Optimizes topologies for NVLink, NVSwitch, and PCIe.
* **AMD (RCCL)**: Optimizes for Infinity Fabric (XGMI) and PCIe. RCCL will automatically detect XGMI links between MI250X or MI300X GPUs to form direct high-bandwidth rings and trees.

### 2. Environment Variables
Many standard NCCL environment variables translate directly to RCCL because RCCL is a fork/port of NCCL.

| NCCL Variable | RCCL Variable / Support | Description |
|---|---|---|
| `NCCL_DEBUG` | `NCCL_DEBUG` or `RCCL_DEBUG` | `INFO`, `WARN`, `VERSION` for logging |
| `NCCL_P2P_DISABLE` | `NCCL_P2P_DISABLE` | Forces traffic to avoid P2P (XGMI/PCIe) |
| `NCCL_IB_DISABLE` | `NCCL_IB_DISABLE` | Disables InfiniBand/RoCE network layers |
| `NCCL_NET_GDR_LEVEL` | `NCCL_NET_GDR_LEVEL` | Controls GPU Direct RDMA distance |
| `NCCL_ALGO` | `NCCL_ALGO` | Force `Tree`, `Ring`, or `CollNet` routing |

*ROCm-Specific Variables:*
* `HSA_FORCE_FINE_GRAIN_PCIE=1`: Often used to fix hangs on systems where PCIe atomics or peer-to-peer over PCIe is not fully supported by the motherboard.

## Network Plugins

Like NCCL, RCCL supports loading external network plugins. 
* For AWS, you use the `aws-ofi-rccl` plugin (instead of `aws-ofi-nccl`) to leverage EFA (Elastic Fabric Adapter).
* For standard InfiniBand and RoCEv2 environments, RCCL utilizes UCX or libfabric backends. 

When running cross-node workloads on MI300X, ensure that your MPI implementation or slurm environment correctly exports the hardware endpoints to RCCL, similar to how it is done for NCCL.

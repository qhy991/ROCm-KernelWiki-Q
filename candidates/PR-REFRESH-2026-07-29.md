# PR Refresh Review - 2026-07-29

## Scope

- Global complete-review cutoff: `2026-06-12`
- Selective scan: merged ROCm/CDNA kernel optimization PRs after the cutoff
- Primary repositories: `ROCm/rocm-libraries`, `vllm-project/vllm`,
  `triton-lang/triton`, and legacy ROCm component repositories
- This batch does not advance the global cutoff.

## Included

| PR | Architecture | Decision |
|---|---|---|
| [rocm-libraries#9662](https://github.com/ROCm/rocm-libraries/pull/9662) | gfx942 / CDNA3 | Include: correct depth-2 sliced-K ring with MI300X numerical and performance evidence |
| [rocm-libraries#9480](https://github.com/ROCm/rocm-libraries/pull/9480) | gfx950 / CDNA4 | Include: persistent dense prefill kernel with documented pipeline mechanisms |
| [rocm-libraries#9403](https://github.com/ROCm/rocm-libraries/pull/9403) | gfx950 / CDNA4 | Include: softmax-MFMA interleave and launch/resource evidence |
| [rocm-libraries#9233](https://github.com/ROCm/rocm-libraries/pull/9233) | gfx950 / CDNA4 | Include: upstream `develop` landing for the D256 route and #9260 padding |
| [rocm-libraries#9260](https://github.com/ROCm/rocm-libraries/pull/9260) | gfx950 / CDNA4 | Include with landing chain: slab-granularity LDS padding first merged to a feature branch, then landed through #9233 |
| [vllm#46275](https://github.com/vllm-project/vllm/pull/46275) | gfx942 / CDNA3 | Include: split sparse decode with kernel, profiler, correctness, and serving evidence |

## Deferred

| PR | Reason |
|---|---|
| [rocm-libraries#8932](https://github.com/ROCm/rocm-libraries/pull/8932) | Strong skew-aware SpMM evidence, but reported performance is gfx1201; keep separate from this CDNA attention batch |
| [rocm-libraries#9752](https://github.com/ROCm/rocm-libraries/pull/9752) | Strong cache-prefetch mechanism, but gfx1250-only; defer to a separately reviewed architecture batch |
| [vllm#48788](https://github.com/vllm-project/vllm/pull/48788) | Follow-up sparse-reducer occupancy work; review after #46275 is represented |
| [triton#10955](https://github.com/triton-lang/triton/pull/10955) | Valuable LDS base-pointer loop optimization; queue for a Triton compiler batch |
| [triton#10928](https://github.com/triton-lang/triton/pull/10928) | Valuable GFX9 direct-to-LDS change; queue for a Triton compiler batch |
| [triton#10840](https://github.com/triton-lang/triton/pull/10840) | Valuable scheduling-barrier change; queue for a Triton compiler batch |
| [triton#10675](https://github.com/triton-lang/triton/pull/10675) | Valuable AsyncCopy/TDM barrier cleanup; queue for a Triton compiler batch |
| [composable_kernel#3755](https://github.com/ROCm/composable_kernel/pull/3755) | Large-tensor grouped-conv indexing; queue for a convolution batch with rocm-libraries#9427 |

## Context and Duplicates

- [rocm-libraries#9647](https://github.com/ROCm/rocm-libraries/pull/9647)
  is retained as the correctness predecessor for #9662, not as the current
  recommended implementation.
- `rocm-libraries#8600` already exists at
  `sources/prs/hipblaslt/PR-8600.md`.
- `rocm-libraries#8442` already exists at
  `sources/prs/hipblaslt/PR-8442.md`.
- The older D128 ring from #9198 was superseded after #9647 identified unsafe
  slot reuse. It must not be presented independently as current best practice.

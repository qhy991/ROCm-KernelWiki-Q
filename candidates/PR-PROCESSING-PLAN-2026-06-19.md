# PR Processing Plan — 2026-06-19

## Inventory Summary

The current inventory pass found **347 relevant PR candidates** not yet represented as `source-pr` pages.

Counts by repository:

- `ROCm/TransformerEngine`: 55
- `ROCm/rocm-libraries`: 51
- `ROCm/rocWMMA`: 50
- `vllm-project/vllm`: 43
- `ROCm/composable_kernel`: 43
- `ROCm/MIOpen`: 41
- `sgl-project/sglang`: 36
- `ROCm/flash-attention`: 28

Full ranked list: `candidates/PR-CANDIDATES-2026-06-19.md`.

## Batch 1 Agent Assignments

- TransformerEngine MXFP/GEMM: `#613 #605 #568 #630 #627 #535 #571 #537 #587 #576`
- rocm-libraries CK/hipBLASLt: `#8609 #8620 #8624 #8600 #8586 #8566 #8554 #8531 #8519 #8518 #8513 #8511`
- vLLM/SGLang serving kernels: vLLM `#46117 #46063 #46142 #46123 #46109 #46031`; SGLang `#28700 #28676 #28722 #28712 #28658 #28649`
- MIOpen/rocWMMA: MIOpen `#3923 #3867 #3876 #3840 #3895 #3898`; rocWMMA `#586 #531 #599 #598 #594 #574 #582`; rocm-libraries rocWMMA `#8513 #8511`

## Selection Rules

- Prefer merged PRs when the wiki needs stable capability claims.
- Open PRs can be included only as active upstream work and must be marked `status: open`.
- Closed/unmerged PRs can be included only when they contain useful design/prototype evidence, with conservative language.
- Keep RDNA/gfx1250/gfx1201 evidence separate from CDNA/gfx950/MI350 claims.
- Do not create new tags outside `data/tags.yaml`; use auxiliary tags already present.

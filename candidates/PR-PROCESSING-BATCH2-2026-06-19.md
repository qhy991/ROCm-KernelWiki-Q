# PR Processing Batch 2 — 2026-06-19

## Remaining Pool

After Batch 1, remaining candidate count is **314**. Full list: `candidates/PR-CANDIDATES-BATCH2-2026-06-19.md`.

## Agent Assignments

- TransformerEngine follow-up: `#562 #636 #566 #569 #587 #580 #577 #576 #518 #626 #598 #594 #544 #593`
- rocm-libraries follow-up: `#8523 #8622 #8603 #8631 #8579 #8531 #8518 #8524 #8445 #8449`
- vLLM/SGLang follow-up: vLLM `#46142 #46109 #46148 #46080 #46076 #46020`; SGLang `#28662 #28639 #28715 #28649 #28629 #28620 #28619 #28616`
- CK/FlashAttention follow-up: CK `#3741 #3650 #3732 #3727 #3726 #3725 #3723 #3716 #3696 #3669 #3616 #3708 #3685`; FlashAttention `#61 #98 #74 #70 #68 #65 #92 #91 #109 #115`

## Conservative Source Rules

- Open PRs: mark as active upstream work only.
- Closed PRs: include only as design/prototype/correctness evidence, never as merged feature.
- CI-only PRs: write as compatibility/testing evidence, not kernel capability.
- gfx1250/RDNA evidence must not be generalized to gfx950/CDNA.

# ROCm-KernelWiki-Q — Claude Code Agent Reference

This file provides the schema, navigation, and query reference for agents working with this knowledge base.

## Schema Reference

### Required Frontmatter Fields

Every page must have:

```yaml
---
id: <type-prefix>-<slug>            # Unique ID, matches page type
title: Human-readable title
type: <page-type>                   # source-pr | source-doc | source-blog | wiki-hardware | wiki-technique | wiki-kernel | wiki-pattern | wiki-language | wiki-migration
architectures: [cdna1, cdna2, ...]  # At least one
tags: [list, of, controlled, tags]  # From data/tags.yaml
confidence: <level>                 # verified | source-reported | inferred | experimental
---
```

### Optional Fields by Type

**source-pr**:
```yaml
repo: org/repo
pr: 1234
author: github-handle
date: 'YYYY-MM-DD'
url: https://github.com/...
source_category: upstream-code
techniques: [list]
hardware_features: [list]
kernel_types: [list]
languages: [list]
captured_at: 'YYYY-MM-DD'
status: merged | open | closed
inclusion_reason: "why this PR matters"
```

**wiki-kernel**:
```yaml
kernel_types: [gemm, attention, ...]
languages: [hip-cpp, ck-dsl, ...]
related: [list-of-page-ids]
sources: [list-of-source-ids]
performance_claims:
  - gpu: MI300X
    dtype: fp16
    shape: "M=1024, N=1024, K=1024"
    metric: TFLOPS
    value: 580
    utilization: 89%
    source_id: doc-xxx
artifact_dir: artifacts/kernels/<name>
reproducibility: concept | pseudocode | snippet | runnable | benchmarked
```

**wiki-hardware**:
```yaml
hardware_features: [mfma, lds, dpp, ...]
related: [list-of-page-ids]
sources: [list-of-source-ids]
cuda_equivalent: <cuda-feature-name>  # For cross-referencing
```

**wiki-migration**:
```yaml
from_architecture: cdna3
to_architecture: cdna4
from_concept: <name>
to_concept: <name>
difficulty: easy | moderate | hard
```

## Controlled Vocabulary

See `data/tags.yaml` for the full tag registry. Key dimensions:

- **architectures**: cdna1, cdna2, cdna3, cdna4
- **hardware_features**: mfma, scaled-mfma, lds, lds-transpose, dpp, gws, dual-cma
- **techniques**: ck-tile, bank-conflict-padding, persistent-kernel, mfma-scheduling, double-buffering
- **kernel_types**: gemm, attention, moe, grouped-gemm, conv, reduction
- **languages**: hip-cpp, ck-dsl, triton-rocm, iree-mlir, assembly

## Navigation Paths

1. **Problem-based**: `queries/by-problem.md` → symptom → pattern → technique
2. **Technique-based**: `queries/by-technique.md` → technique → related hardware/kernels
3. **Hardware-based**: `queries/by-hardware-feature.md` → feature → techniques that use it
4. **Kernel-based**: `queries/by-kernel-type.md` → kernel → implementations + optimizations
5. **Language-based**: `queries/by-language.md` → language → guides + examples
6. **Migration-based**: `wiki/migration/` → step-by-step migration guides

## Query Examples for Agents

```
# Find how to use MFMA on CDNA4
python3 scripts/query.py "MFMA matrix core programming CDNA4"

# Find optimization techniques for memory-bound kernels on MI300X
python3 scripts/query.py --symptom memory-bound --architecture cdna3

# Find all GEMM kernel implementations
python3 scripts/query.py --type kernel --tag gemm

# Find migration guide from CUDA to HIP
python3 scripts/get_page.py migration-cuda-to-hip

# Find all pages related to a specific hardware feature
python3 scripts/query.py --tag dpp --type technique
```

## Data Refresh

```bash
# 1. Update PR candidate ledgers
python3 scripts/refresh_candidate_ledger.py

# 2. Generate source pages from new PRs
python3 scripts/generate-pr-pages.py

# 3. Author wiki pages citing sources

# 4. Regenerate indices
python3 scripts/generate-indices.py

# 5. Validate
python3 scripts/validate.py

# 6. Update cutoff
# Edit data/refresh-cutoff.yaml
```

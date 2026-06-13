# ROCm-KernelWiki-Q

GPU kernel optimization knowledge graph for AMD ROCm / CDNA architectures.

A three-layer knowledge base (Sources → Wiki → Queries) covering CDNA1–CDNA4 GPU kernel
optimization techniques, libraries, and migration patterns. Designed for LLM agent retrieval
and human exploration.

Inspired by [KernelWiki-Q](../KernelWiki-Q/) (NVIDIA Blackwell/Hopper).

## Architecture

```
sources/          Layer 1: Raw data — PRs, blogs, docs, contests
    → wiki/       Layer 2: Synthesized knowledge pages
        → queries/  Layer 3: Auto-generated cross-reference indices
```

## Coverage

| Dimension | Contents |
|-----------|----------|
| **Architectures** | CDNA1 (MI100), CDNA2 (MI250), CDNA3 (MI300X/A), CDNA4 (MI350X) |
| **Libraries** | Composable Kernel (CK), hipBLASLt, ROCm FlashAttention · _planned: rocBLAS, MIOpen, rocWMMA_ |
| **Languages** | HIP C++, CK DSL, Triton-ROCm, GCN Assembly · _planned: IREE/MLIR_ |
| **Kernel Types** | GEMM, Attention, Grouped-GEMM, Conv, MoE (source PRs) · _wiki synthesis: Attention; planned: GEMM/MoE/Conv pages_ |
| **Techniques** | MFMA scheduling, bank-conflict avoidance, DPP patterns, persistent kernels, LDS transpose |
| **Migration** | CUDA→HIP · _planned: CDNA3→CDNA4, WMMA→MFMA_ |

> Coverage reflects what is **ingested as source PRs/docs** vs **synthesized into wiki pages**. Rows marked _planned_ are tracked in `data/tags.yaml` but not yet backed by content — agents should not claim coverage for them.

## Quick Start

```bash
# Search
python3 scripts/query.py "how to optimize GEMM on MI300X"

# Get a page
python3 scripts/get_page.py hw-mfma-matrix-core

# Browse by dimension (short --type names accepted: hardware, technique, kernel, pr, ...)
python3 scripts/query.py --tag mfma --type hardware
python3 scripts/query.py --technique bank-conflict-padding --architecture cdna3

# Load the manifest of all pages (machine-readable)
cat queries/pages.json        # or queries/INDEX.md for a browsable list

# Validate all pages (add --strict in CI to gate on warnings too)
python3 scripts/validate.py

# Regenerate indices + manifest
python3 scripts/generate-indices.py

# Enrich PR pages with GitHub body / changed-files summaries (requires gh)
python3 scripts/enrich_pr_pages.py

# Re-classify PR pages from body + changed files (fills kernel_types/languages/techniques)
python3 scripts/reclassify_pr_pages.py

# Repair PR metadata when the URL points to a different GitHub repo
python3 scripts/repair_pr_pages.py
```

## Page Types

| Type | ID Prefix | Purpose |
|------|-----------|---------|
| source-pr | `pr-` | Merged PRs from tracked repos |
| source-doc | `doc-` | Official AMD docs, whitepapers |
| source-blog | `blog-` | Community blog posts, tutorials |
| wiki-hardware | `hw-` | Hardware feature pages (MFMA, LDS, DPP...) |
| wiki-technique | `technique-` | Optimization techniques |
| wiki-kernel | `kernel-` | Kernel case studies with perf claims |
| wiki-pattern | `pattern-` | Problem → solution diagnosis |
| wiki-language | `lang-` | DSL/language guides |
| wiki-migration | `migration-` | Architecture migration guides |

## Confidence Levels

| Level | Meaning |
|-------|---------|
| `verified` | Official doc + upstream code evidence |
| `source-reported` | ≥1 authoritative source |
| `inferred` | Synthesized from multiple sources |
| `experimental` | Undocumented, assembly-level tricks |

## License

MIT

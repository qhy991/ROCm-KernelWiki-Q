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
| **Libraries** | Composable Kernel (CK), hipBLASLt, rocBLAS, MIOpen, rocWMMA, ROCm FlashAttention |
| **Languages** | HIP C++, CK DSL, Triton-ROCm, IREE/MLIR, GCN Assembly |
| **Kernel Types** | GEMM, Attention, MoE, Grouped-GEMM, Conv, Reduction |
| **Techniques** | MFMA scheduling, bank-conflict avoidance, DPP patterns, persistent kernels, LDS transpose |
| **Migration** | CUDA→HIP, CDNA3→CDNA4, WMMA→MFMA |

## Quick Start

```bash
# Search
python3 scripts/query.py "how to optimize GEMM on MI300X"

# Get a page
python3 scripts/get_page.py hw-mfma-matrix-core

# Browse by dimension
python3 scripts/query.py --tag mfma --type hardware
python3 scripts/query.py --technique bank-conflict-padding --architecture cdna3

# Validate all pages
python3 scripts/validate.py

# Regenerate indices
python3 scripts/generate-indices.py

# Repair PR metadata when URL points to a different GitHub repo
python3 scripts/repair_pr_pages.py

# Enrich PR pages with GitHub body / changed-files summaries (requires gh)
python3 scripts/enrich_pr_pages.py
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

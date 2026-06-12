#!/usr/bin/env python3
"""
Generate KernelWiki-Q source-pr pages from GitHub PR JSON data.

Usage:
    # From CK PRs:
    gh pr list --repo ROCm/composable_kernel --state merged --limit 100 \
        --json number,title,author,mergedAt,url > /tmp/ck_prs.json
    python3 scripts/generate_pr_pages.py --json /tmp/ck_prs.json --repo composable_kernel

    # From hipBLASLt PRs:
    gh pr list --repo ROCm/hipBLASLt --state merged --limit 100 \
        --json number,title,author,mergedAt,url > /tmp/hipblaslt_prs.json
    python3 scripts/generate_pr_pages.py --json /tmp/hipblaslt_prs.json --repo hipblaslt

    # From flash-attention PRs:
    gh pr list --repo ROCm/flash-attention --state merged --limit 50 \
        --json number,title,author,mergedAt,url > /tmp/fa_prs.json
    python3 scripts/generate_pr_pages.py --json /tmp/fa_prs.json --repo flash-attention
"""

import argparse
import json
import os
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = ROOT / "sources" / "prs"

REPO_MAP = {
    "composable_kernel": {
        "org": "ROCm",
        "full_name": "ROCm/composable_kernel",
        "description": "Composable Kernel library",
    },
    "hipblaslt": {
        "org": "ROCm",
        "full_name": "ROCm/hipBLASLt",
        "description": "hipBLASLt GEMM tuning library",
    },
    "flash-attention": {
        "org": "ROCm",
        "full_name": "ROCm/flash-attention",
        "description": "ROCm Flash Attention",
    },
}

# Auto-classify PRs by title keywords
TAG_RULES = [
    (r"\bgemm\b", ["gemm"]),
    (r"\battention\b|\bfmha\b|\bsdotpa\b|\bflash\b", ["attention", "flash-attention"]),
    (r"\bmoe\b|\bmixture.of.expert\b", ["moe", "grouped-gemm"]),
    (r"\bgrouped.gemm\b", ["grouped-gemm"]),
    (r"\bconv\b|\bconvolution\b", ["conv"]),
    (r"\breduc\b", ["reduction"]),
    (r"\bmfma\b|\bmatrix.core\b", ["mfma"]),
    (r"\bfp8\b|\bbf8\b|\bf8\b", ["fp8"]),
    (r"\bfp4\b|\bnvfp4\b|\bmxfp4\b", ["fp4"]),
    (r"\bbf16\b", ["bf16"]),
    (r"\bfp16\b|\bf16\b", ["fp16"]),
    (r"\bfused?\b", ["fused-kernel"]),
    (r"\boptimi[zs]\b|\bperf\b|\bspeed\b|\btuning\b|\btune\b", ["optimization"]),
    (r"\bck.tile\b|\bck_tile\b", ["ck-tile"]),
    (r"\btriton\b", ["triton-rocm"]),
    (r"\btensile\b", ["tensilelite"]),
    (r"\bquantiz\b|\bquant\b|\bscale\b", ["quantization"]),
    (r"\bgfx94\b|\bmi300\b|\bcdna3\b", ["cdna3"]),
    (r"\bgfx95\b|\bmi350\b|\bcdna4\b", ["cdna4"]),
    (r"\blds\b|\blocal.data\b", ["lds"]),
    (r"\bsoftmax\b", ["softmax"]),
    (r"\bnorm\b|\blayer.?norm\b|\brms.?norm\b", ["layernorm"]),
]

KERNEL_TYPE_RULES = [
    (r"\bgemm\b", "gemm"),
    (r"\battention\b|\bfmha\b|\bflash\b", "attention"),
    (r"\bmoe\b", "moe"),
    (r"\bgrouped.gemm\b", "grouped-gemm"),
    (r"\bconv\b", "conv"),
    (r"\breduc\b", "reduction"),
    (r"\bsoftmax\b", "softmax"),
    (r"\bnorm\b|\blayer.?norm\b|\brms.?norm\b", "layernorm"),
]

LANGUAGE_RULES = [
    (r"\btriton\b", "triton-rocm"),
    (r"\bck.tile\b|\bck_tile\b", "ck-dsl"),
    (r"\bassembly\b|\bisa\b", "assembly"),
]


def classify_pr(title: str) -> dict:
    """Auto-classify a PR by title keywords."""
    title_lower = title.lower()

    tags = set()
    kernel_types = set()
    languages = set()
    techniques = set()

    for pattern, tag_list in TAG_RULES:
        if re.search(pattern, title_lower):
            tags.update(tag_list)

    for pattern, kt in KERNEL_TYPE_RULES:
        if re.search(pattern, title_lower):
            kernel_types.add(kt)

    for pattern, lang in LANGUAGE_RULES:
        if re.search(pattern, title_lower):
            languages.add(lang)

    # Determine inclusion
    is_kernel_related = bool(tags) or any(
        kw in title_lower
        for kw in ["kernel", "gemm", "attention", "mfma", "moe", "fused", "perf",
                    "optim", "ck_tile", "tensile", "tuning", "fp8", "bf16", "fp16"]
    )

    return {
        "tags": sorted(tags),
        "kernel_types": sorted(kernel_types),
        "languages": sorted(languages),
        "techniques": sorted(techniques),
        "is_kernel_related": is_kernel_related,
    }


def generate_pr_page(pr: dict, repo_key: str) -> str:
    """Generate a source-pr page markdown."""
    repo = REPO_MAP.get(repo_key, {})
    org = repo.get("org", "ROCm")
    full_name = repo.get("full_name", repo_key)

    number = pr.get("number", 0)
    title = pr.get("title", "Untitled")
    canonical_url = f"https://github.com/{full_name}/pull/{number}"
    url = pr.get("url") or canonical_url
    if "github.com/" in url:
        url_repo = url.split("github.com/", 1)[-1].split("/")
        if len(url_repo) >= 2:
            actual_repo = f"{url_repo[0]}/{url_repo[1]}"
            if actual_repo.lower() != full_name.lower():
                url = f"https://github.com/{actual_repo}/pull/{number}"
                full_name = actual_repo
    merged_at = pr.get("mergedAt", "")
    if merged_at:
        merged_at = merged_at[:10]  # YYYY-MM-DD

    author_data = pr.get("author", {})
    if isinstance(author_data, dict):
        author = author_data.get("login", "unknown")
    else:
        author = str(author_data)

    classification = classify_pr(title)
    tags = classification["tags"]
    kernel_types = classification["kernel_types"]
    languages = classification["languages"]

    # Default tags
    if not tags:
        tags = ["rocm-kernel"]
    tags = list(set(tags + ["rocm", repo_key]))

    # Determine architectures
    archs = []
    title_lower = title.lower()
    if "gfx950" in title_lower or "mi350" in title_lower or "cdna4" in title_lower:
        archs.append("cdna4")
    elif "gfx942" in title_lower or "gfx940" in title_lower or "mi300" in title_lower or "cdna3" in title_lower:
        archs.append("cdna3")
    elif "gfx90a" in title_lower or "mi250" in title_lower or "cdna2" in title_lower:
        archs.append("cdna2")
    if not archs:
        archs = ["cdna2", "cdna3", "cdna4"]

    # Determine inclusion reason
    if classification["is_kernel_related"]:
        inclusion_reason = "kernel-related changes"
    else:
        inclusion_reason = "may contain relevant infrastructure changes"

    body = pr.get("body", "") or ""
    body_lines = [line.strip() for line in body.splitlines() if line.strip()]
    summary = ""
    for line in body_lines:
        if line.startswith("<!--"):
            continue
        summary = line[:500]
        break
    if not summary and body_lines:
        summary = " ".join(body_lines)[:500]

    # Build page ID
    page_id = f"pr-{repo_key}-{number}"

    # Escape title for YAML
    safe_title = title.replace('"', '\\"')
    safe_full = full_name
    safe_incl = inclusion_reason
    safe_url = url
    safe_archs = ", ".join(archs)
    safe_tags = ", ".join(tags)
    safe_kt = ", ".join(kernel_types)
    safe_langs = ", ".join(languages)
    safe_merged = merged_at or "unknown"
    today = date.today().isoformat()

    # Build YAML frontmatter
    frontmatter = f"""---
id: {page_id}
type: source-pr
repo: {safe_full}
pr: {number}
title: "{safe_title}"
author: {author}
date: '{safe_merged}'
url: {safe_url}
source_category: upstream-code
architectures: [{safe_archs}]
tags: [{safe_tags}]
kernel_types: [{safe_kt}]
languages: [{safe_langs}]
captured_at: '{today}'
status: merged
inclusion_reason: "{safe_incl}"
confidence: source-reported
---

# {title}

Merged PR #{number} in [{safe_full}]({safe_url}).

**Author:** {author}
**Merged:** {safe_merged}

## Description

{("> " + summary.replace(chr(10), chr(10) + "> ")) if summary else f"> Auto-imported from [{safe_full} #{number}]({safe_url})."}

See the PR for full details including code changes and review discussion.

## References

- [PR #{number}]({safe_url})
"""

    return frontmatter


def main():
    parser = argparse.ArgumentParser(description="Generate PR source pages from GitHub JSON")
    parser.add_argument("--json", required=True, help="Path to PR JSON file from gh pr list")
    parser.add_argument("--repo", required=True, help="Repository key (composable_kernel, hipblaslt, flash-attention)")
    parser.add_argument("--outdir", default=None, help="Output directory (default: sources/prs/<repo>)")
    parser.add_argument("--filter-kernel", action="store_true", help="Only include kernel-related PRs")
    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}")
        return 1

    with open(args.json) as f:
        prs = json.load(f)

    repo_key = args.repo
    out_dir = args.outdir or str(SOURCES_DIR / repo_key)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    skipped = 0

    for pr in prs:
        title = pr.get("title", "")
        number = pr.get("number", 0)

        # Optional filter
        if args.filter_kernel:
            classification = classify_pr(title)
            if not classification["is_kernel_related"]:
                skipped += 1
                continue

        page_content = generate_pr_page(pr, repo_key)
        page_path = os.path.join(out_dir, f"PR-{number}.md")

        with open(page_path, "w") as f:
            f.write(page_content)

        count += 1

    print(f"Generated {count} PR pages in {out_dir}")
    if skipped:
        print(f"Skipped {skipped} non-kernel PRs")
    return 0


if __name__ == "__main__":
    exit(main())

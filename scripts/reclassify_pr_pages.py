#!/usr/bin/env python3
"""
Re-classify source-pr pages using the FULL signal already on disk:
title + description summary + the `## Changed Files` paths that enrich_pr_pages.py
fetched. The original generate_pr_pages.classify_pr() only read the PR title, so
~70% of pages have empty kernel_types and ~78% empty languages even though the
changed-file paths make them trivially recoverable.

This pass only ever ADDS information (union with existing values) and only narrows
`architectures` when a specific CDNA gfx/MI token is found — it never fabricates an
architecture. RDNA/gfx1x-only pages are flagged with an `rdna` tag for review rather
than silently asserted as CDNA.

Usage:
    python3 scripts/reclassify_pr_pages.py            # apply
    python3 scripts/reclassify_pr_pages.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = ROOT / "sources" / "prs"

# --- classification rules (regex matched against title + body + changed paths) ---

KERNEL_TYPE_RULES = [
    ("grouped-gemm", r"grouped[_-]?gemm|group_gemm"),
    ("gemm", r"\bgemm\b|_gemm|/gemm|universal_gemm"),
    ("attention", r"attention|fmha|flash[_-]?attn|flash_fwd|flash_bwd|\bmha\b|sdpa|\bmqa\b|\bgqa\b|kv[_-]?cache|paged|rotary|\brope\b|alibi|causal|varlen|seqlen|\bbshd\b"),
    ("moe", r"\bmoe\b|mixture[_-]?of[_-]?expert|moe_|fused_moe"),
    ("conv", r"\bconv\b|conv2d|conv3d|convolution|conv_fwd|conv_bwd|grouped_conv"),
    ("reduction", r"\breduce\b|reduction|reduc_"),
    ("softmax", r"softmax"),
    ("layernorm", r"layer[_-]?norm|layernorm"),
    ("rmsnorm", r"rms[_-]?norm|rmsnorm"),
    ("embedding", r"embedding|embed_"),
]

# language is keyed off file extensions in the changed-file paths, plus a few tokens
EXT_LANG = {
    ".cpp": "hip-cpp", ".hpp": "hip-cpp", ".h": "hip-cpp", ".hip": "hip-cpp",
    ".cu": "hip-cpp", ".cuh": "hip-cpp", ".cc": "hip-cpp", ".cxx": "hip-cpp",
    ".py": "python", ".mlir": "iree-mlir", ".s": "assembly", ".asm": "assembly",
}

TECHNIQUE_RULES = [
    ("mfma-scheduling", r"interwave|intrawave|mfma[_-]?schedul|warp[_-]?schedul|\bscheduler\b"),
    ("double-buffering", r"double[_-]?buffer|prefetch|software[_-]?pipelin"),
    ("persistent-kernel", r"persistent"),
    ("bank-conflict-padding", r"bank[_-]?conflict|lds[_-]?pad|swizzl"),
    ("async-copy", r"async[_-]?copy|async_load|cp_async|global[_-]?load[_-]?lds"),
    ("register-tiling", r"register[_-]?til"),
    ("vectorized-load", r"vector[_-]?load|vectoriz"),
    ("ck-tile-programming", r"ck_tile|ck-tile"),
]

HW_RULES = [
    ("mfma", r"\bmfma\b|matrix[_-]?core|smfmac|\bwmma\b"),
    ("scaled-mfma", r"scaled[_-]?mfma|\bmxfp|\bmx[_-]?gemm|mx_gemm"),
    ("block-scale", r"block[_-]?scal|\bmxfp4\b|\bmxfp8\b"),
    ("lds-transpose", r"ds_read_tr|read[_-]?transpose|transpose.{0,12}lds|lds.{0,12}transpose"),
    ("lds", r"\blds\b|local[_-]?data[_-]?share|shared[_-]?mem"),
    ("dpp", r"\bdpp\b"),
    ("gws", r"\bgws\b|global[_-]?wave"),
]

ARCH_RULES = [
    ("cdna4", r"gfx950|mi350|mi355|cdna4|cdna 4"),
    ("cdna3", r"gfx942|gfx940|mi300|cdna3|cdna 3"),
    ("cdna2", r"gfx90a|mi250|mi210|cdna2|cdna 2"),
    ("cdna1", r"gfx908|mi100|cdna1|cdna 1"),
]
RDNA_RE = r"gfx12|gfx11|gfx10|\brdna\b|\bnavi\b"
DEFAULT_ARCH_TRIPLE = ["cdna2", "cdna3", "cdna4"]


def extract_frontmatter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    return (yaml.safe_load(m.group(1)) or {}), m.group(2)


def changed_paths(body: str) -> list[str]:
    """Pull file paths out of the '## Changed Files' section."""
    paths = []
    in_section = False
    for line in body.splitlines():
        if line.strip().startswith("## Changed Files"):
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            m = re.match(r"\s*-\s+`([^`]+)`", line)
            if m:
                paths.append(m.group(1))
    return paths


def description_summary(body: str) -> str:
    """First quoted line under '## Description'."""
    desc = body.split("## Description", 1)[-1]
    for line in desc.splitlines():
        s = line.strip()
        if s.startswith(">"):
            content = s[1:].strip()
            if content and not content.startswith("Auto-imported from"):
                return content
    return ""


def _match_first(rules: list[tuple[str, str]], text: str) -> list[str]:
    out = []
    for label, pat in rules:
        if re.search(pat, text, re.IGNORECASE):
            out.append(label)
    return out


def classify(title: str, body: str, paths: list[str], repo: str = "") -> dict:
    path_text = " ".join(paths).lower()
    text = f"{title}\n{description_summary(body)}\n{path_text}".lower()

    kernel_types = _match_first(KERNEL_TYPE_RULES, text)
    # Repo context: the ROCm/flash-attention repo exists to ship attention kernels,
    # so even a title-only stub there is attention-related.
    if "flash-attention" in repo.lower() and "attention" not in kernel_types:
        kernel_types.append("attention")
    # if grouped-gemm matched, gemm is implied
    if "grouped-gemm" in kernel_types and "gemm" not in kernel_types:
        kernel_types.append("gemm")

    languages = set()
    for p in paths:
        ext = "." + p.rsplit(".", 1)[-1].lower() if "." in p else ""
        if ext in EXT_LANG:
            languages.add(EXT_LANG[ext])
        if "ck_tile" in p.lower() or "ck/tile" in p.lower():
            languages.add("ck-dsl")
    if re.search(r"\btriton\b", text):
        languages.add("triton-rocm")
    if re.search(r"\bck_tile\b|\bck-tile\b", text):
        languages.add("ck-dsl")

    techniques = _match_first(TECHNIQUE_RULES, text)
    hardware_features = _match_first(HW_RULES, text)

    archs_cdna = _match_first(ARCH_RULES, text)
    is_rdna = bool(re.search(RDNA_RE, text)) and not archs_cdna

    return {
        "kernel_types": sorted(set(kernel_types)),
        "languages": sorted(languages),
        "techniques": sorted(set(techniques)),
        "hardware_features": sorted(set(hardware_features)),
        "archs_cdna": sorted(set(archs_cdna)),
        "is_rdna": is_rdna,
    }


def reclassify_page(path: Path, dry_run: bool, stats: Counter) -> bool:
    text = path.read_text(encoding="utf-8")
    fm, body = extract_frontmatter(text)
    if fm.get("type") != "source-pr":
        return False

    paths = changed_paths(body)
    cls = classify(fm.get("title", ""), body, paths, fm.get("repo", ""))
    changed = False

    def merge_list(field: str, new_vals: list[str]):
        nonlocal changed
        cur = fm.get(field) or []
        if isinstance(cur, str):
            cur = [cur]
        merged = sorted(set(cur) | set(new_vals))
        if merged != sorted(set(cur)):
            was_empty = not cur
            fm[field] = merged
            changed = True
            if was_empty and merged:
                stats[f"filled_{field}"] += 1

    merge_list("kernel_types", cls["kernel_types"])
    merge_list("languages", cls["languages"])
    merge_list("techniques", cls["techniques"])
    merge_list("hardware_features", cls["hardware_features"])

    # Narrow architectures only when we found specific CDNA evidence and the page
    # is currently on the meaningless default triple.
    cur_arch = fm.get("architectures") or []
    if cls["archs_cdna"] and sorted(cur_arch) == sorted(DEFAULT_ARCH_TRIPLE):
        if sorted(cls["archs_cdna"]) != sorted(cur_arch):
            fm["architectures"] = cls["archs_cdna"]
            changed = True
            stats["narrowed_arch"] += 1

    # Flag RDNA-only pages for review rather than asserting CDNA.
    if cls["is_rdna"]:
        tags = fm.get("tags") or []
        if "rdna" not in tags:
            fm["tags"] = sorted(set(tags) | {"rdna"})
            changed = True
            stats["flagged_rdna"] += 1

    # Cheap retrieval win: give the page a `description` for query.py output.
    if not fm.get("description"):
        summary = description_summary(body)
        if summary:
            fm["description"] = summary[:160]
            changed = True
            stats["added_description"] += 1

    if changed and not dry_run:
        new_text = "---\n" + yaml.safe_dump(fm, sort_keys=False).strip() + "\n---\n\n" + body.lstrip("\n")
        path.write_text(new_text, encoding="utf-8")
    if changed:
        stats["pages_changed"] += 1
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-classify PR pages from body + changed files")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stats: Counter = Counter()
    total = 0
    for path in sorted(SOURCES_DIR.rglob("PR-*.md")):
        total += 1
        reclassify_page(path, args.dry_run, stats)

    print(f"Scanned {total} PR pages{' (dry-run)' if args.dry_run else ''}")
    for k in ("pages_changed", "filled_kernel_types", "filled_languages",
              "filled_techniques", "filled_hardware_features", "narrowed_arch",
              "flagged_rdna", "added_description"):
        print(f"  {k:26} {stats[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

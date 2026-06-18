#!/usr/bin/env python3
"""Infer ROCm architecture tags for source-pr pages from explicit evidence.

This pass only narrows pages that are still on the default
`[cdna2, cdna3, cdna4]` triple. It does not guess CDNA from repository context;
the page text must contain architecture evidence such as gfx950, gfx1201,
MI300, CDNA4, or RDNA4.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = ROOT / "sources" / "prs"
DEFAULT_ARCH_TRIPLE = ["cdna2", "cdna3", "cdna4"]

ARCH_PATTERNS = [
    ("cdna4", r"\bgfx950\b|\bmi35[05]x?\b|\bcdna\s*4\b"),
    ("cdna3", r"\bgfx94[02]\b|\bmi300[ax]?\b|\bcdna\s*3\b"),
    ("cdna2", r"\bgfx90a\b|\bmi25[0x]?\b|\bmi210\b|\bcdna\s*2\b"),
    ("cdna1", r"\bgfx908\b|\bmi100\b|\bcdna\s*1\b"),
    ("rdna4", r"\bgfx12[0-9a-z]*\b|\brdna\s*4\b|\bnavi4[0-9]\b"),
    ("rdna3", r"\bgfx11[0-9a-z]*\b|\brdna\s*3\b|\bnavi3[0-9]\b"),
    ("rdna2", r"\bgfx10[0-9a-z]*\b|\brdna\s*2\b|\bnavi2[0-9]\b"),
]


@dataclass
class Stats:
    scanned: int = 0
    changed: int = 0
    narrowed_default_arch: int = 0
    added_rdna_tag: int = 0


def extract_frontmatter(text: str) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text
    return yaml.safe_load(match.group(1)) or {}, match.group(2)


def infer_architectures(title: str, body: str) -> list[str]:
    text = f"{title}\n{body}".lower()
    archs = []
    for arch, pattern in ARCH_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            archs.append(arch)
    return archs


def update_page_architectures(path: Path, dry_run: bool, stats: Stats) -> bool:
    text = path.read_text(encoding="utf-8")
    fm, body = extract_frontmatter(text)
    if fm.get("type") != "source-pr":
        return False

    stats.scanned += 1
    current_archs = fm.get("architectures") or []
    if isinstance(current_archs, str):
        current_archs = [current_archs]
    if sorted(current_archs) != sorted(DEFAULT_ARCH_TRIPLE):
        return False

    inferred = infer_architectures(fm.get("title", ""), body)
    if not inferred:
        return False

    fm["architectures"] = inferred
    stats.narrowed_default_arch += 1

    if any(arch.startswith("rdna") for arch in inferred):
        tags = fm.get("tags") or []
        if "rdna" not in tags:
            fm["tags"] = sorted(set(tags) | {"rdna"})
            stats.added_rdna_tag += 1

    if not dry_run:
        new_text = "---\n" + yaml.safe_dump(fm, sort_keys=False).strip() + "\n---\n\n" + body.lstrip("\n")
        path.write_text(new_text, encoding="utf-8")

    stats.changed += 1
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer ROCm architecture values for PR pages")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    parser.add_argument("--repo", default=None, help="Only process one PR source directory")
    args = parser.parse_args()

    stats = Stats()
    roots = [SOURCES_DIR / args.repo] if args.repo else sorted(p for p in SOURCES_DIR.iterdir() if p.is_dir())
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("PR-*.md")):
            if update_page_architectures(path, dry_run=args.dry_run, stats=stats):
                print(f"inferred {path.relative_to(ROOT)}")

    print(f"Scanned {stats.scanned} PR pages{' (dry-run)' if args.dry_run else ''}")
    print(f"  changed                 {stats.changed}")
    print(f"  narrowed_default_arch   {stats.narrowed_default_arch}")
    print(f"  added_rdna_tag          {stats.added_rdna_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

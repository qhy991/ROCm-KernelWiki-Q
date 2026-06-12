#!/usr/bin/env python3
"""
Repair source-pr pages whose `repo` field does not match the GitHub URL.

Fixes canonical repo metadata and page IDs derived from the URL host repo.
Optionally enriches repaired pages via `enrich_pr_pages.enrich_page`.

Usage:
    python3 scripts/repair_pr_pages.py
    python3 scripts/repair_pr_pages.py --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = ROOT / "sources" / "prs"


def repo_slug(full_name: str) -> str:
    return full_name.split("/", 1)[-1].lower().replace("-", "_")


def parse_repo_from_url(url: str) -> str | None:
    if "github.com/" not in url:
        return None
    tail = url.split("github.com/", 1)[-1]
    parts = tail.split("/")
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def repair_page(path: Path, dry_run: bool = False) -> bool:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return False

    fm = yaml.safe_load(match.group(1)) or {}
    body = match.group(2)
    if fm.get("type") != "source-pr":
        return False

    declared_repo = fm.get("repo", "")
    url_repo = parse_repo_from_url(fm.get("url", ""))
    if not url_repo or not declared_repo:
        return False
    if url_repo.lower() == declared_repo.lower():
        return False

    number = fm.get("pr")
    slug = repo_slug(url_repo)
    fm["repo"] = url_repo
    fm["url"] = f"https://github.com/{url_repo}/pull/{number}"
    fm["id"] = f"pr-{slug}-{number}"
    fm["source_category"] = "cross-repo"
    tags = set(fm.get("tags") or [])
    tags.add("cross-repo")
    fm["tags"] = sorted(tags)

    body = re.sub(
        r"Merged PR #\d+ in \[.*?\]\([^)]+\)\.",
        f"Merged PR #{number} in [{url_repo}](https://github.com/{url_repo}/pull/{number}).",
        body,
        count=1,
    )
    body = re.sub(
        r"\[PR #\d+\]\([^)]+\)",
        f"[PR #{number}](https://github.com/{url_repo}/pull/{number})",
        body,
    )
    new_text = "---\n" + yaml.safe_dump(fm, sort_keys=False).strip() + "\n---\n\n" + body

    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair PR pages with mismatched repo metadata")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repaired = 0
    for path in sorted(SOURCES_DIR.rglob("PR-*.md")):
        if repair_page(path, dry_run=args.dry_run):
            repaired += 1
            print(f"repaired {path.relative_to(ROOT)}")

    print(f"Repaired {repaired} pages")
    if not args.dry_run and repaired:
        from enrich_pr_pages import enrich_page

        enriched = 0
        for path in sorted(SOURCES_DIR.rglob("PR-*.md")):
            if enrich_page(path):
                enriched += 1
        print(f"Enriched {enriched} pages after repair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

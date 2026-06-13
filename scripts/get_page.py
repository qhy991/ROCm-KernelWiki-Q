#!/usr/bin/env python3
"""
Fetch a ROCm-KernelWiki-Q page by id.

Usage:
    python3 scripts/get_page.py kernel-flash-attention-rocm
    python3 scripts/get_page.py kernel-flash-attention-rocm --body-only
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = ROOT / "wiki"
SOURCES_DIR = ROOT / "sources"


def extract_frontmatter(filepath):
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    match = re.match(r"^---\n(.*?)\n---\n?", content, re.DOTALL)
    if not match:
        return None, content
    import yaml

    try:
        fm = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None, content
    body = content[match.end() :]
    return fm, body


def find_page(page_id):
    for dir_path in [WIKI_DIR, SOURCES_DIR]:
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.rglob("*.md")):
            fm, body = extract_frontmatter(md_file)
            if fm and fm.get("id") == page_id:
                return fm, body, md_file.relative_to(ROOT)
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Get a ROCm-KernelWiki-Q page by id")
    parser.add_argument("page_id", help="Page id (from query.py results)")
    parser.add_argument(
        "--body-only",
        action="store_true",
        help="Print markdown body without YAML frontmatter",
    )
    args = parser.parse_args()

    fm, body, path = find_page(args.page_id)
    if fm is None:
        print(f"Page not found: {args.page_id}", file=sys.stderr)
        return 1

    if args.body_only:
        print(body.lstrip("\n"), end="")
        if not body.endswith("\n"):
            print()
    else:
        title = fm.get("title", args.page_id)
        print(f"# {title}")
        print(f"ID: {args.page_id}")
        print(f"Path: {path}")
        print()
        print(body.lstrip("\n"), end="")
        if not body.endswith("\n"):
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

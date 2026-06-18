#!/usr/bin/env python3
"""
Enrich existing source-pr pages with GitHub PR summaries.

Uses `gh pr view` for body, merge date, and changed files. When the PR body
is empty, falls back to a files-based summary derived from the title and diff.

Usage:
    python3 scripts/enrich_pr_pages.py
    python3 scripts/enrich_pr_pages.py --repo composable_kernel --limit 20
    python3 scripts/enrich_pr_pages.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = ROOT / "sources" / "prs"

SKIP_BODY_PREFIXES = (
    "## proposed changes",
    "## summary",
    "## description",
    "## motivation",
    "## checklist",
    "---",
)

KERNEL_FILE_HINTS = (
    ".cu", ".cpp", ".hpp", ".h", ".hip", ".cuh", ".mlir", ".py", ".yaml", ".yml"
)


def extract_frontmatter(text: str) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text
    fm = yaml.safe_load(match.group(1)) or {}
    return fm, match.group(2)


def is_stub_page(body: str) -> bool:
    if "See the PR for full details" not in body:
        return False
    desc = body.split("## Description", 1)[-1]
    for line in desc.splitlines():
        line = line.strip()
        if not line.startswith(">"):
            continue
        content = line[1:].strip()
        if not content:
            continue
        if content.startswith("Auto-imported from"):
            return True
        if content.lower().startswith(SKIP_BODY_PREFIXES):
            return True
    return "> Auto-imported from" in desc


def summarize_body(body: str, title: str) -> str:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("<!--"):
            continue
        lowered = line.lower().rstrip(":")
        if lowered in SKIP_BODY_PREFIXES:
            continue
        if lowered.startswith("## "):
            continue
        if len(line) >= 12:
            return line[:500]
    return ""


def summarize_files(files: list[dict], title: str) -> str:
    if not files:
        return title[:500]

    kernel_paths = []
    other_paths = []
    for item in files:
        path = item.get("path", "")
        if any(path.endswith(ext) or ext in path for ext in KERNEL_FILE_HINTS):
            kernel_paths.append(path)
        else:
            other_paths.append(path)

    focus = kernel_paths[:6] or other_paths[:6]
    path_text = ", ".join(f"`{path}`" for path in focus)
    more = len(files) - len(focus)
    suffix = f" (+{more} more files)" if more > 0 else ""
    return f"{title}. Changed {len(files)} files including {path_text}{suffix}."[:500]


def build_summary(pr_data: dict) -> str:
    title = pr_data.get("title", "").strip()
    body = pr_data.get("body", "") or ""
    files = pr_data.get("files", []) or []

    summary = summarize_body(body, title)
    if summary and summary.lower().rstrip(":") not in SKIP_BODY_PREFIXES:
        return summary
    return summarize_files(files, title)


def pr_status_and_date(pr_data: dict, fallback_date: str = "unknown") -> tuple[str, str]:
    """Return canonical wiki status/date from GitHub PR metadata."""
    merged_at = pr_data.get("mergedAt")
    if merged_at:
        return "merged", merged_at[:10]

    state = str(pr_data.get("state") or "").lower()
    closed_at = pr_data.get("closedAt")
    created_at = pr_data.get("createdAt")
    if state == "closed" and closed_at:
        return "closed", closed_at[:10]
    if state == "open":
        return "open", created_at[:10] if created_at else fallback_date
    return state or "closed", created_at[:10] if created_at else fallback_date


def fetch_pr_data(repo: str, number: int) -> dict:
    cmd = [
        "gh",
        "pr",
        "view",
        str(number),
        "--repo",
        repo,
        "--json",
        "body,closedAt,createdAt,files,mergedAt,state,title,url",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def format_files_section(files: list[dict]) -> str:
    if not files:
        return ""

    lines = ["## Changed Files", ""]
    for item in files[:20]:
        path = item.get("path", "")
        additions = item.get("additions", 0)
        deletions = item.get("deletions", 0)
        lines.append(f"- `{path}` (+{additions}/-{deletions})")
    if len(files) > 20:
        lines.append(f"- ... and {len(files) - 20} more files")
    lines.append("")
    return "\n".join(lines)


def enrich_page(
    path: Path,
    force: bool = False,
    dry_run: bool = False,
    only_unknown_date: bool = False,
) -> bool:
    text = path.read_text(encoding="utf-8")
    fm, body = extract_frontmatter(text)
    if fm.get("type") != "source-pr":
        return False

    repo = fm.get("repo", "")
    number = fm.get("pr")
    if not repo or not number:
        return False

    has_unknown_date = str(fm.get("date", "")).lower() == "unknown"
    if only_unknown_date and not has_unknown_date:
        return False

    if not force and not is_stub_page(body) and not has_unknown_date:
        return False

    try:
        pr_data = fetch_pr_data(repo, int(number))
    except subprocess.CalledProcessError:
        return False

    summary = build_summary(pr_data) or fm.get("title", "").strip()
    if not summary:
        return False

    canonical_url = f"https://github.com/{repo}/pull/{number}"
    status, date_value = pr_status_and_date(pr_data, fallback_date=fm.get("date", "unknown"))
    files_section = format_files_section(pr_data.get("files", []) or [])
    status_label = "Merged" if status == "merged" else "Closed" if status == "closed" else "Opened"

    new_body = f"""# {fm.get('title', path.stem)}

{status_label} PR #{number} in [{repo}]({canonical_url}).

**Author:** {fm.get('author', 'unknown')}
**{status_label}:** {date_value}

## Description

> {summary}

{files_section}
See the PR for full details including code changes and review discussion.

## References

- [PR #{number}]({canonical_url})
"""

    frontmatter = dict(fm)
    frontmatter["url"] = canonical_url
    frontmatter["status"] = status
    if date_value != "unknown":
        frontmatter["date"] = date_value

    new_text = "---\n" + yaml.safe_dump(frontmatter, sort_keys=False).strip() + "\n---\n\n" + new_body
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich PR pages with GitHub summaries")
    parser.add_argument("--repo", default=None, help="Only enrich one repo key")
    parser.add_argument("--limit", type=int, default=0, help="Max pages to enrich")
    parser.add_argument("--force", action="store_true", help="Re-enrich non-stub pages too")
    parser.add_argument("--only-unknown-date", action="store_true", help="Only repair pages whose date is unknown")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    updated = 0
    for repo_key in ("composable_kernel", "hipblaslt", "flash-attention"):
        if args.repo and args.repo != repo_key:
            continue
        repo_dir = SOURCES_DIR / repo_key
        if not repo_dir.exists():
            continue
        for path in sorted(repo_dir.glob("PR-*.md")):
            if enrich_page(
                path,
                force=args.force,
                dry_run=args.dry_run,
                only_unknown_date=args.only_unknown_date,
            ):
                updated += 1
                print(f"enriched {path.relative_to(ROOT)}")
            if args.limit and updated >= args.limit:
                break
        if args.limit and updated >= args.limit:
            break

    print(f"Enriched {updated} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

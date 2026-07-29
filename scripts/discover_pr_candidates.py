#!/usr/bin/env python3
"""Discover merged GitHub PRs without ingesting them into the knowledge base."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SOURCES_DIR = ROOT / "sources" / "prs"

POSITIVE_SIGNALS = {
    "attention": 5,
    "flash-attn": 5,
    "flash attention": 5,
    "gemm": 4,
    "mfma": 4,
    "xdlops": 4,
    "kernel": 3,
    "streamk": 4,
    "split-k": 4,
    "split kv": 4,
    "lds": 3,
    "bank conflict": 4,
    "prefetch": 3,
    "persistent": 3,
    "triton": 3,
    "gfx942": 5,
    "gfx950": 5,
    "mi300": 5,
    "mi300a": 5,
    "mi300x": 5,
    "mi350": 5,
    "mi350x": 5,
    "mi355": 5,
    "mi355x": 5,
    "performance": 2,
    "perf": 2,
    "optimization": 2,
    "optimize": 2,
}

NEGATIVE_TITLE_PREFIXES = {
    "ci": -100,
    "ci/cd": -100,
    "docs": -100,
    "doc": -100,
    "build(deps)": -100,
    "chore(deps)": -100,
    "test": -20,
}


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_repo(repo: str) -> str:
    return repo.strip().rstrip("/").lower()


def existing_pr_pages(
    sources_dir: Path = SOURCES_DIR,
    repo_aliases: dict[str, str] | None = None,
) -> dict[tuple[str, int], list[dict]]:
    pages: dict[tuple[str, int], list[dict]] = {}
    if not sources_dir.exists():
        return pages
    for path in sources_dir.rglob("PR-*.md"):
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not match:
            continue
        try:
            frontmatter = yaml.safe_load(match.group(1)) or {}
            repo = normalize_repo(str(frontmatter.get("repo", "")))
            repo = (repo_aliases or {}).get(repo, repo)
            number = int(frontmatter.get("pr"))
        except (TypeError, ValueError, yaml.YAMLError):
            continue
        if repo:
            try:
                display_path = str(path.relative_to(ROOT))
            except ValueError:
                display_path = str(path)
            pages.setdefault((repo, number), []).append({
                "path": display_path,
                "status": str(frontmatter.get("status", "")).lower(),
                "captured_at": str(frontmatter.get("captured_at", "")),
            })
    return pages


def existing_pr_keys(sources_dir: Path = SOURCES_DIR) -> set[tuple[str, int]]:
    return set(existing_pr_pages(sources_dir))


def score_candidate(title: str, body: str = "") -> tuple[int, list[str]]:
    title_lower = title.lower()
    text = f"{title}\n{body}".lower()
    score = 0
    signals = []
    for signal, weight in POSITIVE_SIGNALS.items():
        pattern = rf"(?<![a-z0-9]){re.escape(signal)}(?![a-z0-9])"
        if re.search(pattern, text):
            score += weight
            signals.append(signal)

    normalized_title = title_lower.strip()
    leading_labels = []
    while True:
        match = re.match(r"^\[([^\]]+)\]\s*", normalized_title)
        if not match:
            break
        leading_labels.extend(re.findall(r"[a-z0-9_-]+", match.group(1)))
        normalized_title = normalized_title[match.end():].strip()

    conventional_match = re.match(
        r"^([a-z0-9_/-]+)(?:\(([^)]+)\))?\s*:", normalized_title
    )
    conventional_type = conventional_match.group(1) if conventional_match else ""
    conventional_label = ""
    if conventional_match:
        conventional_label = conventional_type
        if conventional_match.group(2):
            conventional_label += f"({conventional_match.group(2)})"

    dependency_scope = (
        conventional_match
        and conventional_type in {"build", "chore"}
        and str(conventional_match.group(2) or "").startswith("deps")
    )
    if dependency_scope:
        score -= 100
        signals.append(f"penalty:{conventional_type}(deps*)")
    for prefix, penalty in NEGATIVE_TITLE_PREFIXES.items():
        if dependency_scope and prefix.startswith(f"{conventional_type}(deps"):
            continue
        plain_prefix = re.match(rf"^{re.escape(prefix)}(?:\s|$)", normalized_title)
        conventional_prefix = (
            conventional_label == prefix
            or ("(" not in prefix and conventional_type == prefix)
        )
        if prefix in leading_labels or conventional_prefix or plain_prefix:
            score += penalty
            signals.append(f"penalty:{prefix}")
            break
    return score, signals


def tracked_repositories() -> list[str]:
    data = load_yaml(DATA_DIR / "tags.yaml")
    repos = []
    for item in data.get("tracked_repos", []):
        org = str(item.get("org", "")).strip()
        name = str(item.get("name", "")).strip()
        if org and name:
            repos.append(f"{org}/{name}")
    return repos


def default_since() -> str | None:
    data = load_yaml(DATA_DIR / "refresh-cutoff.yaml")
    value = (data.get("sources", {}).get("prs", {}).get("cutoff")
             or data.get("refresh_cutoff"))
    return str(value) if value else None


def require_gh() -> None:
    if not shutil.which("gh"):
        raise RuntimeError("GitHub CLI `gh` is required")
    result = subprocess.run(
        ["gh", "auth", "status", "--hostname", "github.com"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"GitHub CLI is not authenticated: {detail}")


def gh_api(endpoint: str, fields: dict[str, str] | None = None) -> object:
    command = [
        "gh", "api", "--method", "GET", endpoint,
    ]
    for key, value in (fields or {}).items():
        command.extend(["-f", f"{key}={value}"])
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"GitHub API request failed for {endpoint}: {detail}")
    return json.loads(result.stdout)


def resolve_repository(repo: str) -> tuple[str, str]:
    repo_data = gh_api(f"repos/{repo}")
    if not isinstance(repo_data, dict):
        raise RuntimeError(f"unexpected repository response for {repo}")
    canonical_repo = str(repo_data.get("full_name") or repo)
    default_branch = str(repo_data.get("default_branch", ""))
    return canonical_repo, default_branch


def fetch_repo_pass(
    repo: str,
    since: str,
    until: str,
    per_page: int,
    max_pages: int,
    default_branch: str,
) -> tuple[list[dict], dict]:
    records_by_number = {}
    scanned_numbers = set()
    pulls_scanned = 0
    duplicates_discarded = 0
    for page in range(1, max_pages + 1):
        response = gh_api(
            f"repos/{repo}/pulls",
            {
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": str(per_page),
                "page": str(page),
            },
        )
        if not isinstance(response, list):
            raise RuntimeError(f"unexpected pull-list response for {repo}")

        pulls_scanned += len(response)
        for item in response:
            number = int(item["number"])
            if number in scanned_numbers:
                duplicates_discarded += 1
            scanned_numbers.add(number)

            merged_at = str(item.get("merged_at") or "")
            closed_at = str(item.get("closed_at") or "")
            current_status = "merged" if merged_at else "closed"
            event_at = merged_at or closed_at
            if event_at and since <= event_at[:10] <= until:
                item["repo"] = repo
                item["default_branch"] = default_branch
                item["_current_status"] = current_status
                item["_event_at"] = event_at
                records_by_number[number] = item

        oldest_updated = min(
            (str(item.get("updated_at") or "")[:10] for item in response),
            default="",
        )
        reached_boundary = (
            len(response) < per_page
            or (oldest_updated and oldest_updated < since)
        )
        if reached_boundary:
            return list(records_by_number.values()), {
                "pages_scanned": page,
                "pulls_scanned": pulls_scanned,
                "duplicates_discarded": duplicates_discarded,
                "scanned_fingerprint": tuple(sorted(scanned_numbers)),
            }

    raise RuntimeError(
        f"{repo} did not reach the {since} boundary after {max_pages} pages; "
        "increase --max-pages instead of accepting truncated discovery"
    )


def record_fingerprint(records: list[dict]) -> tuple:
    return tuple(sorted(
        (
            int(record["number"]),
            str(record.get("_current_status", "")),
            str(record.get("_event_at", "")),
            str((record.get("base") or {}).get("ref", "")),
            str(record.get("merge_commit_sha", "")),
        )
        for record in records
    ))


def fetch_repo(
    repo: str,
    since: str,
    until: str,
    per_page: int,
    max_pages: int,
    stability_passes: int = 2,
    max_scan_passes: int = 4,
    default_branch: str | None = None,
) -> tuple[list[dict], dict]:
    if default_branch is None:
        repo, default_branch = resolve_repository(repo)

    previous_fingerprint = None
    consecutive_stable = 0
    total_pages = 0
    total_pulls = 0
    total_duplicates = 0
    for scan_pass in range(1, max_scan_passes + 1):
        records, pass_coverage = fetch_repo_pass(
            repo,
            since,
            until,
            per_page,
            max_pages,
            default_branch,
        )
        fingerprint = (
            record_fingerprint(records),
            pass_coverage["scanned_fingerprint"],
        )
        if pass_coverage["duplicates_discarded"]:
            consecutive_stable = 0
            previous_fingerprint = None
        elif fingerprint == previous_fingerprint:
            consecutive_stable += 1
        else:
            consecutive_stable = 1
            previous_fingerprint = fingerprint
        total_pages += pass_coverage["pages_scanned"]
        total_pulls += pass_coverage["pulls_scanned"]
        total_duplicates += pass_coverage["duplicates_discarded"]

        if consecutive_stable >= stability_passes:
            merged_count = sum(
                record.get("_current_status") == "merged"
                for record in records
            )
            return records, {
                "repo": repo,
                "default_branch": default_branch,
                "pages_scanned": total_pages,
                "pulls_scanned": total_pulls,
                "merged_in_window": merged_count,
                "closed_unmerged_in_window": len(records) - merged_count,
                "duplicates_discarded": total_duplicates,
                "scan_passes": scan_pass,
                "stable_passes": consecutive_stable,
                "complete": True,
            }

    raise RuntimeError(
        f"{repo} result set did not stabilize for {stability_passes} "
        f"consecutive passes after {max_scan_passes} scans"
    )


def normalize_record(record: dict) -> dict:
    author = record.get("user") or {}
    base = record.get("base") or {}
    base_ref = str(base.get("ref", ""))
    default_branch = str(record.get("default_branch", ""))
    score, signals = score_candidate(
        str(record.get("title", "")),
        str(record.get("body", "") or ""),
    )
    return {
        "repo": record["repo"],
        "number": int(record["number"]),
        "title": record.get("title", ""),
        "url": record.get("html_url", ""),
        "author": author.get("login", "unknown") if isinstance(author, dict) else str(author),
        "merged_at": record.get("merged_at", ""),
        "event_at": record.get("_event_at")
                    or record.get("merged_at")
                    or record.get("closed_at", ""),
        "status": record.get("_current_status")
                  or ("merged" if record.get("merged_at") else "closed"),
        "merge_commit": record.get("merge_commit_sha", ""),
        "base_ref": base_ref,
        "default_branch": default_branch,
        "base_matches_current_default": bool(
            base_ref and base_ref == default_branch
        ),
        "score": score,
        "signals": signals,
        "body_excerpt": re.sub(r"\s+", " ", str(record.get("body", "") or "")).strip()[:500],
    }


def branch_note(record: dict) -> str:
    if record["base_matches_current_default"]:
        return f"base `{record['base_ref']}` (matches current default)"
    return (
        f"base `{record['base_ref']}` (does not match current default "
        f"`{record['default_branch']}`; trace landing chain)"
    )


def render_markdown(
    records: list[dict],
    low_score_records: list[dict],
    refresh_records: list[dict],
    since: str,
    until: str,
    coverage: list[dict],
    summary: dict,
) -> str:
    lines = [
        "# ROCm Kernel PR Candidate Inventory",
        "",
        f"Discovery window: `{since}` through `{until}` (inclusive).",
        "",
        "This file is discovery output only. Review PR bodies and changed files",
        "before authoring any source page.",
        "",
        "Repository coverage:",
        "",
    ]
    lines.extend(
        f"- `{item['repo']}`: default `{item['default_branch']}`, "
        f"{item['pages_scanned']} pages / {item['pulls_scanned']} closed PRs "
        f"scanned across {item['scan_passes']} passes, "
        f"{item['merged_in_window']} merged and "
        f"{item['closed_unmerged_in_window']} closed-unmerged in window, "
        f"{item['duplicates_discarded']} duplicates discarded, "
        "stable boundary reached"
        for item in coverage
    )
    lines.extend([
        "",
        f"Merged PRs in window: **{summary['merged_in_window']}**",
        f"Closed-unmerged PRs in window: **{summary['closed_unmerged_in_window']}**",
        f"Existing pages already current: **{summary['existing_excluded']}**",
        f"Existing pages needing status refresh: **{summary['existing_refresh']}**",
        f"New closed-unmerged PRs ignored: **{summary['closed_unmerged_ignored']}**",
        f"Below score {summary['min_score']}: **{summary['below_score']}**",
        f"Candidates: **{len(records)}**",
        "",
    ])
    for record in records:
        signals = ", ".join(record["signals"]) or "none"
        lines.append(
            f"- score={record['score']} "
            f"[`{record['repo']}#{record['number']}`]({record['url']}) "
            f"`{record['merged_at'][:10]}`, {branch_note(record)} - "
            f"{record['title']} "
            f"(signals: {signals})"
        )
    lines.extend([
        "",
        "## Existing Source Refresh Queue",
        "",
        "GitHub now reports these PRs as merged or closed, but at least one",
        "existing source page has a stale status. Refresh it in place.",
        "",
    ])
    for record in refresh_records:
        pages = ", ".join(
            f"`{item['path']}` ({item['status'] or 'unknown'})"
            for item in record["existing_pages"]
        )
        lines.append(
            f"- [`{record['repo']}#{record['number']}`]({record['url']}) "
            f"`{record['event_at'][:10]}` current status `{record['status']}`, "
            f"{branch_note(record)} - "
            f"{record['title']}; existing: {pages}"
        )
    lines.extend([
        "",
        "## Below-Score Review Queue",
        "",
        "These merged PRs remain part of the audit window. Review or explicitly",
        "reject them before advancing the global cutoff.",
        "",
    ])
    for record in low_score_records:
        lines.append(
            f"- score={record['score']} "
            f"[`{record['repo']}#{record['number']}`]({record['url']}) "
            f"`{record['merged_at'][:10]}`, {branch_note(record)} - "
            f"{record['title']}"
        )
    lines.append("")
    return "\n".join(lines)


def write_output(content: str, output: str | None) -> None:
    if not output:
        print(content, end="")
        return
    path = Path(output)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    try:
        display_path = path.relative_to(ROOT)
    except ValueError:
        display_path = path
    print(f"Wrote {display_path}")


def api_page_size(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def minimum_two(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover merged PR candidates without creating source pages"
    )
    parser.add_argument("--repo", action="append", default=[],
                        help="OWNER/REPO to scan; repeat for multiple repositories")
    parser.add_argument("--since", default=None,
                        help="Merged-at lower bound (default: data/refresh-cutoff.yaml)")
    parser.add_argument("--until", default=date.today().isoformat(),
                        help="Merged-at upper bound, inclusive (default: today)")
    parser.add_argument("--per-page", type=api_page_size, default=100)
    parser.add_argument("--max-pages", type=positive_int, default=100,
                        help="Fail if the lower boundary is not reached within this many pages")
    parser.add_argument("--stability-passes", type=minimum_two, default=2,
                        help="Required consecutive scans with the same PR set")
    parser.add_argument("--max-scan-passes", type=positive_int, default=4,
                        help="Fail if results do not stabilize within this many scans")
    parser.add_argument("--min-score", type=int, default=1)
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    requested_repos = args.repo or tracked_repositories()
    since = args.since or default_since()
    until = args.until
    if not requested_repos:
        parser.error("no repositories configured")
    if not since:
        parser.error("no cutoff configured; pass --since")
    try:
        since_date = date.fromisoformat(since)
        until_date = date.fromisoformat(until)
    except ValueError:
        parser.error("--since and --until must use YYYY-MM-DD")
    if since_date > until_date:
        parser.error("--since must not be after --until")
    if args.max_scan_passes < args.stability_passes:
        parser.error("--max-scan-passes must be at least --stability-passes")

    try:
        require_gh()
        repos = []
        repo_aliases = {}
        seen_repos = set()
        for requested_repo in requested_repos:
            canonical_repo, default_branch = resolve_repository(requested_repo)
            key = normalize_repo(canonical_repo)
            repo_aliases[normalize_repo(requested_repo)] = key
            repo_aliases[key] = key
            if key not in seen_repos:
                seen_repos.add(key)
                repos.append((canonical_repo, default_branch))

        raw_records = []
        coverage = []
        for repo, default_branch in repos:
            repo_records, repo_coverage = fetch_repo(
                repo,
                since,
                until,
                args.per_page,
                args.max_pages,
                args.stability_passes,
                args.max_scan_passes,
                default_branch,
            )
            raw_records.extend(repo_records)
            coverage.append(repo_coverage)
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    unique_raw_records = {}
    for record in raw_records:
        key = (normalize_repo(str(record["repo"])), int(record["number"]))
        unique_raw_records[key] = record
    raw_records = list(unique_raw_records.values())

    existing = existing_pr_pages(repo_aliases=repo_aliases)
    candidates = []
    low_score_records = []
    refresh_records = []
    existing_excluded = 0
    closed_unmerged_ignored = 0
    for raw in raw_records:
        record = normalize_record(raw)
        key = (normalize_repo(record["repo"]), record["number"])
        existing_pages = existing.get(key, [])
        if existing_pages:
            if not all(
                page["status"] == record["status"]
                for page in existing_pages
            ):
                record["existing_pages"] = existing_pages
                refresh_records.append(record)
                continue
            if not args.include_existing:
                existing_excluded += 1
                continue
        if record["status"] != "merged":
            closed_unmerged_ignored += 1
            continue
        if record["score"] < args.min_score:
            low_score_records.append(record)
            continue
        candidates.append(record)
    candidates.sort(key=lambda item: (item["score"], item["merged_at"]), reverse=True)
    low_score_records.sort(key=lambda item: item["merged_at"], reverse=True)
    refresh_records.sort(key=lambda item: item["event_at"], reverse=True)
    merged_in_window = sum(
        (
            record.get("_current_status")
            or ("merged" if record.get("merged_at") else "closed")
        ) == "merged"
        for record in raw_records
    )
    closed_unmerged_in_window = len(raw_records) - merged_in_window
    summary = {
        "merged_in_window": merged_in_window,
        "closed_unmerged_in_window": closed_unmerged_in_window,
        "existing_excluded": existing_excluded,
        "existing_refresh": len(refresh_records),
        "closed_unmerged_ignored": closed_unmerged_ignored,
        "below_score": len(low_score_records),
        "min_score": args.min_score,
        "candidates": len(candidates),
    }

    if args.format == "json":
        content = json.dumps(
            {
                "window": {"since": since, "until": until},
                "coverage": coverage,
                "summary": summary,
                "candidates": candidates,
                "existing_source_refresh_queue": refresh_records,
                "below_score_review_queue": low_score_records,
            },
            indent=2,
        ) + "\n"
    else:
        content = render_markdown(
            candidates,
            low_score_records,
            refresh_records,
            since,
            until,
            coverage,
            summary,
        )
    write_output(content, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

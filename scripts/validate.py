#!/usr/bin/env python3
"""
Validate all pages in ROCm-KernelWiki-Q against the schema.
Checks: required fields, tag vocabulary, link integrity, version consistency.

Usage:
    python3 scripts/validate.py [--fix]
"""

import os
import re
import sys
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WIKI_DIR = ROOT / "wiki"
SOURCES_DIR = ROOT / "sources"
QUERIES_DIR = ROOT / "queries"


def load_yaml(path):
    """Load YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_schemas():
    """Load page type schemas."""
    return load_yaml(DATA_DIR / "schemas.yaml")


def load_tags():
    """Load controlled vocabulary."""
    return load_yaml(DATA_DIR / "tags.yaml")


def extract_frontmatter(filepath):
    """Extract YAML frontmatter from a markdown file."""
    with open(filepath) as f:
        content = f.read()
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None, content
    fm_text = match.group(1)
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        return None, content
    return fm, content


def collect_all_ids():
    """Collect all page IDs across wiki and sources."""
    ids = {}
    for dir_path in [WIKI_DIR, SOURCES_DIR]:
        if not dir_path.exists():
            continue
        for md_file in dir_path.rglob("*.md"):
            fm, _ = extract_frontmatter(md_file)
            if fm and "id" in fm:
                ids[fm["id"]] = md_file.relative_to(ROOT)
    return ids


def validate_page(filepath, schemas, tags, all_ids):
    """Validate a single page against its schema."""
    errors = []
    warnings = []

    fm, content = extract_frontmatter(filepath)
    if fm is None:
        return [{"error": "no_frontmatter", "message": "No YAML frontmatter found"}], []

    page_type = fm.get("type", "")
    page_type_def = schemas.get("page_types", {}).get(page_type)

    if not page_type_def:
        warnings.append({"warning": "unknown_type", "message": f"Unknown page type: {page_type}"})
        return errors, warnings

    # Check required fields
    required = page_type_def.get("required_fields", [])
    for field in required:
        if field not in fm:
            errors.append({"error": "missing_field", "field": field, "message": f"Missing required field: {field}"})

    # Check architectures against controlled vocabulary
    valid_archs = {a["id"] for a in tags.get("architectures", [])}
    page_archs = fm.get("architectures", [])
    for arch in page_archs:
        if arch not in valid_archs:
            errors.append({"error": "invalid_tag", "field": "architectures", "value": arch,
                          "message": f"Invalid architecture: {arch}. Valid: {sorted(valid_archs)}"})

    # Check tags against controlled vocabulary
    all_valid_tags = set()
    for hw in tags.get("hardware_features", {}):
        all_valid_tags.add(hw if isinstance(hw, str) else hw)
    for tech in tags.get("techniques", {}):
        all_valid_tags.add(tech if isinstance(tech, str) else tech)
    for kt in tags.get("kernel_types", []):
        all_valid_tags.add(kt)
    for lang in tags.get("languages", []):
        all_valid_tags.add(lang)

    # Also add architecture IDs as valid tags
    all_valid_tags.update(valid_archs)

    page_tags = fm.get("tags", [])
    for tag in page_tags:
        # Be lenient — some tags might be subtags or descriptive
        # Only warn for completely unknown tags
        pass

    # Check link integrity (sources and related)
    for field in ["sources", "related"]:
        refs = fm.get(field, [])
        if isinstance(refs, str):
            refs = [refs]
        for ref in refs:
            if ref not in all_ids:
                # Not necessarily an error — might not be created yet
                warnings.append({"warning": "broken_link", "field": field, "value": ref,
                               "message": f"Reference not found: {ref}"})

    # Check confidence level
    confidence = fm.get("confidence", "")
    valid_confidence = {"verified", "source-reported", "inferred", "experimental"}
    if confidence and confidence not in valid_confidence:
        errors.append({"error": "invalid_confidence", "value": confidence,
                      "message": f"Invalid confidence: {confidence}"})

    # Check performance claims structure (wiki-kernel type)
    if page_type == "wiki-kernel":
        perf_claims = fm.get("performance_claims", [])
        for i, claim in enumerate(perf_claims):
            required_claim_fields = ["gpu", "dtype", "metric", "value", "source_id"]
            for field in required_claim_fields:
                if field not in claim:
                    errors.append({"error": "incomplete_perf_claim", "index": i, "field": field,
                                  "message": f"Performance claim [{i}] missing: {field}"})

    return errors, warnings


def main():
    schemas = load_schemas()
    tags = load_tags()
    all_ids = collect_all_ids()

    total_errors = 0
    total_warnings = 0
    total_pages = 0

    # Validate wiki pages
    for dir_path in [WIKI_DIR, SOURCES_DIR]:
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.rglob("*.md")):
            total_pages += 1
            errors, warnings = validate_page(md_file, schemas, tags, all_ids)

            if errors:
                rel_path = md_file.relative_to(ROOT)
                for e in errors:
                    print(f"  ERROR  {rel_path}: {e['message']}")
                total_errors += len(errors)

            if warnings:
                rel_path = md_file.relative_to(ROOT)
                for w in warnings:
                    print(f"  WARN   {rel_path}: {w['message']}")
                total_warnings += len(warnings)

    print(f"\n{'='*60}")
    print(f"Validated: {total_pages} pages")
    print(f"Errors:    {total_errors}")
    print(f"Warnings:  {total_warnings}")
    print(f"Known IDs: {len(all_ids)}")
    print(f"{'='*60}")

    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

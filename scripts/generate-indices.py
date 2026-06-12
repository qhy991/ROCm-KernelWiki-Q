#!/usr/bin/env python3
"""
Generate cross-reference indices (queries/*.md) from wiki page frontmatter.

Usage:
    python3 scripts/generate-indices.py
"""

import os
import re
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = ROOT / "wiki"
SOURCES_DIR = ROOT / "sources"
QUERIES_DIR = ROOT / "queries"


def extract_frontmatter(filepath):
    """Extract YAML frontmatter from a markdown file."""
    with open(filepath) as f:
        content = f.read()
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None


def collect_pages():
    """Collect all pages with their frontmatter."""
    pages = []
    for dir_path in [WIKI_DIR, SOURCES_DIR]:
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.rglob("*.md")):
            fm = extract_frontmatter(md_file)
            if fm:
                rel = md_file.relative_to(ROOT)
                pages.append({"path": str(rel), **fm})
    return pages


def generate_by_hardware(pages):
    """Generate index by hardware feature."""
    groups = defaultdict(list)
    for p in pages:
        for tag in p.get("tags", []):
            if tag in ("mfma", "lds", "dpp", "gws", "wavefront", "scaled-mfma",
                       "lds-transpose", "dual-cma", "matrix-core", "compute-unit"):
                groups[tag].append(p)
        for hw in p.get("hardware_features", []):
            groups[hw].append(p)

    lines = ["# Index: By Hardware Feature\n"]
    for feature in sorted(groups.keys()):
        entries = groups[feature]
        # Deduplicate by id
        seen = set()
        unique = []
        for e in entries:
            eid = e.get("id", e["path"])
            if eid not in seen:
                seen.add(eid)
                unique.append(e)

        lines.append(f"\n## {feature} ({len(unique)} pages)\n")
        for e in sorted(unique, key=lambda x: x.get("id", x["path"])):
            eid = e.get("id", "?")
            title = e.get("title", eid)
            ptype = e.get("type", "?")
            archs = ", ".join(e.get("architectures", []))
            lines.append(f"- [{title}]({e['path']}) `[{ptype}]` arch:{archs}")

    return "\n".join(lines)


def generate_by_technique(pages):
    """Generate index by optimization technique."""
    techniques = defaultdict(list)
    for p in pages:
        for tech in p.get("techniques", p.get("tags", [])):
            if tech.startswith(("ck-", "bank-", "persistent-", "mfma-", "double-",
                               "async-", "register-", "wave-", "swizzle", "vectorized",
                               "occupancy-")):
                techniques[tech].append(p)

    lines = ["# Index: By Technique\n"]
    for tech in sorted(techniques.keys()):
        entries = techniques[tech]
        seen = set()
        unique = []
        for e in entries:
            eid = e.get("id", e["path"])
            if eid not in seen:
                seen.add(eid)
                unique.append(e)

        lines.append(f"\n## {tech} ({len(unique)} pages)\n")
        for e in sorted(unique, key=lambda x: x.get("id", x["path"])):
            eid = e.get("id", "?")
            title = e.get("title", eid)
            lines.append(f"- [{title}]({e['path']})")

    return "\n".join(lines)


def generate_by_kernel_type(pages):
    """Generate index by kernel type."""
    kernel_types = defaultdict(list)
    for p in pages:
        for kt in p.get("kernel_types", []):
            kernel_types[kt].append(p)

    lines = ["# Index: By Kernel Type\n"]
    for kt in sorted(kernel_types.keys()):
        entries = kernel_types[kt]
        seen = set()
        unique = []
        for e in entries:
            eid = e.get("id", e["path"])
            if eid not in seen:
                seen.add(eid)
                unique.append(e)

        lines.append(f"\n## {kt} ({len(unique)} pages)\n")
        for e in sorted(unique, key=lambda x: x.get("id", x["path"])):
            eid = e.get("id", "?")
            title = e.get("title", eid)
            conf = e.get("confidence", "?")
            archs = ", ".join(e.get("architectures", []))
            lines.append(f"- [{title}]({e['path']}) conf:{conf} arch:{archs}")

    return "\n".join(lines)


def generate_by_language(pages):
    """Generate index by programming language."""
    langs = defaultdict(list)
    for p in pages:
        for lang in p.get("languages", []):
            langs[lang].append(p)

    lines = ["# Index: By Language\n"]
    for lang in sorted(langs.keys()):
        entries = langs[lang]
        seen = set()
        unique = []
        for e in entries:
            eid = e.get("id", e["path"])
            if eid not in seen:
                seen.add(eid)
                unique.append(e)

        lines.append(f"\n## {lang} ({len(unique)} pages)\n")
        for e in sorted(unique, key=lambda x: x.get("id", x["path"])):
            eid = e.get("id", "?")
            title = e.get("title", eid)
            lines.append(f"- [{title}]({e['path']})")

    return "\n".join(lines)


def generate_by_problem(pages):
    """Generate index by problem/symptom."""
    lines = ["# Index: By Problem\n"]
    lines.append("\nSymptom → Pattern → Technique → Solution\n")

    patterns = [p for p in pages if p.get("type") == "wiki-pattern"]
    for p in sorted(patterns, key=lambda x: x.get("id", "")):
        title = p.get("title", "?")
        eid = p.get("id", "?")
        symptoms = p.get("symptoms", p.get("tags", []))
        lines.append(f"\n### {title}\n")
        lines.append(f"- ID: `{eid}`")
        lines.append(f"- Path: `{p['path']}`")
        if symptoms:
            lines.append(f"- Tags: {', '.join(symptoms)}")
        related = p.get("related", [])
        if related:
            lines.append(f"- Related: {', '.join(f'`{r}`' for r in related)}")

    return "\n".join(lines)


def main():
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    pages = collect_pages()
    print(f"Collected {len(pages)} pages")

    indices = {
        "by-hardware-feature.md": generate_by_hardware(pages),
        "by-technique.md": generate_by_technique(pages),
        "by-kernel-type.md": generate_by_kernel_type(pages),
        "by-language.md": generate_by_language(pages),
        "by-problem.md": generate_by_problem(pages),
    }

    for name, content in indices.items():
        path = QUERIES_DIR / name
        with open(path, "w") as f:
            f.write(content)
        print(f"  Generated: {name}")

    print(f"\nDone: {len(indices)} indices generated in queries/")


if __name__ == "__main__":
    main()

"""
Smoke + regression tests for the ROCm-KernelWiki-Q tooling and data.

Run: python3 -m pytest tests/ -q
These guard the failure modes that previously shipped silently: a no-op validator,
docs referencing missing scripts, broken index links, and title-only PR classification.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import validate  # noqa: E402
import get_page  # noqa: E402
import enrich_pr_pages as ep  # noqa: E402
import infer_rocm_arch as ia  # noqa: E402
import reclassify_pr_pages as rc  # noqa: E402


# --- classification -------------------------------------------------------

def test_classify_recovers_kernel_type_and_language_from_paths():
    body = (
        "## Description\n\n> Add interwave scheduler.\n\n"
        "## Changed Files\n\n"
        "- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp` (+23/-0)\n"
        "- `include/ck_tile/ops/gemm_quant/pipeline/foo.hpp` (+1/-1)\n"
    )
    paths = rc.changed_paths(body)
    assert paths, "changed_paths should parse the Changed Files section"
    cls = rc.classify("feat: Add Interwave scheduler for memory pipeline", body, paths,
                      repo="ROCm/composable_kernel")
    assert "gemm" in cls["kernel_types"]
    assert "hip-cpp" in cls["languages"]
    assert "ck-dsl" in cls["languages"]
    assert "mfma-scheduling" in cls["techniques"]


def test_classify_flash_attention_repo_defaults_to_attention():
    cls = rc.classify("Enable MQA/GQA in backward", "## Description\n\n> x\n", [],
                      repo="ROCm/flash-attention")
    assert "attention" in cls["kernel_types"]


def test_classify_treats_matmul_as_gemm():
    cls = rc.classify("Fix MXFP4 dequant matmul on MI300X", "## Description\n\n> x\n", [],
                      repo="modular/modular")
    assert "gemm" in cls["kernel_types"]


def test_classify_treats_prefill_decode_qkv_as_attention():
    cls = rc.classify(
        "shared-engine prefill/decode optimizations and fused qkv projection",
        "## Description\n\n> x\n",
        [],
        repo="ROCm/hipBLASLt",
    )
    assert "attention" in cls["kernel_types"]


def test_classify_does_not_fabricate_cdna_for_rdna():
    cls = rc.classify("Support gfx1200 / RDNA4 path", "", [])
    assert cls["archs_cdna"] == []
    assert cls["is_rdna"] is True


def test_infer_architecture_recognizes_rdna4_from_gfx12_signal():
    result = ia.infer_architectures(
        "gfx1201 gemm tuning for RDNA4",
        "## Changed Files\n\n- `configs/gemm/gfx1201-GEMM-A8W8.json` (+1/-0)\n",
    )

    assert result == ["rdna4"]


def test_infer_architecture_replaces_default_cdna_triple_for_rdna4(tmp_path):
    page = tmp_path / "PR-1.md"
    page.write_text(
        "---\n"
        "id: pr-test-1\n"
        "type: source-pr\n"
        "repo: example/repo\n"
        "pr: 1\n"
        "title: gfx1201 RDNA4 GEMM tuning\n"
        "author: test\n"
        "date: '2026-01-01'\n"
        "url: https://github.com/example/repo/pull/1\n"
        "source_category: upstream-code\n"
        "architectures: [cdna2, cdna3, cdna4]\n"
        "tags: [rdna, rocm]\n"
        "captured_at: '2026-06-12'\n"
        "status: merged\n"
        "inclusion_reason: test\n"
        "---\n\n"
        "## Description\n\n> Tuned gfx1201 kernels.\n",
        encoding="utf-8",
    )

    stats = ia.Stats()
    assert ia.update_page_architectures(page, dry_run=False, stats=stats) is True

    fm, _ = validate.extract_frontmatter(page)
    assert fm["architectures"] == ["rdna4"]
    assert stats.narrowed_default_arch == 1


def test_pr_status_and_date_prefers_merged_at_for_merged_prs():
    status, date = ep.pr_status_and_date({
        "state": "MERGED",
        "mergedAt": "2026-01-02T03:04:05Z",
        "closedAt": "2026-01-03T03:04:05Z",
    }, fallback_date="unknown")

    assert status == "merged"
    assert date == "2026-01-02"


def test_pr_status_and_date_uses_closed_at_for_unmerged_closed_prs():
    status, date = ep.pr_status_and_date({
        "state": "CLOSED",
        "mergedAt": None,
        "closedAt": "2023-05-12T12:51:24Z",
    }, fallback_date="unknown")

    assert status == "closed"
    assert date == "2023-05-12"


def test_pr_status_and_date_falls_back_to_created_at_when_close_dates_missing():
    status, date = ep.pr_status_and_date({
        "state": "OPEN",
        "mergedAt": None,
        "closedAt": None,
        "createdAt": "2026-06-10T11:12:13Z",
    }, fallback_date="unknown")

    assert status == "open"
    assert date == "2026-06-10"


# --- validation -----------------------------------------------------------

def test_repository_has_zero_validation_errors():
    schemas = validate.load_yaml(validate.DATA_DIR / "schemas.yaml")
    vocab = validate.build_vocab(validate.load_yaml(validate.DATA_DIR / "tags.yaml"))
    all_ids = validate.collect_all_ids()
    errors = []
    for d in (validate.WIKI_DIR, validate.SOURCES_DIR):
        for md in d.rglob("*.md"):
            errs, _ = validate.validate_page(md, schemas, vocab, all_ids)
            errors += [f"{md}: {e}" for e in errs]
    assert errors == [], "validation errors:\n" + "\n".join(errors)


def test_curated_non_pr_pages_use_canonical_vocab_terms():
    schemas = validate.load_yaml(validate.DATA_DIR / "schemas.yaml")
    vocab = validate.build_vocab(validate.load_yaml(validate.DATA_DIR / "tags.yaml"))
    all_ids = validate.collect_all_ids()
    bad_warnings = []
    for d in (validate.WIKI_DIR, validate.SOURCES_DIR):
        for md in d.rglob("*.md"):
            fm, _ = validate.extract_frontmatter(md)
            if not fm or fm.get("type") == "source-pr":
                continue
            _, warnings = validate.validate_page(md, schemas, vocab, all_ids)
            bad_warnings += [
                f"{md.relative_to(ROOT)}: {warning}"
                for warning in warnings
                if "alias; use" in warning or "off-vocabulary value" in warning
            ]
    assert bad_warnings == [], "canonical vocabulary warnings:\n" + "\n".join(bad_warnings)


def test_validator_actually_flags_bad_data(tmp_path):
    schemas = validate.load_yaml(validate.DATA_DIR / "schemas.yaml")
    vocab = validate.build_vocab(validate.load_yaml(validate.DATA_DIR / "tags.yaml"))
    bad = tmp_path / "bad.md"
    bad.write_text("---\nid: wrongprefix-x\ntype: wiki-hardware\n"
                   "architectures: [cdna9]\nconfidence: bogus\ntags: []\nsources: []\n---\n# x\n")
    errs, _ = validate.validate_page(bad, schemas, vocab, {})
    joined = " ".join(errs)
    assert "architecture" in joined        # cdna9 invalid
    assert "confidence" in joined          # bogus invalid
    assert "prefix" in joined              # id prefix mismatch


def test_validator_accepts_rdna_architectures(tmp_path):
    schemas = validate.load_yaml(validate.DATA_DIR / "schemas.yaml")
    vocab = validate.build_vocab(validate.load_yaml(validate.DATA_DIR / "tags.yaml"))
    page = tmp_path / "rdna.md"
    page.write_text(
        "---\n"
        "id: pr-test-rdna\n"
        "type: source-pr\n"
        "repo: example/repo\n"
        "pr: 1\n"
        "title: RDNA4 kernel tuning\n"
        "author: test\n"
        "date: '2026-01-01'\n"
        "url: https://github.com/example/repo/pull/1\n"
        "source_category: upstream-code\n"
        "architectures: [rdna4]\n"
        "tags: [rdna, rocm]\n"
        "captured_at: '2026-06-12'\n"
        "status: merged\n"
        "inclusion_reason: test\n"
        "---\n# x\n",
        encoding="utf-8",
    )

    errs, warnings = validate.validate_page(page, schemas, vocab, {})

    assert not [err for err in errs if "invalid architecture" in err]
    assert "source-pr still on default" not in " ".join(warnings)


def test_no_duplicate_ids():
    ids = validate.collect_all_ids()
    dups = {k: v for k, v in ids.items() if len(v) > 1}
    assert dups == {}, f"duplicate ids: {dups}"


# --- docs & links ---------------------------------------------------------

def test_scripts_referenced_in_docs_exist():
    missing = []
    for doc in ("README.md", "CLAUDE.md"):
        text = (ROOT / doc).read_text()
        for name in re.findall(r"scripts/([A-Za-z0-9_\-]+\.py)", text):
            if not (ROOT / "scripts" / name).exists():
                missing.append(f"{doc} -> scripts/{name}")
    assert missing == [], f"docs reference missing scripts: {missing}"


def test_generated_index_links_resolve():
    qdir = ROOT / "queries"
    broken = []
    for md in qdir.glob("*.md"):
        for rel in re.findall(r"\]\((\.\./[^)]+\.md)\)", md.read_text()):
            if not (qdir / rel).resolve().exists():
                broken.append(f"{md.name} -> {rel}")
    assert broken == [], f"broken index links: {broken}"


def test_manifest_is_valid_and_covers_all_pages():
    records = json.loads((ROOT / "queries" / "pages.json").read_text())
    ids = validate.collect_all_ids()
    assert len(records) == len(ids)
    assert all(r["id"] for r in records)


def test_curated_kernel_pages_have_source_density():
    kernel_pages = []
    sparse = []
    for md in sorted((validate.WIKI_DIR / "kernels").glob("*.md")):
        fm, _ = validate.extract_frontmatter(md)
        if not fm:
            sparse.append(f"{md.relative_to(ROOT)}: missing frontmatter")
            continue
        kernel_pages.append(fm)
        if len(fm.get("sources", []) or []) < 3:
            sparse.append(f"{md.relative_to(ROOT)}: fewer than 3 sources")

    assert len(kernel_pages) >= 12
    assert sparse == []


# --- get_page -------------------------------------------------------------

def test_get_page_resolves_known_id():
    index = get_page.build_index()
    assert "hw-mfma-matrix-core" in index
    path, fm, body = index["hw-mfma-matrix-core"]
    assert fm["type"] == "wiki-hardware"
    assert body.strip()


def test_query_supports_common_operator_filter():
    result = subprocess.run(
        [sys.executable, "scripts/query.py", "--type", "kernel", "--operator", "gemm", "--limit", "3"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "kernel-gemm-mfma-rocm" in result.stdout


def test_get_page_supports_common_frontmatter_only_alias():
    result = subprocess.run(
        [sys.executable, "scripts/get_page.py", "kernel-gemm-mfma-rocm", "--frontmatter-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "id: kernel-gemm-mfma-rocm" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

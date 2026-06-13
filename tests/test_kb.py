"""
Smoke + regression tests for the ROCm-KernelWiki-Q tooling and data.

Run: python3 -m pytest tests/ -q
These guard the failure modes that previously shipped silently: a no-op validator,
docs referencing missing scripts, broken index links, and title-only PR classification.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import validate  # noqa: E402
import get_page  # noqa: E402
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


def test_classify_does_not_fabricate_cdna_for_rdna():
    cls = rc.classify("Support gfx1200 / RDNA4 path", "", [])
    assert cls["archs_cdna"] == []
    assert cls["is_rdna"] is True


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


# --- get_page -------------------------------------------------------------

def test_get_page_resolves_known_id():
    index = get_page.build_index()
    assert "hw-mfma-matrix-core" in index
    path, fm, body = index["hw-mfma-matrix-core"]
    assert fm["type"] == "wiki-hardware"
    assert body.strip()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

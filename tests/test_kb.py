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
import discover_pr_candidates as dpc  # noqa: E402


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


def test_candidate_discovery_prioritizes_kernel_work_over_ci():
    kernel_score, signals = dpc.score_candidate(
        "perf(rocke): gfx942 flash-attn LDS prefetch optimization",
        "Interleave MFMA and async loads for MI300.",
    )
    ci_score, _ = dpc.score_candidate(
        "ci: update ROCm workflow",
        "Update test runner configuration.",
    )

    assert kernel_score > ci_score
    assert "gfx942" in signals
    assert "mfma" in signals


def test_candidate_discovery_penalizes_repeated_ci_prefixes_and_word_boundaries():
    ci_score, ci_signals = dpc.score_candidate(
        "[ROCm][CI] kernel performance tests",
        "Update the test runner.",
    )
    false_positive_score, false_positive_signals = dpc.score_candidate(
        "perfect docs wording",
    )
    docs_score, docs_signals = dpc.score_candidate(
        "docs(rocke): refresh architecture foundations",
        "Document gfx942 kernels, MFMA scheduling, and LDS prefetch.",
    )
    cicd_score, cicd_signals = dpc.score_candidate(
        "CI/CD: update ROCm kernel tests",
    )
    deps_score, deps_signals = dpc.score_candidate(
        "build(deps-dev): bump triton kernel package",
    )
    mi300x_score, mi300x_signals = dpc.score_candidate("MI300X tuning")

    assert ci_score < 1
    assert "penalty:ci" in ci_signals
    assert false_positive_score == 0
    assert "perf" not in false_positive_signals
    assert docs_score < 1
    assert "penalty:docs" in docs_signals
    assert cicd_score < 1
    assert "penalty:ci/cd" in cicd_signals
    assert deps_score < 1
    assert "penalty:build(deps*)" in deps_signals
    assert mi300x_score > 0
    assert "mi300x" in mi300x_signals


def test_candidate_discovery_deduplicates_by_canonical_repo(tmp_path):
    page = tmp_path / "sources" / "prs" / "vllm" / "PR-42.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\n"
        "id: pr-vllm-42\n"
        "type: source-pr\n"
        "repo: VLLM-PROJECT/vLLM\n"
        "pr: 42\n"
        "status: open\n"
        "captured_at: '2026-07-01'\n"
        "---\n",
        encoding="utf-8",
    )

    assert ("vllm-project/vllm", 42) in dpc.existing_pr_keys(
        tmp_path / "sources" / "prs"
    )
    records = dpc.existing_pr_pages(tmp_path / "sources" / "prs")
    assert records[("vllm-project/vllm", 42)][0]["status"] == "open"
    canonical_records = dpc.existing_pr_pages(
        tmp_path / "sources" / "prs",
        {"vllm-project/vllm": "canonical/vllm"},
    )
    assert ("canonical/vllm", 42) in canonical_records


def test_candidate_discovery_pages_to_boundary_and_records_base_branch(monkeypatch):
    def fake_api(endpoint, fields=None):
        if endpoint == "repos/ROCm/example":
            return {
                "default_branch": "develop",
                "full_name": "ROCm/canonical-example",
            }
        page = int(fields["page"])
        if page == 1:
            return [
                {
                    "number": 2,
                    "title": "perf: gfx950 MFMA kernel",
                    "body": "",
                    "html_url": "https://github.com/ROCm/example/pull/2",
                    "user": {"login": "dev"},
                    "base": {"ref": "develop"},
                    "merged_at": "2026-07-20T00:00:00Z",
                    "updated_at": "2026-07-20T00:00:00Z",
                    "merge_commit_sha": "abc",
                },
                {
                    "number": 1,
                    "title": "perf: gfx950 LDS kernel",
                    "body": "",
                    "html_url": "https://github.com/ROCm/example/pull/1",
                    "user": {"login": "dev"},
                    "base": {"ref": "feature"},
                    "merged_at": "2026-07-19T00:00:00Z",
                    "updated_at": "2026-07-19T00:00:00Z",
                    "merge_commit_sha": "def",
                },
            ]
        return [
            {
                "number": 3,
                "title": "close abandoned kernel experiment",
                "body": "",
                "html_url": "https://github.com/ROCm/example/pull/3",
                "user": {"login": "dev"},
                "base": {"ref": "develop"},
                "merged_at": None,
                "closed_at": "2026-07-18T00:00:00Z",
                "updated_at": "2026-07-18T00:00:00Z",
                "merge_commit_sha": None,
            },
            {
                "number": 0,
                "merged_at": "2026-06-10T00:00:00Z",
                "closed_at": "2026-06-10T00:00:00Z",
                "updated_at": "2026-06-10T00:00:00Z",
            }
        ]

    monkeypatch.setattr(dpc, "gh_api", fake_api)
    records, coverage = dpc.fetch_repo(
        "ROCm/example",
        "2026-06-12",
        "2026-07-29",
        per_page=2,
        max_pages=5,
    )
    normalized = [dpc.normalize_record(record) for record in records]

    assert coverage["complete"] is True
    assert coverage["repo"] == "ROCm/canonical-example"
    assert coverage["pages_scanned"] == 4
    assert coverage["scan_passes"] == 2
    assert len(normalized) == 3
    assert coverage["merged_in_window"] == 2
    assert coverage["closed_unmerged_in_window"] == 1
    assert normalized[0]["base_matches_current_default"] is True
    assert normalized[1]["base_ref"] == "feature"
    assert normalized[1]["base_matches_current_default"] is False
    assert normalized[2]["status"] == "closed"


def test_candidate_discovery_fails_instead_of_truncating(monkeypatch):
    def fake_api(endpoint, fields=None):
        if endpoint == "repos/ROCm/example":
            return {"default_branch": "develop"}
        return [
            {
                "number": 1,
                "merged_at": "2026-07-20T00:00:00Z",
                "updated_at": "2026-07-20T00:00:00Z",
            }
        ]

    monkeypatch.setattr(dpc, "gh_api", fake_api)
    with pytest.raises(RuntimeError, match="increase --max-pages"):
        dpc.fetch_repo(
            "ROCm/example",
            "2026-06-12",
            "2026-07-29",
            per_page=1,
            max_pages=1,
        )


def test_candidate_discovery_keeps_low_score_prs_in_review_queue():
    record = {
        "repo": "ROCm/example",
        "number": 7,
        "title": "refactor helper",
        "url": "https://github.com/ROCm/example/pull/7",
        "merged_at": "2026-07-20T00:00:00Z",
        "base_ref": "feature",
        "default_branch": "develop",
        "base_matches_current_default": False,
        "score": 0,
        "signals": [],
    }
    markdown = dpc.render_markdown(
        [],
        [record],
        [],
        "2026-07-01",
        "2026-07-29",
        [{
            "repo": "ROCm/example",
            "default_branch": "develop",
            "pages_scanned": 1,
            "pulls_scanned": 10,
            "merged_in_window": 1,
            "closed_unmerged_in_window": 0,
            "duplicates_discarded": 0,
            "scan_passes": 2,
        }],
        {
            "merged_in_window": 1,
            "closed_unmerged_in_window": 0,
            "existing_excluded": 0,
            "existing_refresh": 0,
            "closed_unmerged_ignored": 0,
            "below_score": 1,
            "min_score": 1,
        },
    )

    assert "Below-Score Review Queue" in markdown
    assert "ROCm/example#7" in markdown
    assert "does not match current default" in markdown


def test_candidate_discovery_refreshes_existing_open_page_when_pr_merged(
    monkeypatch,
    tmp_path,
):
    raw_record = {
        "repo": "ROCm/example",
        "default_branch": "develop",
        "number": 7,
        "title": "perf: gfx942 kernel",
        "body": "",
        "html_url": "https://github.com/ROCm/example/pull/7",
        "user": {"login": "dev"},
        "base": {"ref": "develop"},
        "merged_at": "2026-07-20T00:00:00Z",
        "updated_at": "2026-07-20T00:00:00Z",
        "merge_commit_sha": "a" * 40,
    }
    raw_closed_record = {
        "repo": "ROCm/example",
        "default_branch": "develop",
        "number": 8,
        "title": "close abandoned kernel experiment",
        "body": "",
        "html_url": "https://github.com/ROCm/example/pull/8",
        "user": {"login": "dev"},
        "base": {"ref": "develop"},
        "merged_at": None,
        "closed_at": "2026-07-21T00:00:00Z",
        "updated_at": "2026-07-21T00:00:00Z",
        "merge_commit_sha": None,
        "_current_status": "closed",
        "_event_at": "2026-07-21T00:00:00Z",
    }
    monkeypatch.setattr(dpc, "require_gh", lambda: None)
    monkeypatch.setattr(
        dpc,
        "resolve_repository",
        lambda repo: ("ROCm/example", "develop"),
    )
    fetch_calls = []

    def fake_fetch(*args):
        fetch_calls.append(args[0])
        return (
            [raw_record, raw_closed_record],
            {
                "repo": "ROCm/example",
                "default_branch": "develop",
                "pages_scanned": 2,
                "pulls_scanned": 4,
                "merged_in_window": 1,
                "closed_unmerged_in_window": 1,
                "duplicates_discarded": 0,
                "scan_passes": 2,
                "stable_passes": 2,
                "complete": True,
            },
        )

    monkeypatch.setattr(
        dpc,
        "fetch_repo",
        fake_fetch,
    )
    monkeypatch.setattr(
        dpc,
        "existing_pr_pages",
        lambda **kwargs: {
            ("rocm/example", 7): [{
                "path": "sources/prs/example/PR-7.md",
                "status": "open",
                "captured_at": "2026-07-01",
            }],
            ("rocm/example", 8): [{
                "path": "sources/prs/example/PR-8.md",
                "status": "open",
                "captured_at": "2026-07-01",
            }],
        },
    )
    output = tmp_path / "discovery.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "discover_pr_candidates.py",
            "--repo", "ROCm/example",
            "--repo", "legacy/example",
            "--since", "2026-07-01",
            "--until", "2026-07-29",
            "--format", "json",
            "--output", str(output),
        ],
    )

    assert dpc.main() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["summary"]["existing_refresh"] == 2
    assert fetch_calls == ["ROCm/example"]
    assert result["candidates"] == []
    refresh_statuses = {
        item["number"]: item["status"]
        for item in result["existing_source_refresh_queue"]
    }
    assert refresh_statuses == {7: "merged", 8: "closed"}


# --- validation -----------------------------------------------------------

def test_repository_has_zero_validation_errors_and_respects_warning_budget():
    schemas = validate.load_yaml(validate.DATA_DIR / "schemas.yaml")
    vocab = validate.build_vocab(validate.load_yaml(validate.DATA_DIR / "tags.yaml"))
    baseline = validate.load_yaml(validate.DATA_DIR / "quality-baseline.yaml")
    all_ids = validate.collect_all_ids()
    errors = []
    warnings = []
    for d in (validate.WIKI_DIR, validate.SOURCES_DIR):
        for md in d.rglob("*.md"):
            errs, warns = validate.validate_page(md, schemas, vocab, all_ids)
            errors += [f"{md}: {e}" for e in errs]
            warnings += [f"{md}: {warning}" for warning in warns]
    assert errors == [], "validation errors:\n" + "\n".join(errors)
    max_warnings = baseline["validation"]["max_warnings"]
    assert len(warnings) <= max_warnings, (
        f"warning budget exceeded: {len(warnings)} > {max_warnings}"
    )


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


def test_no_new_curated_kernel_pages_have_sparse_sources():
    legacy_minimum_sources = {
        "activation-kernels.md": 0,
        "all-reduce-rocm.md": 0,
        "batched-gemm-rocm.md": 0,
        "embedding-lookup.md": 0,
        "flash-decoding-rocm.md": 1,
        "fused-attention-bias.md": 0,
        "fused-moe-gemm-rocm.md": 0,
        "gemm-rocm.md": 2,
        "histogram-rocm.md": 0,
        "kv-cache-rocm.md": 0,
        "layernorm-rocm.md": 0,
        "mla-attention-rocm.md": 1,
        "prefix-sum-scan.md": 0,
        "quantized-gemm-w4a16.md": 0,
        "quantized-gemm-w8a8.md": 0,
        "reduction-rocm.md": 1,
        "rms-norm-quant-fused.md": 0,
        "rotary-embedding-rocm.md": 0,
        "sparse-gemm-rocm.md": 0,
        "speculative-decoding-rocm.md": 0,
        "topk-softmax-rocm.md": 0,
        "transpose-rocm.md": 0,
    }
    all_ids = validate.collect_all_ids()
    kernel_pages = []
    unexpected = []
    for md in sorted((validate.WIKI_DIR / "kernels").glob("*.md")):
        fm, _ = validate.extract_frontmatter(md)
        if not fm:
            unexpected.append(f"{md.relative_to(ROOT)}: missing frontmatter")
            continue
        kernel_pages.append(fm)
        sources = set(fm.get("sources", []) or [])
        minimum = legacy_minimum_sources.get(md.name, 3)
        if len(sources) < minimum:
            unexpected.append(
                f"{md.relative_to(ROOT)}: {len(sources)} unique sources, "
                f"expected at least {minimum}"
            )
        resolved = sum(source in all_ids for source in sources)
        if minimum > 0 and resolved < minimum:
            unexpected.append(
                f"{md.relative_to(ROOT)}: {resolved} resolved sources, expected at least {minimum}"
            )

    assert len(kernel_pages) >= 12
    assert unexpected == [], "new source-density debt:\n" + "\n".join(unexpected)


# --- get_page -------------------------------------------------------------

def test_get_page_resolves_known_id():
    index = get_page.build_index()
    assert "hw-mfma-matrix-core" in index
    path, fm, body = index["hw-mfma-matrix-core"]
    assert fm["type"] == "wiki-hardware"
    assert body.strip()


def test_query_supports_common_operator_filter():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/query.py",
            "--type", "kernel",
            "--operator", "gemm",
            "--limit", "3",
            "--paths-only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    paths = [ROOT / line for line in result.stdout.splitlines() if line]
    assert len(paths) == 3
    for path in paths:
        fm, _ = validate.extract_frontmatter(path)
        values = set(fm.get("kernel_types", [])) | set(fm.get("tags", []))
        assert "gemm" in values


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

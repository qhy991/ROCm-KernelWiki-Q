# PR Refresh Workflow

This workflow keeps ROCm-KernelWiki-Q current without turning GitHub search
results into low-quality source stubs. Discovery, source capture, synthesis,
and GPU reproduction are separate gates.

## 1. Start From a Clean Branch

```bash
git switch main
git pull --ff-only
git switch -c refresh/prs-YYYY-MM-DD
python3 scripts/validate.py
python3 -m pytest tests/ -q
```

Record the existing validation warning count. A refresh must produce zero
errors, stay within the tracked net warning budget, and leave every touched
Wiki/source page warning-free. Existing warnings are cleanup debt, not
permission to add more.

## 2. Discover Candidates

Authenticate GitHub CLI before discovery:

```bash
gh auth status || gh auth login
python3 scripts/discover_pr_candidates.py \
  --repo ROCm/rocm-libraries \
  --repo vllm-project/vllm \
  --until YYYY-MM-DD \
  --output candidates/PR-CANDIDATES-YYYY-MM-DD.md
```

The script defaults to the PR cutoff in `data/refresh-cutoff.yaml`, walks the
closed-PR REST endpoint until it proves the lower date boundary was reached,
resolves renamed repositories through GitHub's canonical `full_name`,
deduplicates by canonical repository and PR number, and repeats the scan until
the result set is stable. It fails instead of silently accepting a page limit
or unstable scan. It excludes already-merged sources, queues stale open/closed
sources for refresh, records the base/current-default branch distinction, and
ranks likely kernel work. Its output is an inventory, never an ingestion
input.

The score is triage, not a proof that low-scoring PRs are irrelevant. The
output reports how many merged PRs were already represented or filtered below
the score. Review that low-score bucket as well before advancing a global
cutoff.

An existing page is not silently deduplicated when GitHub now reports a
different terminal status (`merged` or closed-unmerged). It appears in the
existing-source refresh queue and must be updated in place. New
closed-unmerged PRs are counted but are not kernel candidates by default.

The repository default branch is read at scan time. If a repository renamed
its default branch, use the PR's `baseRefName`, merge history, and parent PRs
to establish the historical landing chain.

For a selected PR, capture authoritative detail:

```bash
gh pr view NUMBER --repo OWNER/REPO \
  --json author,baseRefName,body,createdAt,files,headRefName,mergeCommit,mergedAt,state,title,url
```

Do not advance the global cutoff after a selective scan. Advance it only when
every tracked repository reports complete boundary coverage and has been
reviewed through the same date. A selective batch is recorded in `candidates/`
instead.

## 3. Apply the Quality Gate

Prefer merged PRs. Open or closed-unmerged PRs require an explicit reason and
must not be described as shipped behavior.

Keep a PR only when all of these are true:

1. It changes a kernel, code generator, dispatch rule, or kernel-facing
   correctness/performance contract.
2. The PR body or changed files expose a concrete mechanism, not only a title.
3. The page can name at least one controlled `kernel_types` or `languages`
   value.
4. It is not docs-only, CI-only, dependency-only, or a duplicate.
5. Architecture claims come from explicit gfx/GPU evidence or changed paths.
6. Performance numbers include shape, dtype, GPU, baseline, and provenance.
7. The PR targets the repository default branch, or its later landing chain is
   identified and cited. A feature-branch merge is not a shipped date.

Follow supersession chains. A later correctness revert or replacement must be
captured with the current recommendation; an obsolete fast path must not be
presented as best practice.

## 4. Author the Source Page

Create one evidence-rich page under `sources/prs/<repo-key>/PR-NUMBER.md`.
Preserve established IDs, for example:

```text
ROCm/rocm-libraries  -> pr-rocm_libraries-NUMBER
vllm-project/vllm   -> pr-vllm-NUMBER
```

The page must contain:

- canonical repository, PR number, merged date, author, URL, and status;
- base branch and merge commit; for feature-branch PRs, the parent landing PR
  and upstream landing date;
- controlled architecture, tag, kernel, language, technique, and hardware
  fields;
- the implementation mechanism and affected paths;
- source-reported correctness and performance evidence with test conditions;
- limitations, feature flags, architecture boundaries, and follow-up risks.

Never put HTTP URLs in `sources:` or `related:` arrays. Those arrays contain
internal page IDs only.

## 5. Synthesize the Wiki

Update an existing curated page before creating a new one. Add the source ID
to its `sources:` list and explain the reusable design lesson, not just the PR
title. Keep PR-specific measurements attributed to the source page.

Use confidence conservatively:

- `source-reported`: upstream PR evidence was reviewed;
- `verified`: official documentation and upstream code agree;
- `experimental`: a local reproduction or assembly experiment lacks broader
  upstream guarantees.

Do not upgrade a claim to `verified` merely because a smoke test passed.

## 6. Regenerate and Validate

```bash
python3 scripts/generate-indices.py
python3 scripts/validate.py
python3 scripts/validate.py --strict path/to/changed-page.md [...]
python3 -m pytest tests/ -q
git diff --check
git status --short
```

Review the diff for generated-index drift, duplicate IDs, unresolved
references, off-vocabulary values, and accidental bulk pages.

`data/quality-baseline.yaml` is an incremental net warning budget. Full
validation fails if total debt rises above it; targeted `--strict` validation
ensures touched pages do not trade a new warning for a repaired legacy one.
Lower the budget when warnings are removed; never raise it to admit new
warnings. Repository-wide `--strict` remains the target once legacy debt
reaches zero.

## 7. Reproduce on an Idle AMD GPU

Reproduction is optional per PR, but the status must be explicit:

- `metadata-only`: page/schema checks;
- `compile-only`: target architecture compilation;
- `correctness`: upstream numerical test;
- `benchmark`: controlled baseline and candidate measurements.

Before using a shared node:

```bash
amd-smi process --gpu all
rocm-smi
```

Pin a specific idle device by HSA UUID, not only by ordinal:

```bash
export ROCR_VISIBLE_DEVICES=GPU-<uuid>
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
```

Before a PR-specific build, the repository's minimal HIP smoke can check that
the selected device, compiler, launch, synchronization, and copy path work:

```bash
/opt/rocm/bin/hipcc --offload-arch=gfx942 -O2 \
  scripts/hip_gpu_smoke.cpp -o /tmp/hip_gpu_smoke
/tmp/hip_gpu_smoke
```

This is platform evidence only. It must not be reported as correctness or
performance validation of a PR.

Record the upstream commit, ROCm version, GPU/gfx target, command, inputs,
warmup, repetitions, numerical tolerance, and raw result. Never extrapolate a
gfx942 result to gfx950, or the reverse.

## 8. Review and Publish

The final review should answer:

1. Which PRs were included, deferred, rejected, or found to be duplicates?
2. Which claims are source-reported versus locally reproduced?
3. Did page count, error count, warning count, and tests remain healthy?
4. Was the cutoff advanced legitimately?

Push or merge only after those answers and the generated indices are in the
same commit as the source and Wiki changes.

# Contributing to ROCm-KernelWiki-Q

This is a three-layer knowledge base: **sources** (raw PR/doc/blog pages) →
**wiki** (synthesized knowledge) → **queries** (generated indices). The most
important rule: **wiki pages must connect back to the source layer.**

## Authoring a new wiki page — checklist

1. **Pick the id prefix and required frontmatter** for the page type from
   `data/schemas.yaml` (e.g. `hw-` for `wiki-hardware`, `technique-` for
   `wiki-technique`). The id must start with that prefix.
2. **Use only controlled-vocabulary values** for `architectures`, `tags`,
   `kernel_types`, `languages`, `techniques`, `hardware_features` — see
   `data/tags.yaml` (canonical terms, `aliases`, and `aux_tags`).
3. **Cite sources.** Every wiki page must list real `sources:` ids — including
   at least one `pr-*` id when the topic is backed by a tracked PR. Find them:
   ```bash
   python3 scripts/query.py "<topic>" --type pr
   python3 scripts/get_page.py <pr-id>          # confirm it says what you cite
   ```
4. **Calibrate confidence honestly** (`verified` = official doc *and* upstream
   code; otherwise `source-reported` / `inferred` / `experimental`). Do not put
   unsourced performance numbers on a `verified` page; attach a `source_id` or
   mark them illustrative.
5. **Regenerate and validate, then commit the page AND queries/ together:**
   ```bash
   python3 scripts/generate-indices.py
   python3 scripts/validate.py        # 0 errors required; clean up warnings
   python3 -m pytest tests/ -q
   ```

## Adding / refreshing source PRs

See the *Data Refresh* section of `CLAUDE.md`. After `enrich_pr_pages.py`, always
run `reclassify_pr_pages.py` so `kernel_types` / `languages` / `techniques` are
populated from the changed-file paths (the original importer only read titles).

## CI

`.github/workflows/validate.yml` runs `validate.py`, asserts `queries/` is not
stale, and runs the tests. Keep `validate.py` at **0 errors**.

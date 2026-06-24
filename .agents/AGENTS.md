# Rule: PR Ingestion and Wiki Quality Control

When acting as a subagent or primary agent handling PR ingestion, automated scraping, or manual additions to the Ascend/ROCm Kernel Wikis and Context Hub, you MUST strictly adhere to the following quality control rules. These rules were derived from a massive 9-phase system purge that cleared over 10,000 validation warnings and errors.

## 1. No Empty PR Scrapes
- **Issue**: Previously, scraping scripts ingested thousands of PRs (from vLLM, Triton, MIOpen, etc.) that contained no technical implementation details.
- **Rule**: If a parsed PR fails to yield ANY `kernel_types` and ANY `languages` in its frontmatter, it is an invalid stub. **Do not save it.** If it is already saved, delete the source markdown file immediately.

## 2. No Orphaned Downstream Pages
- **Issue**: Deleting bad source PRs without deleting the summaries generated from them resulted in thousands of broken graph edges.
- **Rule**: If you delete a PR file in `sources/prs/`, you MUST simultaneously delete any corresponding summary page in `wiki/techniques/` (e.g., `wiki/techniques/pr-vllm-999.md`). Never leave a downstream page pointing to a deleted source.

## 3. Strict Internal Pointer Formats (No HTTP Links in `sources`)
- **Issue**: PR URLs like `https://github.com/vllm-project/vllm/pull/123` were incorrectly injected into the `sources:` array, causing indexer failures.
- **Rule**: The `sources:` and `related:` arrays in frontmatter must ONLY contain internal reference IDs (e.g., `pr-vllm-123`). If a raw HTTP link exists, it must be regex-translated into its canonical local ID.

## 4. Mandatory Tag Registration
- **Issue**: Cutting-edge tags (e.g., `mxfp8`, `runtime-dispatch`, `sparse-attention`) introduced by new PRs caused hundreds of "off-vocabulary" validation warnings.
- **Rule**: Before committing a new page with novel architecture features, optimization techniques, or kernel types, you MUST register the new tags in `data/tags.yaml` under their correct categorical array (`kernel_types`, `techniques`, `hardware_features`, or `aux_tags`).

## 5. Global Link Integrity on ID Changes
- **Issue**: Renaming hardware IDs from `hw-cube` to `wiki-hardware-cube` broke hundreds of incoming links across the entire repository.
- **Rule**: If you rename or modify an `id:` in the frontmatter of any document, you MUST perform a global workspace search-and-replace to update all files that referenced the old ID in their `related:` or `sources:` arrays.

## 6. Context-Hub API Minimum Quality Standard
- **Issue**: 1,828 ROCm APIs and several Ascend C APIs were mechanically dumped into the Context Hub as empty parameter stubs without implementations, violating LLM usage guidelines.
- **Rule**: Every single API document (`DOC.md`) committed to `context-hub` MUST contain a `## Semantics` section (explaining pseudo-code behavior) and a `## Example` section containing a complete, compiling C++/HIP/AscendC code block. Do not commit empty stubs.

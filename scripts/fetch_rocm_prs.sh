#!/bin/bash
set -e
cd /Users/haiyan-mini/Agent4Kernel/rocm-kernelwiki-q

REPOS=(
    "ROCm/triton"
    "ROCm/vllm"
    "ROCm/MIOpen"
    "ROCm/rocBLAS"
    "ROCm/rocWMMA"
    "ROCm/aotriton"
    "ROCm/flash-attention"
    "ROCm/hipBLASLt"
    "ROCm/composable_kernel"
)

for REPO in "${REPOS[@]}"; do
    REPO_NAME=$(basename $REPO)
    echo "======================================"
    echo "Fetching PRs for $REPO..."
    
    # Check if repo exists and is accessible
    if gh repo view $REPO >/dev/null 2>&1; then
        gh pr list --repo $REPO --state merged --limit 300 --json number,title,author,mergedAt,url,body > /tmp/${REPO_NAME}_prs.json
        if [ -s /tmp/${REPO_NAME}_prs.json ]; then
            echo "Generating pages for $REPO_NAME..."
            python3 scripts/generate_pr_pages.py --json /tmp/${REPO_NAME}_prs.json --repo ${REPO_NAME} || echo "Warning: Error generating pages for $REPO_NAME"
        else
            echo "Warning: No PRs found or failed to fetch for $REPO"
        fi
    else
        echo "Warning: Repo $REPO not found or inaccessible"
    fi
done

echo "======================================"
echo "Phase 1 PR fetching completed."

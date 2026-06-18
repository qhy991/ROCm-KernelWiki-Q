import os
import glob

repos = [
    "flash-attention",
    "hipblaslt",
    "composable_kernel",
    "triton",
    "vllm",
    "MIOpen",
    "rocBLAS",
    "rocWMMA",
    "aotriton"
]

base_dir = "/Users/haiyan-mini/Agent4Kernel/rocm-kernelwiki-q/sources/prs"
output_file = "/Users/haiyan-mini/.gemini/antigravity/brain/94869b0c-ed94-4506-a713-22e5e2ffceaa/scratch/rocm_pr_checklist.md"

with open(output_file, 'w') as f:
    f.write("# ROCm PR Processing Checklist\n\n")
    for repo in repos:
        repo_dir = os.path.join(base_dir, repo)
        if not os.path.isdir(repo_dir):
            continue
        pr_files = glob.glob(os.path.join(repo_dir, "*.md"))
        f.write(f"## {repo} ({len(pr_files)} PRs)\n")
        
        for pr_file in sorted(pr_files):
            basename = os.path.basename(pr_file).replace(".md", "")
            f.write(f"- [ ] {basename}\n")
        f.write("\n")

print(f"Checklist generated at {output_file}")

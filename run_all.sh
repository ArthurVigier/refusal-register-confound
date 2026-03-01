#!/bin/bash
# run_all.sh — Full experiment pipeline
# Runs all 7 notebooks in sequence. Cache-safe: completed models are skipped.
# Expected total runtime: ~10-12 hours on A100 (80GB)

set -e

echo "============================================"
echo "  Register Geometry — Full Pipeline"
echo "============================================"

cd "$(dirname "$0")"

run_notebook() {
    local nb="$1"
    local name=$(basename "$nb" .ipynb)
    echo ""
    echo "──────────────────────────────────────────"
    echo "  Running: $name"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "──────────────────────────────────────────"
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=36000 \
        --ExecutePreprocessor.kernel_name=python3 \
        --output-dir=notebooks/ \
        "$nb" 2>&1 | tail -5
    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
}

# Phase 1 — Core extraction (required by all downstream)
run_notebook notebooks/01_rhat_extraction.ipynb

# Phase 2 — Behavioral + RepEng (independent of each other)
run_notebook notebooks/02_refusal_rate.ipynb
run_notebook notebooks/03_repeng_baseline.ipynb

# Phase 3 — Figures (requires Phase 1-2)
run_notebook notebooks/04_figures.ipynb

# Phase 4 — Robustness + Register dissociation (requires Phase 1)
run_notebook notebooks/05_pair_robustness.ipynb
run_notebook notebooks/06_register_vs_content.ipynb

# Phase 5 — Content direction (requires Phase 1 + data/register_content_stimuli.csv)
run_notebook notebooks/07_content_direction.ipynb

echo ""
echo "============================================"
echo "  Pipeline complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results in: results/"
echo "============================================"

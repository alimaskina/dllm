#!/usr/bin/env bash
# MATH-500 adaptive_k_mixed: mixed K4/K2 + force-K2 sanity (no redundant KIVI2 baseline).
set -euo pipefail
cd "$(dirname "$0")"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${1:-../checkpoints/math500_adaptive_k_mixed_n100}"

export CUDA_VISIBLE_DEVICES="$GPU"
conda run --no-capture-output -n fast_dllm python run_kv_eval.py \
  --task hendrycks_math500 \
  --n 100 \
  --seed 1234 \
  --device cuda:0 \
  --max-new-tokens 1024 \
  --bd-size 32 \
  --threshold 1.0 \
  --mode adaptive_k_mixed_suite \
  --k4-mass 0.75 \
  --k2-mass-upper 0.95 \
  --high-precision-mass 0.75 \
  --out-dir "$OUT"

echo "Done → $OUT/summary.json"

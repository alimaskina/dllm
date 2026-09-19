#!/usr/bin/env bash
# MATH-500 n=100: uniform K2/V4 vs adaptive K2 + V4/V2
set -euo pipefail
cd "$(dirname "$0")"
conda run --no-capture-output -n fast_dllm python run_kv_eval.py \
  --task hendrycks_math500 \
  --n 100 \
  --seed 1234 \
  --max-new-tokens 1024 \
  --device cuda:0 \
  --mode k2v4_adaptive_suite \
  --out-dir ../checkpoints/math500_k2v4_adaptive_n100 \
  "$@"

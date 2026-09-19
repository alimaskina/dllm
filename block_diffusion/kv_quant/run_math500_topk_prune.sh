#!/usr/bin/env bash
# MATH500 n=100: top-K KV prune experiment (128fp / 256kivi8 / 512kivi4 / 1024kivi2)
set -euo pipefail
cd "$(dirname "$0")"
conda run --no-capture-output -n fast_dllm python run_math500_topk_prune.py \
  --n 100 \
  --seed 1234 \
  --max-new-tokens 1024 \
  --device cuda:0 \
  --out-dir ../checkpoints/math500_topk_prune_n100 \
  "$@"

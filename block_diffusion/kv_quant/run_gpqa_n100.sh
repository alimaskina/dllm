#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
conda run --no-capture-output -n fast_dllm python run_kv_eval.py \
  --task gpqa_diamond \
  --n 100 \
  --max-new-tokens 1536 \
  --mode kivi_suite \
  --device cuda:0 \
  --out-dir ../checkpoints/gpqa_diamond_kivi_n100 \
  2>&1 | tee ../checkpoints/gpqa_diamond_kivi_n100.log

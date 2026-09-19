#!/usr/bin/env bash
# baseline + KIVI4 + KIVI2 on MATH500 and GPQA diamond, n=100, GPU 0
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0

run_one() {
  local task=$1 out=$2 log=$3 maxtok=$4
  echo "=== ${task} n=100 max_new_tokens=${maxtok} → ${out} ==="
  conda run --no-capture-output -n fast_dllm python run_kv_eval.py \
    --task "${task}" \
    --n 100 \
    --max-new-tokens "${maxtok}" \
    --mode kivi_suite \
    --device cuda:0 \
    --out-dir "../checkpoints/${out}" \
    2>&1 | tee "../checkpoints/${log}"
}

run_one hendrycks_math500 gsm8k_math500_kivi_n100 math500_kivi_n100.log 1024
run_one gpqa_diamond gpqa_diamond_kivi_n100 gpqa_diamond_kivi_n100.log 1536

echo "All done."

#!/bin/bash
# Sweep bd_size on GSM8K — small_block_size = bd_size (no sub-blocks)
#
# Usage:
#   bash run_sweep.sh                  # n=64, bd=16,32,64
#   bash run_sweep.sh n128             # 128 samples
#   bash run_sweep.sh bd16,32          # custom bd sizes

set -euo pipefail

export HF_HOME="${HF_HOME:-/home/alimaskina/.cache/huggingface}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

N=64
BD_SIZES="16,32,64"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
OUT="checkpoints/sweep_bd_n64_sbeq"
LOG="results_sweep_bd_n${N}_sbeq.log"

for arg in "$@"; do
  case "$arg" in
    n[0-9]*) N="${arg#n}"; OUT="checkpoints/sweep_bd_n${N}_sbeq"; LOG="results_sweep_bd_n${N}_sbeq.log" ;;
    bd*) BD_SIZES="${arg#bd}" ;;
    *)
      echo "Unknown flag: $arg. Use: n<N> | bd<SIZES>"
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

echo "GPU=$GPU  n=$N  bd_sizes=$BD_SIZES  small_block=bd_size  out=$OUT"

conda run -n fast_dllm python run_sweep.py \
  --n "$N" \
  --bd-sizes "$BD_SIZES" \
  --device cuda:0 \
  --out-root "$OUT" \
  --small-block-equals-block \
  2>&1 | tee "$LOG"

conda run -n fast_dllm python compare_sweep.py --sweep-root "$OUT"

echo "Done → $OUT/compare_report.md"

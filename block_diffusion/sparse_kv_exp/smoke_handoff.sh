#!/usr/bin/env bash
# Quick smoke: 1 example × 2 configs on one GPU before full handoff sweep.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-fast_dllm_v2_7b}"
GPU="${DEVICE:-cuda:0}"
OUT="${OUTPUT_DIR:-${DIR}/results/handoff_smoke/${MODEL}}"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
case "$MODEL" in
  fast_dllm_v2_7b) conda activate fast_dllm ;;
  llada2_mini_16b) conda activate llada_quant ;;
  *) conda activate fast_dllm ;;
esac

mkdir -p "$OUT"

echo "Smoke handoff: model=$MODEL gpu=$GPU out=$OUT"

python "${DIR}/run_longbench_sweep.py" \
  --sweep handoff \
  --model "$MODEL" \
  --tasks 2wikimqa \
  --families baseline middle \
  --topk-pcts 5 \
  --num-examples 1 \
  --device "$GPU" \
  --output-dir "$OUT"

python "${DIR}/run_benchmark_sweep.py" \
  --sweep handoff \
  --model "$MODEL" \
  --tasks gsm8k \
  --families baseline middle \
  --topk-pcts 5 \
  --num-examples 1 \
  --device "$GPU" \
  --output-dir "$OUT"

python "${DIR}/run_handoff_report.py" --output-dir "$OUT"
echo "OK → ${OUT}/report.md"

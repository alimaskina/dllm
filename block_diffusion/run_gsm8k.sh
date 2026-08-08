#!/bin/bash
# Fast-dLLM v2 7B — GSM8K block volatility experiment
#
# Official eval params (NVlabs/Fast-dLLM v2/eval_script.sh):
#   num_fewshot=0, threshold=1, bd_size=32, small_block_size=8, max_new_tokens=2048
#   apply_chat_template + fewshot_as_multiturn (replicated in run_volatility.py)
#
# Usage:
#   bash run_gsm8k.sh              # 32 samples on GPU 3
#   bash run_gsm8k.sh smoke        # 4 samples
#   bash run_gsm8k.sh n128           # 128 samples
#   bash run_gsm8k.sh n128 bd64      # bd_size=64

set -euo pipefail

export HF_HOME="${HF_HOME:-/home/alimaskina/.cache/huggingface}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

N=32
SEED=1234
GPU="${CUDA_VISIBLE_DEVICES:-3}"
BD_SIZE=32
TAG="n${N}"

for arg in "$@"; do
  case "$arg" in
    smoke) N=4; TAG="smoke" ;;
    n[0-9]*) N="${arg#n}"; TAG="n${N}" ;;
    bd[0-9]*) BD_SIZE="${arg#bd}"; TAG="${TAG}_bd${BD_SIZE}" ;;
    *)
      echo "Unknown flag: $arg. Use: smoke | n<N> | bd<SIZE>"
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
OUT="checkpoints/gsm8k_volatility_${TAG}"
LOG="results_gsm8k_volatility_${TAG}.log"

echo "GPU=$GPU  n=$N  bd_size=$BD_SIZE  out=$OUT"

conda run -n fast_dllm python run_volatility.py \
  --n "$N" \
  --seed "$SEED" \
  --bd-size "$BD_SIZE" \
  --device "cuda:0" \
  --out-dir "$OUT" \
  2>&1 | tee "$LOG"

conda run -n fast_dllm python analyze_volatility.py \
  --traces "$OUT/traces.jsonl"

echo "Done → $OUT/"

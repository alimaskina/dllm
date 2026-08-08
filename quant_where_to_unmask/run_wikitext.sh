#!/bin/bash
# WikiText continuation generation with per-step traces.
#
# Usage:
#   bash run_wikitext.sh                    # fp16, 256 samples, validation
#   bash run_wikitext.sh smoke              # 32 samples
#   bash run_wikitext.sh n512               # 512 samples
#   bash run_wikitext.sh train n1024        # train split, 1024 samples
#   bash run_wikitext.sh seed99 n512        # different prompt shuffle
#   FRESH=1 bash run_wikitext.sh train n512 # wipe checkpoint and restart

set -euo pipefail

export HF_HOME="${HF_HOME:-/home/alimaskina/.cache/huggingface}"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

MODEL="GSAI-ML/LLaDA-8B-Base"
QUANT="fp16"
GPUS="${CUDA_VISIBLE_DEVICES:-1}"
N=256
PROMPT_TOKENS=48
GEN_LENGTH=64
STEPS=64
BLOCK_LENGTH=64
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
DATASET="wikitext-103-raw-v1"
SPLIT="validation"
SEED=42
TAGS=()

for arg in "$@"; do
  case "$arg" in
    smoke) N=32; TAGS+=("smoke") ;;
    train) SPLIT="train"; TAGS+=("train") ;;
    n[0-9]*) N="${arg#n}"; TAGS+=("n${N}") ;;
    seed[0-9]*) SEED="${arg#seed}"; TAGS+=("seed${SEED}") ;;
    *)
      echo "Unknown flag: $arg. Use: smoke | train | n<N> | seed<N>"
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPUS"

OUT="results_wikitext_${QUANT}_g${GEN_LENGTH}"
if ((${#TAGS[@]})); then
  OUT="${OUT}_$(IFS=_; echo "${TAGS[*]}")"
fi

LOG="${OUT}.log"
CHECKPOINT_DIR="checkpoints/${OUT}"

if [[ "${FRESH:-0}" == "1" ]]; then
  echo "FRESH=1 → removing ${CHECKPOINT_DIR}"
  rm -rf "$CHECKPOINT_DIR"
fi

echo "quant=$QUANT  GPU=$GPUS  n=$N  seed=$SEED  prompt_tokens=$PROMPT_TOKENS  gen/steps/block=$GEN_LENGTH"
echo "  dataset=$DATASET  split=$SPLIT"
echo "  log=$LOG  checkpoint=$CHECKPOINT_DIR"

conda run -n llada_quant python generate_wikitext.py \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --n "$N" \
  --seed "$SEED" \
  --prompt-tokens "$PROMPT_TOKENS" \
  --quant "$QUANT" \
  --model "$MODEL" \
  --gen-length "$GEN_LENGTH" \
  --steps "$STEPS" \
  --block-length "$BLOCK_LENGTH" \
  --gen-batch-size "$GEN_BATCH_SIZE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --meta-out "${CHECKPOINT_DIR}/prompts.json" \
  2>&1 | stdbuf -oL tee "$LOG"

echo "Done → $CHECKPOINT_DIR"
echo "Analyze:"
echo "  python analyze_multitoken_words.py --checkpoint $CHECKPOINT_DIR --kind alpha --word-tier lexical --report"

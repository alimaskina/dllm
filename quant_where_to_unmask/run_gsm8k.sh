#!/bin/bash
# GSM8K evaluation — official LLaDA setup via lm-eval
# Uses conda env llada_quant (same as quant_unmask)
#
# Paper settings (EVAL.md):
#   gen_length=1024, steps=1024, block_length=1024  → GSM8K ~70.3%
#   gen_length=256,   steps=256,   block_length=256    → GSM8K ~70.0% (~4× faster)
#
# Usage:
#   bash run_gsm8k.sh fp16                    # full, 1 GPU, 1024
#   bash run_gsm8k.sh fp16 smoke              # 10 samples
#   bash run_gsm8k.sh fp16 fast               # full, 1 GPU, 256 steps
#   bash run_gsm8k.sh fp16 2gpu               # full, 2 GPU, 1024
#   bash run_gsm8k.sh fp16 fast 2gpu          # full, 2 GPU, 256 steps
#   bash run_gsm8k.sh fp16 fast n256          # 256 samples, 1 GPU, 256 steps
#   bash run_gsm8k.sh int4 smoke fast 2gpu   # combine flags

set -euo pipefail

export HF_HOME="${HF_HOME:-/home/alimaskina/.cache/huggingface}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

MODEL="GSAI-ML/LLaDA-8B-Base"
GEN_LENGTH=1024
STEPS=1024
BLOCK_LENGTH=1024
NGPU=1
GPUS="${CUDA_VISIBLE_DEVICES:-2}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"

QUANT="${1:?Usage: run_gsm8k.sh <fp16|bf16|int4|int8> [smoke] [fast] [2gpu]}"
shift

case "$QUANT" in
  fp16) DTYPE="fp16" ;;
  bf16) DTYPE="bf16" ;;
  int4) DTYPE="int4" ;;
  int8) DTYPE="int8" ;;
  *)
    echo "Unknown quant: $QUANT. Use: fp16 | bf16 | int4 | int8"
    exit 1
    ;;
esac

# Parse optional mode flags
SMOKE=0
FAST=0
LIMIT_N=""
TAGS=()
for arg in "$@"; do
  case "$arg" in
    smoke) SMOKE=1; LIMIT_N=10 ;;
    fast)  FAST=1; GEN_LENGTH=256; STEPS=256; BLOCK_LENGTH=256 ;;
    2gpu)  NGPU=2; GPUS="${GPUS_2:-2,3}" ;;
    n[0-9]*)
      LIMIT_N="${arg#n}"
      ;;
    *)
      echo "Unknown flag: $arg. Use: smoke | fast | 2gpu | n<N>"
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPUS"

OUT="results_gsm8k_${DTYPE}"
LIMIT_ARGS=()

if [[ "$SMOKE" -eq 1 ]]; then
  OUT="${OUT}_smoke"
  TAGS+=("smoke")
fi
if [[ -n "$LIMIT_N" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT_N")
  TAGS+=("n${LIMIT_N}")
  if [[ "$SMOKE" -eq 0 ]]; then
    OUT="${OUT}_n${LIMIT_N}"
  fi
fi
if [[ "$FAST" -eq 1 ]]; then
  OUT="${OUT}_fast"
  TAGS+=("fast/256")
fi
if [[ "$NGPU" -eq 2 ]]; then
  OUT="${OUT}_2gpu"
  TAGS+=("2gpu")
fi

LOG="${OUT}.log"
CHECKPOINT_DIR="checkpoints/${OUT}"

if [[ "${FRESH:-0}" == "1" ]]; then
  echo "FRESH=1 → removing ${CHECKPOINT_DIR}"
  rm -rf "$CHECKPOINT_DIR"
fi

echo "quant=$DTYPE  GPUs=$GPUS  nproc=$NGPU  gen/steps/block=$GEN_LENGTH  gen_batch=$GEN_BATCH_SIZE  tags=${TAGS[*]:-full}"
echo "  log=$LOG  checkpoint=$CHECKPOINT_DIR"

MODEL_ARGS="model_path=${MODEL},quant=${DTYPE},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},checkpoint_dir=${CHECKPOINT_DIR},gen_batch_size=${GEN_BATCH_SIZE}"

run_eval() {
  if [[ "$NGPU" -gt 1 ]]; then
    local port="${MAIN_PORT:-29500}"
    conda run -n llada_quant accelerate launch \
      --num_processes "$NGPU" \
      --multi_gpu \
      --main_process_port "$port" \
      eval_llada.py \
      --tasks gsm8k \
      --model llada_dist \
      --model_args "$MODEL_ARGS" \
      --output_path "${OUT}.json" \
      --log_samples \
      "${LIMIT_ARGS[@]}"
  else
    conda run -n llada_quant python eval_llada.py \
      --tasks gsm8k \
      --model llada_dist \
      --model_args "$MODEL_ARGS" \
      --output_path "${OUT}.json" \
      --log_samples \
      "${LIMIT_ARGS[@]}"
  fi
}

run_eval 2>&1 | stdbuf -oL tee "$LOG"

echo "Done → ${OUT}.json"
echo "Checkpoint → ${CHECKPOINT_DIR}/"

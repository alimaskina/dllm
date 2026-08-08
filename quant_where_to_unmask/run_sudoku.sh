#!/bin/bash
# Sudoku evaluation — LLaDA via lm-eval + custom sudoku task
# Dataset: Ritvik19/sudoku-dataset (256 fixed samples, seed=42)
#
# Usage:
#   bash run_sudoku.sh fp16                    # full 256, 1 GPU
#   bash run_sudoku.sh fp16 smoke              # 10 samples
#   bash run_sudoku.sh fp16 fast 2gpu          # same as default (161 steps)
#   bash run_sudoku.sh fp16 fast 2gpu n256     # explicit (default size)
#   bash run_sudoku.sh fp16 8shot 2gpu n256   # 8-shot compact Input/Output format
#   bash run_sudoku.sh fp16 4x4 2gpu n256       # 4x4 Dream data, 8-shot, 100 test
#   bash run_sudoku.sh fp16 4x4 4shot 2gpu n256 # 4x4 Dream data, 4-shot, 100 test
#   bash run_sudoku.sh fp16 4x4 2shot 2gpu n256 # 4x4 Dream data, 2-shot, 100 test
#   bash run_sudoku.sh fp16 4x4 1shot large 2gpu  # 1-shot, 900 test (all Dream files)
#   bash run_sudoku.sh fp16 4x4 4shot large 2gpu  # 4-shot, 900 test
#   bash run_sudoku.sh fp16 4x4 2shot large notail 2gpu  # gen_length=20 (no tail)
#   bash run_sudoku.sh fp16 4x4 2shot large g21 2gpu    # gen_length=21 (+ trailing \n)

set -euo pipefail

export HF_HOME="${HF_HOME:-/home/alimaskina/.cache/huggingface}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"

MODEL="GSAI-ML/LLaDA-8B-Base"
# Gold solution is always 161 tokens (9 rows × "d d ... d\n").
GEN_LENGTH=161
STEPS=161
BLOCK_LENGTH=161
NGPU=1
GPUS="${CUDA_VISIBLE_DEVICES:-2}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
REMASKING="${REMASKING:-low_confidence}"
TASKS_INCLUDE="$(pwd)/tasks"
TASK="sudoku"

QUANT="${1:?Usage: run_sudoku.sh <fp16|bf16|int4|int8> [smoke] [fast|8shot|4x4] [2gpu] [n<N>]}"
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

SMOKE=0
FAST=0
FEWSHOT=0
FOURBY4=0
FOURSHOT4=0
TWOSHOT2=0
ONESHOT1=0
LARGE=0
NOTAIL=0
GEN_TAG=""
LIMIT_N=""
TAGS=()
for arg in "$@"; do
  case "$arg" in
    smoke) SMOKE=1; LIMIT_N=10 ;;
    fast)  FAST=1 ;;  # output tag only; gen/steps/block stay 161
    8shot) FEWSHOT=1; TASK="sudoku_8shot"; GEN_LENGTH=89; STEPS=89; BLOCK_LENGTH=89 ;;
    4x4)   FOURBY4=1; TASK="sudoku4_8shot"; GEN_LENGTH=24; STEPS=24; BLOCK_LENGTH=24; REMASKING="low_confidence"; FEWSHOT=1 ;;
    1shot) if [[ "$FOURBY4" -eq 1 ]]; then ONESHOT1=1; TASK="sudoku4_1shot"; else echo "1shot requires 4x4 flag"; exit 1; fi ;;
    large) LARGE=1 ;;
    notail|g20) NOTAIL=1; GEN_LENGTH=20; STEPS=20; BLOCK_LENGTH=20; GEN_TAG="g20" ;;
    g21) GEN_LENGTH=21; STEPS=21; BLOCK_LENGTH=21; GEN_TAG="g21" ;;
    2shot) if [[ "$FOURBY4" -eq 1 ]]; then TWOSHOT2=1; TASK="sudoku4_2shot"; else echo "2shot requires 4x4 flag"; exit 1; fi ;;
    4shot) if [[ "$FOURBY4" -eq 1 ]]; then FOURSHOT4=1; TASK="sudoku4_4shot"; else FEWSHOT=1; TASK="sudoku_8shot"; GEN_LENGTH=89; STEPS=89; BLOCK_LENGTH=89; fi ;;
    2gpu)  NGPU=2; GPUS="${GPUS_2:-2,3}" ;;
    n[0-9]*)
      LIMIT_N="${arg#n}"
      ;;
    *)
      echo "Unknown flag: $arg. Use: smoke | fast | 8shot | 4x4 | 1shot | 2shot | 4shot | large | notail | g21 | 2gpu | n<N>"
      exit 1
      ;;
  esac
done

if [[ "$LARGE" -eq 1 ]]; then
  if [[ "$ONESHOT1" -eq 1 ]]; then TASK="sudoku4_1shot_large"; fi
  if [[ "$TWOSHOT2" -eq 1 ]]; then TASK="sudoku4_2shot_large"; fi
  if [[ "$FOURSHOT4" -eq 1 ]]; then TASK="sudoku4_4shot_large"; fi
  LIMIT_N=""
fi

export CUDA_VISIBLE_DEVICES="$GPUS"

OUT="results_sudoku_${DTYPE}"
if [[ "$LARGE" -eq 1 ]]; then
  OUT="${OUT}_large"
fi
if [[ "$ONESHOT1" -eq 1 ]]; then
  OUT="${OUT}_4x4_1shot"
elif [[ "$TWOSHOT2" -eq 1 ]]; then
  OUT="${OUT}_4x4_2shot"
elif [[ "$FOURSHOT4" -eq 1 ]]; then
  OUT="${OUT}_4x4_4shot"
elif [[ "$FOURBY4" -eq 1 ]]; then
  OUT="${OUT}_4x4_8shot"
elif [[ "$FEWSHOT" -eq 1 ]]; then
  OUT="${OUT}_8shot"
fi
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
  TAGS+=("161")
fi
if [[ "$NGPU" -eq 2 ]]; then
  OUT="${OUT}_2gpu"
  TAGS+=("2gpu")
fi
if [[ -n "$GEN_TAG" ]]; then
  OUT="${OUT}_${GEN_TAG}"
  TAGS+=("$GEN_TAG")
fi

LOG="${OUT}.log"
CHECKPOINT_DIR="checkpoints/${OUT}"

if [[ "${FRESH:-0}" == "1" ]]; then
  echo "FRESH=1 → removing ${CHECKPOINT_DIR}"
  rm -rf "$CHECKPOINT_DIR"
fi

echo "quant=$DTYPE  task=$TASK  GPUs=$GPUS  nproc=$NGPU  gen/steps/block=$GEN_LENGTH/$STEPS/$BLOCK_LENGTH  remask=$REMASKING  gen_batch=$GEN_BATCH_SIZE  tags=${TAGS[*]:-full}"
echo "  log=$LOG  checkpoint=$CHECKPOINT_DIR"

MODEL_ARGS="model_path=${MODEL},quant=${DTYPE},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},remasking=${REMASKING},checkpoint_dir=${CHECKPOINT_DIR},gen_batch_size=${GEN_BATCH_SIZE}"

run_eval() {
  if [[ "$NGPU" -gt 1 ]]; then
    local port="${MAIN_PORT:-29500}"
    conda run -n llada_quant accelerate launch \
      --num_processes "$NGPU" \
      --multi_gpu \
      --main_process_port "$port" \
      eval_llada.py \
      --tasks "$TASK" \
      --include_path "$TASKS_INCLUDE" \
      --model llada_dist \
      --model_args "$MODEL_ARGS" \
      --output_path "${OUT}.json" \
      --log_samples \
      "${LIMIT_ARGS[@]}"
  else
    conda run -n llada_quant python eval_llada.py \
      --tasks "$TASK" \
      --include_path "$TASKS_INCLUDE" \
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

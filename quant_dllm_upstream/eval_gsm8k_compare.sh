#!/usr/bin/env bash
# Compare base vs Quant-dLLM 2-bit (dequant FP) on GSM8K, identical hypers.
# Paper-best LLaDA GSM8K: gen/steps/block=1024, remasking=low_confidence, 5-shot.
set -euo pipefail

export HF_HOME="${HF_HOME:-/home/alimaskina/.cache/huggingface}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONUNBUFFERED=1

EVAL_DIR="/home/alimaskina/dllm/quant_where_to_unmask"
OUT_ROOT="/home/alimaskina/dllm/quant_dllm_upstream/eval_results/gsm8k_n20_paper"
PYTHON="/home/alimaskina/miniconda3/envs/llada_quant/bin/python"

BASE_MODEL="/home/alimaskina/dllm/model/LLaDA-8B-Base"
QUANT_MODEL="/home/alimaskina/dllm/quant_dllm_upstream/output/LLaDA-8B-Base_c4_arb-rc_128_hessian_nump_1_order2group_True_gptaq_False_disable_mask_True_no_mask_order_2_slim_False_abmp_ratio_0.05_mcs_prefix_0.25_seed_0_maskx+salience+slim.pt"

VARIANT="${1:?Usage: $0 <base|quant>}"
GPU="${2:?Usage: $0 <base|quant> <gpu_index>}"
LIMIT="${LIMIT:-20}"
GEN_LENGTH="${GEN_LENGTH:-1024}"
STEPS="${STEPS:-1024}"
BLOCK_LENGTH="${BLOCK_LENGTH:-1024}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
# Traces make 1024-step GSM8K ~unusable; accuracy compare does not need them.
RECORD_TRACES="${RECORD_TRACES:-0}"

case "$VARIANT" in
  base)  MODEL="$BASE_MODEL"; TAG="base_fp16" ;;
  quant) MODEL="$QUANT_MODEL"; TAG="quant2bit_fp16" ;;
  *) echo "variant must be base|quant"; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT_ROOT"
cd "$EVAL_DIR"

OUT="${OUT_ROOT}/${TAG}"
LOG="${OUT}.log"
MODEL_ARGS="model_path=${MODEL},quant=fp16,gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},remasking=low_confidence,gen_batch_size=${GEN_BATCH_SIZE}"
if [[ "$RECORD_TRACES" == "1" ]]; then
  CKPT="${OUT_ROOT}/checkpoints_${TAG}"
  rm -rf "$CKPT"
  mkdir -p "$CKPT"
  MODEL_ARGS="${MODEL_ARGS},checkpoint_dir=${CKPT}"
fi

echo "=== GSM8K compare ==="
echo "variant=$VARIANT gpu=$GPU limit=$LIMIT"
echo "hypers gen/steps/block=${GEN_LENGTH}/${STEPS}/${BLOCK_LENGTH} fewshot=5 remasking=low_confidence"
echo "model=$MODEL"
echo "log=$LOG"

"$PYTHON" eval_llada.py \
  --tasks gsm8k \
  --model llada_dist \
  --model_args "$MODEL_ARGS" \
  --num_fewshot 5 \
  --limit "$LIMIT" \
  --batch_size 1 \
  --output_path "${OUT}.json" \
  --log_samples \
  2>&1 | stdbuf -oL tee "$LOG"

echo "Done → ${OUT}.json"

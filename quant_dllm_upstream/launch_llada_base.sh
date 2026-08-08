#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export QUANT_DLLM_MODEL_DIR="${QUANT_DLLM_MODEL_DIR:-/home/alimaskina/dllm/model}"
export QUANT_DLLM_DATA_DIR="${QUANT_DLLM_DATA_DIR:-/home/alimaskina/dllm/data}"
export QUANT_DLLM_OUTPUT_DIR="${QUANT_DLLM_OUTPUT_DIR:-$ROOT/output}"
export QUANT_DLLM_LOG_DIR="${QUANT_DLLM_LOG_DIR:-$ROOT/log}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="${PYTHON:-/home/alimaskina/miniconda3/envs/quant/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEQLEN="${SEQLEN:-2048}"
NSAMPLES="${NSAMPLES:-4}"
SLIM="${SLIM:-0}"
ABMP_RATIO="${ABMP_RATIO:-0.05}"
DISABLE_GPTQ="${DISABLE_GPTQ:-1}"

mkdir -p "$QUANT_DLLM_LOG_DIR" "$QUANT_DLLM_OUTPUT_DIR"
LOG="$QUANT_DLLM_LOG_DIR/run_active_stdout.log"
echo $$ >"$QUANT_DLLM_LOG_DIR/run_active.pid"

EXTRA=()
if [[ "$SLIM" == "1" ]]; then
  EXTRA+=(--slim --abmp_ratio "$ABMP_RATIO")
fi
if [[ "$DISABLE_GPTQ" == "1" ]]; then
  EXTRA+=(--disable_gptq)
fi

exec "$PYTHON" "$ROOT/run_arb_llada.py" \
  "$QUANT_DLLM_MODEL_DIR/LLaDA-8B-Base" c4 arb-rc \
  --blocksize 128 --salient_metric hessian --device "$DEVICE" \
  --save --num_p 1 --order2_group --mcs_prefix_ratio 0.25 \
  --disable_mask --no_mask_order 2 \
  --seqlen "$SEQLEN" --nsamples "$NSAMPLES" \
  "${EXTRA[@]}" \
  >"$LOG" 2>&1

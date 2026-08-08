#!/bin/bash
# Experiment 1: Fixed-state fidelity
# Usage: bash run_exp1.sh [smoke|int8|int4|gptq]

set -e
cd "$(dirname "$0")"

MODEL="GSAI-ML/LLaDA-8B-Base"
N=20
COMP=32

MODE="${1:-int4}"

case "$MODE" in
  smoke)
    # Quick sanity check: small n, no model download needed if already cached
    python exp1_fidelity.py \
      --model "$MODEL" \
      --fp-quant fp16 \
      --q-quant int4 \
      --n-samples 5 \
      --comp-len 16 \
      --out results_exp1_smoke.json
    ;;
  int8)
    python exp1_fidelity.py \
      --model "$MODEL" \
      --fp-quant fp16 \
      --q-quant int8 \
      --n-samples $N \
      --comp-len $COMP \
      --out results_exp1_int8.json
    ;;
  int4)
    python exp1_fidelity.py \
      --model "$MODEL" \
      --fp-quant fp16 \
      --q-quant int4 \
      --n-samples $N \
      --comp-len $COMP \
      --out results_exp1_int4.json
    ;;
  gptq)
    # Use pre-quantized GPTQ checkpoint from HF
    python exp1_fidelity.py \
      --model "$MODEL" \
      --fp-quant fp16 \
      --q-quant fp16 \
      --q-model "FunAGI/LLaDA-8B-Instruct-gptqmodel-4bit" \
      --n-samples $N \
      --comp-len $COMP \
      --out results_exp1_gptq.json
    ;;
  *)
    echo "Unknown mode: $MODE. Use: smoke | int8 | int4 | gptq"
    exit 1
    ;;
esac

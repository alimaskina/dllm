#!/usr/bin/env bash
# Batch: codebook v3, Fast-dLLM native sweep, short prompt gen.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/results"
cd "$ROOT"

echo "=== GPU 0: Codebook Dream v3 (few-shot + oracle compare) ==="
CUDA_VISIBLE_DEVICES=0 nohup python codebook_segmentation_audit.py \
  --dlm dream --ar qwen --n-trials 200 --fewshot 2 --compare-oracle \
  --device cuda:0 \
  --out results/codebook_segmentation_dream_v3.json \
  > results/codebook_dream_v3.log 2>&1 &
echo "codebook_v3_pid=$!"

echo "=== GPU 1: Fast-dLLM native sweep, then short prompt gen ==="
CUDA_VISIBLE_DEVICES=1 nohup bash -c '
  python fast_dllm_policy_sweep.py \
    --max-samples 60 --block-size 128 --device cuda:0 \
    --out results/policy_sweep_fast_dllm_native.json \
    > results/policy_sweep_fast_dllm_native.log 2>&1
  python prompt_gen_eval.py \
    --max-samples 500 --min-tokens 12 --max-tokens 48 \
    --device cuda:0 \
    --out results/prompt_gen_eval_llada_short.json \
    >> results/prompt_gen_short.log 2>&1
' > results/gpu1_chain.log 2>&1 &
echo "gpu1_chain_pid=$!"

echo "Jobs launched. Tail: results/*.log"

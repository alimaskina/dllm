#!/usr/bin/env bash
# Launch pending PoC experiments: codebook #5 + Dream/Fast-dLLM replication.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/results"
cd "$ROOT"

echo "=== GPU 0: Codebook DLM vs AR (Dream vs Qwen7B) ==="
CUDA_VISIBLE_DEVICES=0 nohup python codebook_segmentation_audit.py \
  --dlm dream --ar qwen --n-trials 200 --device cuda:0 \
  --out results/codebook_segmentation_dream.json \
  > results/codebook_dream.log 2>&1 &
echo "codebook_pid=$!"

echo "=== GPU 3: Dream policy sweep (low_confidence) ==="
CUDA_VISIBLE_DEVICES=3 nohup python policy_sweep.py \
  --model dream --policies low_confidence,topk_margin,random \
  --max-samples 60 --device cuda:0 \
  --out results/policy_sweep_dream.json \
  > results/policy_sweep_dream.log 2>&1 &
echo "dream_sweep_pid=$!"

echo "=== GPU 5: Fast-dLLM policy sweep ==="
CUDA_VISIBLE_DEVICES=5 nohup python policy_sweep.py \
  --model fast_dllm --policies low_confidence,topk_margin,random \
  --max-samples 60 --device cuda:0 \
  --out results/policy_sweep_fast_dllm.json \
  > results/policy_sweep_fast_dllm.log 2>&1 &
echo "fast_dllm_sweep_pid=$!"

echo "All jobs launched. Tail logs in results/*.log"

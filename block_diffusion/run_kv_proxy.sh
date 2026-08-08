#!/usr/bin/env bash
# One-shot KV proxy experiment launcher
set -euo pipefail
cd "$(dirname "$0")"

N=${1:-8}
OUT=${2:-checkpoints/kv_proxy_n${N}}
GPU=${CUDA_VISIBLE_DEVICES:-1}

echo "=== KV proxy experiment: n=${N} → ${OUT} (GPU ${GPU}) ==="
CUDA_VISIBLE_DEVICES=${GPU} conda run --no-capture-output -n fast_dllm python run_kv_proxy.py \
  --n "${N}" \
  --max-new-tokens 512 \
  --out-dir "${OUT}"

conda run --no-capture-output -n fast_dllm python analyze_kv_proxy.py \
  --traces "${OUT}/attn_traces.jsonl"

echo "=== Done: ${OUT}/kv_proxy_report.md ==="

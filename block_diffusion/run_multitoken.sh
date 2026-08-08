#!/bin/bash
# Multi-token word analysis on block-diffusion traces
#
# Usage:
#   bash run_multitoken.sh checkpoints/sweep_bd/n128_bd32/traces.jsonl
#   bash run_multitoken.sh checkpoints/sweep_bd/n128_bd32   # auto-find traces.jsonl

set -euo pipefail
cd "$(dirname "$0")"

TRACES="${1:-checkpoints/sweep_bd/n128_bd32/traces.jsonl}"
if [[ -d "$TRACES" ]]; then
  TRACES="${TRACES%/}/traces.jsonl"
fi
OUT_DIR="$(dirname "$TRACES")"
REPORT="${OUT_DIR}/multitoken_report.md"

echo "traces=$TRACES"
conda run -n fast_dllm python analyze_multitoken_words.py --traces "$TRACES" --report
conda run -n fast_dllm python generate_multitoken_report.py --traces "$TRACES" --out "$REPORT"
echo "Done → $REPORT"

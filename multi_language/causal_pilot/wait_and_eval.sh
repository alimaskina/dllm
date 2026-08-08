#!/usr/bin/env bash
set -euo pipefail
PIDS=(3194567 3195298 3195300)
for pid in "${PIDS[@]}"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
cd /home/alimaskina/dllm/multi_language
CUDA_VISIBLE_DEVICES=2 bash causal_pilot/run_causal_pilot.sh eval 2>&1 | tee causal_pilot/logs_eval.txt

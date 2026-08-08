#!/bin/bash
# Monitor GSM8K eval progress (lm-eval writes results only at the end).
#
# Usage:
#   bash watch_progress.sh              # auto-detect running jobs
#   bash watch_progress.sh --watch 30   # refresh every 30s

set -euo pipefail

WATCH="${1:-}"
INTERVAL="${2:-30}"

# Expected seconds per sample (256 steps), from smoke extrapolation
FP16_SEC=51
INT4_SEC=73
TOTAL=256

show_status() {
  local now
  now=$(date '+%H:%M:%S')
  echo "=== $now ==="

  local found=0
  for quant in fp16 int4; do
    local pids
    pids=$(pgrep -f "python3.10 -u eval_llada.py.*quant=${quant}.*limit 256" 2>/dev/null || true)
    [[ -z "$pids" ]] && continue
    found=1

    local nproc elapsed_sec gpu_ids
    nproc=$(echo "$pids" | wc -l)
    # elapsed from oldest worker
    local oldest_pid
    oldest_pid=$(echo "$pids" | head -1)
    elapsed_sec=$(ps -p "$oldest_pid" -o etimes= 2>/dev/null | tr -d ' ')

    if [[ "$quant" == "fp16" ]]; then
      gpu_ids="2,3"
      sec_per_sample=$FP16_SEC
    else
      gpu_ids="4,5"
      sec_per_sample=$INT4_SEC
    fi

    local per_gpu samples_done pct remaining_min
    per_gpu=$((TOTAL / nproc))
    samples_done=$((elapsed_sec / sec_per_sample))
    if (( samples_done > per_gpu )); then samples_done=$per_gpu; fi
    pct=$((samples_done * 100 / per_gpu))
    remaining_min=$(( (per_gpu - samples_done) * sec_per_sample / 60 ))

    echo ""
    echo "[$quant] GPUs $gpu_ids | workers=$nproc | elapsed=$((elapsed_sec/60))m$((elapsed_sec%60))s"
    echo "  ~$samples_done / $per_gpu per GPU (~${pct}%) | ETA ~${remaining_min} min"
    echo "  GPU util:"
    IFS=',' read -ra IDS <<< "$gpu_ids"
    for id in "${IDS[@]}"; do
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i "$id" 2>/dev/null \
        | awk -F', ' '{printf "    GPU %s: %s util, %s VRAM\n", $1, $2, $3}'
    done

    # checkpoint progress (written after each sample)
    local ckpt_dir
    ckpt_dir=$(ls -d checkpoints/results_gsm8k_${quant}* 2>/dev/null | head -1 || true)
    if [[ -n "$ckpt_dir" && -d "$ckpt_dir" ]]; then
      local total_lines=0
      echo "  checkpoint ($ckpt_dir):"
      for f in "$ckpt_dir"/rank*.jsonl; do
        [[ -f "$f" ]] || continue
        local n
        n=$(wc -l < "$f")
        total_lines=$((total_lines + n))
        echo "    $(basename "$f"): $n"
      done
      if [[ "$total_lines" -gt 0 ]]; then
        echo "  total checkpointed: $total_lines / $TOTAL"
        local trace_count=0
        if [[ -d "$ckpt_dir/traces" ]]; then
          trace_count=$(find "$ckpt_dir/traces" \( -name '*.json.gz' -o -name '*.json' \) 2>/dev/null | wc -l)
          echo "  traces saved: $trace_count"
        fi
        conda run -n llada_quant python score_checkpoints.py "$ckpt_dir" --limit "$TOTAL" 2>/dev/null | sed 's/^/    /'
      fi
    fi
  done

  if [[ "$found" -eq 0 ]]; then
    echo "No running n256 eval jobs found."
    echo ""
    echo "Recent results:"
    ls -lt results_gsm8k_*n256* 2>/dev/null | head -6 || echo "  (none)"
    # try to show accuracy from completed json
    for f in results_gsm8k_*_n256_fast_2gpu.json/GSAI-ML__*/results_*.json; do
      [[ -f "$f" ]] || continue
      python3 -c "
import json,sys
d=json.load(open('$f'))
r=d.get('results',{}).get('gsm8k',{})
for k,v in r.items():
    if 'exact_match' in k and 'stderr' not in k:
        print(f'  {k}: {v:.1%}' if isinstance(v,float) and v<=1 else f'  {k}: {v}')
" 2>/dev/null
    done
  fi
  echo ""
}

cd "$(dirname "$0")"

if [[ "$WATCH" == "--watch" ]]; then
  while true; do
    clear 2>/dev/null || true
    show_status
    sleep "$INTERVAL"
  done
else
  show_status
  echo "Tip: bash watch_progress.sh --watch 30"
fi

#!/usr/bin/env bash
# Wait for LongBench per-head sweep, then launch GSM8K / MATH500 / GPQA on 3 GPUs.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUTPUT_DIR:-${DIR}/results/benchmark_sweep_perhead}"
NUM_EX="${NUM_EXAMPLES:-50}"
DEVICE_LIST="${DEVICES:-cuda:6,cuda:3,cuda:7}"
TASKS=(gsm8k math500 gpqa_diamond)
WAIT="${WAIT_FOR_LONGBENCH:-1}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fast_dllm
mkdir -p "$OUT"

if [[ "$WAIT" == "1" ]]; then
  echo "Waiting for LongBench sweep to finish..."
  while pgrep -f "run_longbench_sweep.py" >/dev/null 2>&1; do
    sleep 60
  done
  echo "LongBench done."
fi

pkill -f "run_benchmark_sweep.py.*${OUT#${DIR}/}" 2>/dev/null || true
sleep 1

IFS=',' read -ra GPUS <<< "$DEVICE_LIST"
if (( ${#GPUS[@]} < ${#TASKS[@]} )); then
  echo "Need ${#TASKS[@]} GPUs, got ${#GPUS[@]}" >&2
  exit 1
fi

launch() {
  local gpu="$1" task="$2"
  local log="${OUT}/run_${task}_${gpu//:/}.log"
  echo "→ $task on $gpu"
  nohup python "${DIR}/run_benchmark_sweep.py" \
    --tasks "$task" \
    --num-examples "$NUM_EX" \
    --device "$gpu" \
    --output-dir "$OUT" \
    > "$log" 2>&1 &
  echo "  PID=$!"
}

for i in "${!TASKS[@]}"; do
  launch "${GPUS[$i]}" "${TASKS[$i]}"
done

echo ""
echo "Benchmark sweep → ${OUT}/results.jsonl"
echo "Total runs: 13 configs × ${NUM_EX} ex × 3 tasks = $((13 * NUM_EX * 3))"
echo "Monitor: tail -f ${OUT}/run_*.log"

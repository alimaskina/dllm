#!/usr/bin/env bash
# Launch LongBench sweep shards on multiple GPUs (one task per GPU).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUTPUT_DIR:-${DIR}/results/longbench_sweep_perhead}"
NUM_EX="${NUM_EXAMPLES:-50}"
DEVICE_LIST="${DEVICES:-cuda:6,cuda:3,cuda:5,cuda:7}"
TASKS=(2wikimqa narrativeqa qmsum repobench-p)
# GPU i runs TASKS[i]: cuda:6=2wikimqa, cuda:3=narrativeqa, cuda:5=qmsum, cuda:7=repobench-p

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fast_dllm
mkdir -p "$OUT"

IFS=',' read -ra GPUS <<< "$DEVICE_LIST"
N=${#GPUS[@]}
if (( N < ${#TASKS[@]} )); then
  echo "Need at least ${#TASKS[@]} GPUs, got $N" >&2
  exit 1
fi

# Stop old sweep shards in the same output dir only
pkill -f "run_longbench_sweep.py.*${OUT#${DIR}/}" 2>/dev/null || true
sleep 1

launch() {
  local gpu="$1" task="$2"
  local log="${OUT}/run_${task}_${gpu//:/}.log"
  echo "→ $task on $gpu (log: $log)"
  nohup python "${DIR}/run_longbench_sweep.py" \
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
echo "Launched ${#TASKS[@]} shards → ${OUT}/results.jsonl"
echo "Monitor: tail -f ${OUT}/run_*.log"
echo "Progress: wc -l ${OUT}/results.jsonl"
echo "Report:   python ${DIR}/run_longbench_sweep.py --analyze-only --output-dir ${OUT}"

#!/usr/bin/env bash
# Handoff sweep: LongBench (4) + GSM8K + MATH500 + GPQA on N GPUs.
#
# Usage:
#   MODEL=fast_dllm_v2_7b DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 NUM_EXAMPLES=250 \
#     bash launch_handoff.sh
#
#   MODEL=llada2_mini_16b DEVICES=cuda:4,cuda:5,cuda:6,cuda:7 NUM_EXAMPLES=250 \
#     bash launch_handoff.sh
#
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-fast_dllm_v2_7b}"
NUM_EX="${NUM_EXAMPLES:-250}"
SEED="${SEED:-1234}"
DEVICE_LIST="${DEVICES:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
OUT="${OUTPUT_DIR:-${DIR}/results/handoff/${MODEL}}"

# 7 tasks: 4 LongBench + 3 benchmarks
LONGBENCH_TASKS=(2wikimqa narrativeqa qmsum repobench-p)
BENCH_TASKS=(gsm8k math500 gpqa_diamond)

CONDA_ENV="${CONDA_ENV:-}"
if [[ -z "$CONDA_ENV" ]]; then
  case "$MODEL" in
    fast_dllm_v2_7b) CONDA_ENV=fast_dllm ;;
    llada2_mini_16b) CONDA_ENV=llada_quant ;;
    *) CONDA_ENV=fast_dllm ;;
  esac
fi

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "$CONDA_ENV"
mkdir -p "$OUT"

python - <<PY
import json
from pathlib import Path
from handoff_sweep_configs import HANDOFF_FAMILIES, HANDOFF_TOPK_PCTS
meta = {
    "sweep": "handoff",
    "model": "${MODEL}",
    "topks": [],
    "topk_pcts": list(HANDOFF_TOPK_PCTS),
    "families": list(HANDOFF_FAMILIES),
    "num_examples": ${NUM_EX},
}
Path("${OUT}/sweep_meta.json").write_text(json.dumps(meta, indent=2))
PY

IFS=',' read -ra GPUS <<< "$DEVICE_LIST"
ALL_TASKS=("${LONGBENCH_TASKS[@]}" "${BENCH_TASKS[@]}")
N_GPUS=${#GPUS[@]}
N_TASKS=${#ALL_TASKS[@]}

echo "Model:      $MODEL"
echo "Conda env:  $CONDA_ENV"
echo "Output:     $OUT"
echo "Examples:   $NUM_EX per task"
echo "GPUs:       ${GPUS[*]} ($N_GPUS)"
echo "Tasks:      ${ALL_TASKS[*]} ($N_TASKS)"
echo ""

python - <<PY
from handoff_sweep_configs import handoff_run_count, select_handoff_configs
n = handoff_run_count(num_examples=${NUM_EX})
print(f"Configs:    {len(select_handoff_configs())} (baseline + middle + uniform5 + 2×extreme × 4 pct)")
print(f"Total runs: {n} (= configs × ${NUM_EX} ex × 7 tasks)")
PY

pkill -f "run_longbench_sweep.py.*${OUT#${DIR}/}" 2>/dev/null || true
pkill -f "run_benchmark_sweep.py.*${OUT#${DIR}/}" 2>/dev/null || true
sleep 1

launch_lb() {
  local gpu="$1" task="$2"
  local log="${OUT}/run_lb_${task}_${gpu//:/}.log"
  echo "→ LongBench $task on $gpu"
  nohup python "${DIR}/run_longbench_sweep.py" \
    --sweep handoff \
    --model "$MODEL" \
    --tasks "$task" \
    --num-examples "$NUM_EX" \
    --seed "$SEED" \
    --device "$gpu" \
    --output-dir "$OUT" \
    > "$log" 2>&1 &
  echo "  PID=$!  log=$log"
}

launch_bench() {
  local gpu="$1" task="$2"
  local log="${OUT}/run_bench_${task}_${gpu//:/}.log"
  echo "→ Benchmark $task on $gpu"
  nohup python "${DIR}/run_benchmark_sweep.py" \
    --sweep handoff \
    --model "$MODEL" \
    --tasks "$task" \
    --num-examples "$NUM_EX" \
    --seed "$SEED" \
    --device "$gpu" \
    --output-dir "$OUT" \
    > "$log" 2>&1 &
  echo "  PID=$!  log=$log"
}

for i in "${!ALL_TASKS[@]}"; do
  gpu="${GPUS[$((i % N_GPUS))]}"
  task="${ALL_TASKS[$i]}"
  if [[ " ${LONGBENCH_TASKS[*]} " == *" ${task} "* ]]; then
    launch_lb "$gpu" "$task"
  else
    launch_bench "$gpu" "$task"
  fi
done

echo ""
echo "Monitor:  tail -f ${OUT}/run_*.log"
echo "Progress: wc -l ${OUT}/results.jsonl"
echo "Report:   python ${DIR}/run_handoff_report.py --output-dir ${OUT}"

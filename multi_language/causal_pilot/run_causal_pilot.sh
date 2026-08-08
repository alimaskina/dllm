#!/usr/bin/env bash
# Causal pilot: IID vs WORD vs SPAN continued pretraining on LLaDA-8B-Base.
#
# Phases:
#   0. build probe set (held-out, fixed before any training)
#   1. eval base checkpoint on probe
#   2. train 3 compute-matched runs (same steps/LR/data)
#   3. eval each run on probe
#   4. compare NLL_IID - NLL_WORD / NLL_SPAN
#
# Usage:
#   bash run_causal_pilot.sh              # full pipeline
#   bash run_causal_pilot.sh probe        # phase 0 only
#   bash run_causal_pilot.sh train        # phase 2 only (all 3 modes)
#   bash run_causal_pilot.sh eval         # phase 3+4 (needs trained ckpts)
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PILOT="$ROOT/causal_pilot"
cd "$ROOT"
mkdir -p "$PILOT/results" "$PILOT/checkpoints"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="cuda:0"
STEPS="${MAX_STEPS:-800}"
LR="${LR:-1e-5}"
INTERVENTION="${INTERVENTION_PROB:-0.25}"
INIT="${INIT_CHECKPOINT:-/home/alimaskina/dllm/model/LLaDA-8B-Base}"
PROBE="$PILOT/probe_set_en.json"

run_probe_build() {
  echo "=== Phase 0: build probe set ==="
  CUDA_VISIBLE_DEVICES="$GPU" python "$PILOT/probe_set.py" \
    --n-probe 120 --seed 42 --out "$PROBE"
}

run_eval_base() {
  echo "=== Phase 1: eval BASE on probe ==="
  CUDA_VISIBLE_DEVICES="$GPU" python "$PILOT/eval_probe.py" \
    --run-name base --checkpoint "$INIT" --probe-json "$PROBE" \
    --device "$DEVICE" --out "$PILOT/results/nll_probe_base.json"
}

run_train_mode() {
  local mode="$1"
  local seed="$2"
  echo "=== Train: $mode (seed=$seed) ==="
  CUDA_VISIBLE_DEVICES="$GPU" python "$PILOT/train.py" \
    --mode "$mode" \
    --init-checkpoint "$INIT" \
    --max-steps "$STEPS" \
    --lr "$LR" \
    --intervention-prob "$INTERVENTION" \
    --seed "$seed" \
    --device "$DEVICE" \
    --out-dir "$PILOT/checkpoints"
}

run_train_all() {
  run_train_mode iid 100
  run_train_mode word 101
  run_train_mode span 102
}

# Parallel training: one mode per GPU (GPUS="2,3,4" by default).
run_train_parallel() {
  local gpus="${GPUS:-2,3,4}"
  IFS=',' read -ra GPU_LIST <<< "$gpus"
  if ((${#GPU_LIST[@]} < 3)); then
    echo "GPUS must list 3 devices for iid/word/span (got: $gpus)"
    exit 1
  fi
  local modes=(iid word span)
  local seeds=(100 101 102)
  local pids=()
  for i in 0 1 2; do
    local mode="${modes[$i]}"
    local gpu="${GPU_LIST[$i]}"
    echo "=== Train parallel: $mode on GPU $gpu (seed=${seeds[$i]}) ==="
    CUDA_VISIBLE_DEVICES="$gpu" python "$PILOT/train.py" \
      --mode "$mode" \
      --init-checkpoint "$INIT" \
      --max-steps "$STEPS" \
      --lr "$LR" \
      --intervention-prob "$INTERVENTION" \
      --seed "${seeds[$i]}" \
      --device cuda:0 \
      --out-dir "$PILOT/checkpoints" \
      > "$PILOT/logs_train_${mode}.txt" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if ((fail)); then
    echo "One or more parallel training jobs failed"
    exit 1
  fi
}

run_eval_runs() {
  echo "=== Phase 3: eval trained runs on probe ==="
  for mode in iid word span; do
    CUDA_VISIBLE_DEVICES="$GPU" python "$PILOT/eval_probe.py" \
      --run-name "$mode" \
      --checkpoint "$PILOT/checkpoints/$mode/final" \
      --probe-json "$PROBE" \
      --device "$DEVICE" \
      --out "$PILOT/results/nll_probe_${mode}.json"
  done
}

run_compare() {
  echo "=== Phase 4: compare deltas ==="
  python "$PILOT/compare_runs.py" \
    --results-dir "$PILOT/results" \
    --out "$PILOT/results/causal_pilot_comparison.json"
}

PHASE="${1:-all}"
case "$PHASE" in
  probe) run_probe_build ;;
  base) run_probe_build; run_eval_base ;;
  train) run_train_all ;;
  train_parallel) run_train_parallel ;;
  eval) run_eval_runs; run_compare ;;
  all)
    run_probe_build
    run_eval_base
    run_train_all
    run_eval_runs
    run_compare
    ;;
  *)
    echo "Unknown phase: $PHASE (probe|base|train|train_parallel|eval|all)"
    exit 1
    ;;
esac

echo "Done. Results: $PILOT/results/"

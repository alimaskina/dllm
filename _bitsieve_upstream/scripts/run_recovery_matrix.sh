#!/usr/bin/env bash
# One button for the whole recovery study: calibrate -> verify -> train -> evaluate -> report.
#
#   bash scripts/run_recovery_matrix.sh                 # all visible GPUs, full matrix
#   DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash scripts/run_recovery_matrix.sh
#   PLAN_ONLY=1 bash scripts/run_recovery_matrix.sh     # print the plan, run nothing
#
# Nine jobs: the untrained baseline, plus four branches on each of two training
# tracks. Every job is evaluated over the same 18-cell grid, so in-domain
# recovery, budget generalization (trained at one budget, scored at another) and
# transfer to tasks never trained on all come out of one table.
#
# Jobs are dealt round-robin onto the GPUs and each GPU runs its queue
# SEQUENTIALLY -- one 7B model per card at a time (~21 GB train, ~17 GB eval).
# Everything resumes: finished adapters are not retrained and finished grid
# cells are not re-evaluated, so an interrupted run can simply be restarted.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# ---- knobs ---------------------------------------------------------------
OUT=${OUT:-runs/recovery}
RESULTS=${RESULTS:-results/recovery}
STEPS=${STEPS:-500}
LIMIT=${LIMIT:-60}                      # GSM8K / MATH-500 examples per cell
LIMIT_LONGBENCH=${LIMIT_LONGBENCH:-60}
LIMIT_NARRATIVEQA=${LIMIT_NARRATIVEQA:-60}
CONFIG=${CONFIG:-configs/proposed_a_k4v4_p5.yaml}
MATH_TRAIN_TOPK=${MATH_TRAIN_TOPK:-32}          # train the math track at k=32,
LB_TRAIN_PCT=${LB_TRAIN_PCT:-2.5}               # the LongBench track at 2.5%
# LongBench ships 200 examples per task and no train split, so the same 200 have
# to cover both. Training takes the front, evaluation starts past it.
LONGBENCH_PER_TASK=${LONGBENCH_PER_TASK:-100}
LONGBENCH_HELDOUT_PER_TASK=${LONGBENCH_HELDOUT_PER_TASK:-8}
EXAMPLE_OFFSET=${EXAMPLE_OFFSET:-120}   # skip the LongBench training slice at eval
BRANCHES=${BRANCHES:-"sft gkd sft_noise gkd_noise"}
# Selection in the training loop belongs to branch E by the study design ("no
# selection in the loop" for D), so each branch uses its own default. Set
# STUDENT_SELECT=on to put every branch under the budget instead.
STUDENT_SELECT=${STUDENT_SELECT:-auto}
TRACKS=${TRACKS:-"math longbench"}
# Branch C with an exact student prefix provably cannot learn (JSD is identically
# zero -- docs/recovery_training.md). Default to the variant that can; set
# GKD_STUDENT_CACHE=exact to reproduce the null cell on purpose.
GKD_STUDENT_CACHE=${GKD_STUDENT_CACHE:-quant}

if [ -z "${DEVICES:-}" ]; then
  n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)
  [ "$n" -lt 1 ] && n=1
  DEVICES=$(seq -s, -f 'cuda:%g' 0 $((n - 1)))
fi
IFS=',' read -ra DEV <<< "$DEVICES"
NDEV=${#DEV[@]}

# ---- preflight -----------------------------------------------------------
python - <<'PY' || { echo "environment is not ready; activate the project env first (scripts/setup.sh)"; exit 1; }
import importlib, sys
for m in ("torch", "transformers", "peft", "bitsieve_fastdllm"):
    importlib.import_module(m)
import torch
if not torch.cuda.is_available():
    sys.exit("CUDA is not available")
PY

mkdir -p "$OUT" "$RESULTS" logs

# The 200 examples per LongBench task must cover training AND evaluation. Check
# the arithmetic against the real split size before spending any GPU time: a
# clash otherwise surfaces only at the first trained LongBench cell, hours in.
if [[ " $TRACKS " == *" longbench "* ]]; then
  python scripts/_check_longbench_split.py \
    "$LONGBENCH_PER_TASK" "$LONGBENCH_HELDOUT_PER_TASK" "$EXAMPLE_OFFSET" "$LIMIT_LONGBENCH" || exit 1
fi

label_for() { case "$1" in math) echo "M";; longbench) echo "L";; esac; }

plan_line() { printf '  %-16s %-10s %-12s %s\n' "$1" "$2" "$3" "$4"; }

echo "=== recovery matrix ==="
echo "devices           : $DEVICES  ($NDEV queue(s), one model per card at a time)"
echo "config            : $CONFIG"
echo "train steps       : $STEPS"
echo "eval per cell     : math=$LIMIT longbench=$LIMIT_LONGBENCH narrativeqa=$LIMIT_NARRATIVEQA"
echo "longbench split   : train on the first $LONGBENCH_PER_TASK (+$LONGBENCH_HELDOUT_PER_TASK held out), eval from $EXAMPLE_OFFSET of 200"
echo "math track budget : topk=$MATH_TRAIN_TOPK   longbench track: ${LB_TRAIN_PCT}%"
echo "branch C variant  : --student-cache $GKD_STUDENT_CACHE"
echo "selection in loop : --student-select $STUDENT_SELECT (E only, by design)"
echo
echo "jobs (each evaluated over all 18 grid cells):"
plan_line "A_none" "-" "-" "no training; the training-free baseline"
for track in $TRACKS; do
  for b in $BRANCHES; do
    if [ "$track" = math ]; then budget="topk=$MATH_TRAIN_TOPK"; else budget="${LB_TRAIN_PCT}%"; fi
    plan_line "$(label_for "$track")_$b" "$track" "$budget" "train then grid"
  done
done
echo

if [ -n "${PLAN_ONLY:-}" ]; then echo "PLAN_ONLY set; nothing was run."; exit 0; fi

CALIB=results/training_noise_calibration.json
if [ ! -f "$CALIB" ]; then
  echo "== calibrating the KV-noise surrogate on ${DEV[0]}"
  python scripts/calibrate_training_noise.py --device "${DEV[0]}" --out "$CALIB"
fi

echo "== verifying the training forward against the real decoder (${DEV[0]})"
python tests/training_parity_e2e.py --device "${DEV[0]}" | tee logs/recovery_parity.log
grep -q "ALL PASS" logs/recovery_parity.log || { echo "parity check FAILED; not training"; exit 1; }

# ---- build one queue per GPU --------------------------------------------
for ((d = 0; d < NDEV; d++)); do : > "logs/.rq_$d"; done

emit_grid() {   # label adapter device queue
  local label="$1" adapter="$2" dev="$3" q="$4"
  {
    printf 'echo "== grid %s on %s"\n' "$label" "$dev"
    printf 'python scripts/run_recovery_grid.py run --branch %s --device %s \\\n' "$label" "$dev"
    printf '  --limit %s --limit-longbench %s --limit-narrativeqa %s \\\n' \
           "$LIMIT" "$LIMIT_LONGBENCH" "$LIMIT_NARRATIVEQA"
    printf '  --example-offset %s --out %s' "$EXAMPLE_OFFSET" "$RESULTS"
    [ -n "$adapter" ] && printf ' \\\n  --adapter %s' "$adapter"
    printf '\n'
  } >> "logs/.rq_$q"
}

emit_grid A_none "" "${DEV[0]}" 0

i=0
for track in $TRACKS; do
  for b in $BRANCHES; do
    label="$(label_for "$track")_$b"
    i=$((i + 1)); q=$((i % NDEV)); dev=${DEV[$q]}
    adapter="$OUT/$label/adapter"
    if [ "$track" = math ]; then
      data_args="--dataset math --train-topk $MATH_TRAIN_TOPK"
    else
      data_args="--dataset longbench --train-topk-percent $LB_TRAIN_PCT --longbench-per-task $LONGBENCH_PER_TASK --longbench-heldout-per-task $LONGBENCH_HELDOUT_PER_TASK"
    fi
    extra=""
    [ "$b" = gkd ] && extra="--student-cache $GKD_STUDENT_CACHE"
    {
      printf 'if [ -f %s/adapter_model.safetensors ]; then\n' "$adapter"
      printf '  echo "== skip training %s (adapter exists)"\n' "$label"
      printf 'else\n'
      printf '  echo "== train %s on %s"\n' "$label" "$dev"
      printf '  python scripts/train_recovery.py --branch %s --config %s \\\n' "$b" "$CONFIG"
      printf '    --out %s/%s --device %s --steps %s --student-select %s %s %s\n' \
             "$OUT" "$label" "$dev" "$STEPS" "$STUDENT_SELECT" "$data_args" "$extra"
      printf 'fi\n'
    } >> "logs/.rq_$q"
    emit_grid "$label" "$adapter" "$dev" "$q"
  done
done

# ---- run -----------------------------------------------------------------
pids=()
for ((d = 0; d < NDEV; d++)); do
  if [ -s "logs/.rq_$d" ]; then
    bash "logs/.rq_$d" > "logs/recovery_gpu${d}.log" 2>&1 &
    pids+=($!)
    echo "== ${DEV[$d]} queue started (pid ${pids[-1]}, log logs/recovery_gpu${d}.log)"
  fi
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo
python scripts/run_recovery_grid.py report --out "$RESULTS" | tee "$RESULTS/report.txt"
echo
echo "table -> $RESULTS/report.txt"
[ "$fail" -eq 0 ] || { echo "== at least one queue failed; see logs/recovery_gpu*.log"; exit 1; }

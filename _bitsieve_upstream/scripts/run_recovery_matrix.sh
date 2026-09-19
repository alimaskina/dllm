#!/usr/bin/env bash
# Recovery matrix: calibrate -> parity -> train B..E -> evaluate A..E.
#
#   DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash scripts/run_recovery_matrix.sh
#
# Jobs are dealt round-robin onto the given GPUs and each GPU runs its own queue
# SEQUENTIALLY -- one 7B model per card at a time (~18 GB train, ~16 GB eval).
# Fewer GPUs means a longer queue, never two models on one card.
set -euo pipefail

cd "$(dirname "$0")/.."
DEVICES=${DEVICES:-cuda:0}
CONFIG=${CONFIG:-configs/proposed_a_k4v4_p5.yaml}
OUT=${OUT:-runs/recovery}
RESULTS=${RESULTS:-results/recovery}
STEPS=${STEPS:-500}
LIMIT=${LIMIT:-60}
BENCHMARKS=${BENCHMARKS:-"gsm8k math500 2wikimqa hotpotqa musique narrativeqa"}
# Branch C as specified has no gradient (docs/recovery_training.md); set
# GKD_STUDENT_CACHE=quant for the non-degenerate variant.
GKD_STUDENT_CACHE=${GKD_STUDENT_CACHE:-auto}

IFS=',' read -ra DEV <<< "$DEVICES"
NDEV=${#DEV[@]}
mkdir -p "$OUT" "$RESULTS" logs
export PYTHONPATH="${PYTHONPATH:-}:$PWD/src"

CALIB=results/training_noise_calibration.json
if [ ! -f "$CALIB" ]; then
  echo "== calibrating the noise surrogate on ${DEV[0]}"
  python scripts/calibrate_training_noise.py --device "${DEV[0]}" --out "$CALIB"
fi

echo "== training/inference parity on ${DEV[0]}"
python tests/training_parity_e2e.py --device "${DEV[0]}" | tee logs/recovery_parity.log

BRANCHES=(sft gkd sft_noise gkd_noise)
LABELS=(B_sft C_gkd D_sft_noise E_gkd_noise)

emit_eval() {  # label adapter device queue
  local label="$1" adapter="$2" dev="$3" q="$4" b
  for b in $BENCHMARKS; do
    {
      echo "echo \"== eval $label / $b on $dev\""
      printf 'python -m bitsieve_fastdllm.eval.quality --config %s --benchmark %s \\\n' "$CONFIG" "$b"
      printf '  --device %s --limit %s --output %s/%s_%s.jsonl' "$dev" "$LIMIT" "$RESULTS" "$label" "$b"
      [ -n "$adapter" ] && printf ' \\\n  --adapter %s' "$adapter"
      printf '\n'
    } >> "logs/.rq_$q"
  done
}

for ((d = 0; d < NDEV; d++)); do : > "logs/.rq_$d"; done

# Branch A is training-free: eval only.
emit_eval A_none "" "${DEV[0]}" 0

for i in "${!BRANCHES[@]}"; do
  b=${BRANCHES[$i]}; label=${LABELS[$i]}
  q=$(( (i + 1) % NDEV )); dev=${DEV[$q]}
  extra=""
  if [ "$b" = "gkd" ] && [ "$GKD_STUDENT_CACHE" != "auto" ]; then
    extra="--student-cache $GKD_STUDENT_CACHE"
  fi
  {
    echo "echo \"== train $label on $dev\""
    printf 'python scripts/train_recovery.py --branch %s --config %s \\\n' "$b" "$CONFIG"
    printf '  --out %s/%s --device %s --steps %s %s\n' "$OUT" "$label" "$dev" "$STEPS" "$extra"
  } >> "logs/.rq_$q"
  emit_eval "$label" "$OUT/$label/adapter" "$dev" "$q"
done

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

echo "== results:"
ls -1 "$RESULTS"/*.jsonl 2>/dev/null || echo "  (none)"
[ "$fail" -eq 0 ] || { echo "== a queue failed; see logs/recovery_gpu*.log"; exit 1; }

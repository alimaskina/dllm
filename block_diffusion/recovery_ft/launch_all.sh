#!/bin/bash
# Full recovery matrix: calibrate -> parity -> train B..E -> evaluate A..E.
#
#   DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash launch_all.sh
#
# Jobs are dealt round-robin onto the given GPUs and each GPU runs its own
# queue SEQUENTIALLY — one 7B model per card at a time (~17 GB train,
# ~16 GB eval). Fewer GPUs than jobs just means a longer queue, never two
# models on one card.
set -euo pipefail

cd "$(dirname "$0")"
DEVICES=${DEVICES:-cuda:0}
OUT=${OUT:-runs}
RESULTS=${RESULTS:-results}
STEPS=${STEPS:-500}
NUM_EVAL=${NUM_EVAL:-60}
MAX_LEN=${MAX_LEN:-1280}
CONDA_ENV=${CONDA_ENV:-fast_dllm}
# Branch C as literally specified has no gradient (see README); set
# GKD_STUDENT_CACHE=quant for the non-degenerate variant.
GKD_STUDENT_CACHE=${GKD_STUDENT_CACHE:-auto}

IFS=',' read -ra DEV <<< "$DEVICES"
NDEV=${#DEV[@]}
mkdir -p "$OUT" "$RESULTS" logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

d0=${DEV[0]}
if [ ! -f noise_calibration.json ]; then
  echo "== calibrating the noise surrogate on $d0"
  python calibrate_noise.py --device "$d0" --num-prompts 40 --bits 4 2
fi

echo "== parity check on $d0"
python test_parity.py --device "$d0" | tee logs/parity.log

BRANCHES=(sft gkd sft_noise gkd_noise)
LABELS=(B_sft C_gkd D_sft_noise E_gkd_noise)

# One queue file per GPU; each holds the shell commands that card must run.
for ((d = 0; d < NDEV; d++)); do : > "logs/.queue_$d"; done

# Branch A is training-free: eval only.
cat >> logs/.queue_0 <<EOF
echo "== eval A_none"
python eval_recovery.py --branch A_none --num-examples $NUM_EVAL \\
  --device ${DEV[0]} --out $RESULTS/A_none
EOF

for i in "${!BRANCHES[@]}"; do
  b=${BRANCHES[$i]}; label=${LABELS[$i]}
  q=$(( (i + 1) % NDEV ))
  dev=${DEV[$q]}
  extra=""
  if [ "$b" = "gkd" ] && [ "$GKD_STUDENT_CACHE" != "auto" ]; then
    extra="--student-cache $GKD_STUDENT_CACHE"
  fi
  cat >> "logs/.queue_$q" <<EOF
echo "== train $label on $dev"
python train.py --branch $b --out $OUT/$label --device $dev \\
  --steps $STEPS --max-len $MAX_LEN $extra
echo "== eval $label on $dev"
python eval_recovery.py --branch $label --adapter $OUT/$label/adapter \\
  --num-examples $NUM_EVAL --device $dev --out $RESULTS/$label
EOF
done

pids=()
for ((d = 0; d < NDEV; d++)); do
  if [ -s "logs/.queue_$d" ]; then
    bash "logs/.queue_$d" > "logs/gpu${d}.log" 2>&1 &
    pids+=($!)
    echo "== ${DEV[$d]} queue started (pid ${pids[-1]}, log logs/gpu${d}.log)"
  fi
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

echo "== reports:"
ls -1 "$RESULTS"/*/report.md 2>/dev/null || echo "  (none)"
[ "$fail" -eq 0 ] || { echo "== at least one queue failed; see logs/gpu*.log"; exit 1; }

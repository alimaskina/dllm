#!/bin/bash
# Full recovery matrix: calibrate -> parity -> train B..E -> evaluate A..E.
#
#   DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 bash launch_all.sh
#
# One GPU per branch; the eval grid for each branch runs on the GPU that
# trained it. Branch A needs no training and is evaluated first.
set -euo pipefail

cd "$(dirname "$0")"
DEVICES=${DEVICES:-cuda:0}
OUT=${OUT:-runs}
RESULTS=${RESULTS:-results}
STEPS=${STEPS:-500}
NUM_EVAL=${NUM_EVAL:-60}
CONDA_ENV=${CONDA_ENV:-fast_dllm}
# Branch C as literally specified has no gradient (see README); set
# GKD_STUDENT_CACHE=quant for the non-degenerate variant.
GKD_STUDENT_CACHE=${GKD_STUDENT_CACHE:-auto}

IFS=',' read -ra DEV <<< "$DEVICES"
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

# Branch A: training-free baseline.
echo "== eval A_none on $d0"
python eval_recovery.py --branch A_none --num-examples "$NUM_EVAL" \
  --device "$d0" --out "$RESULTS/A_none" > logs/eval_A_none.log 2>&1 &

pids=()
for i in "${!BRANCHES[@]}"; do
  b=${BRANCHES[$i]}; label=${LABELS[$i]}
  dev=${DEV[$(( (i + 1) % ${#DEV[@]} ))]}
  extra=""
  if [ "$b" = "gkd" ] && [ "$GKD_STUDENT_CACHE" != "auto" ]; then
    extra="--student-cache $GKD_STUDENT_CACHE"
  fi
  (
    echo "== train $label on $dev"
    python train.py --branch "$b" --out "$OUT/$label" --device "$dev" \
      --steps "$STEPS" $extra
    echo "== eval $label on $dev"
    python eval_recovery.py --branch "$label" --adapter "$OUT/$label/adapter" \
      --num-examples "$NUM_EVAL" --device "$dev" --out "$RESULTS/$label"
  ) > "logs/${label}.log" 2>&1 &
  pids+=($!)
done

wait "${pids[@]}"
wait
echo "== done; reports:"
ls -1 "$RESULTS"/*/report.md

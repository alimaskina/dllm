#!/usr/bin/env bash
# MATH500 top-K prune: 2 variants per GPU (4 and 6)
set -euo pipefail
cd "$(dirname "$0")"
OUT=../checkpoints/math500_topk_prune_n100
mkdir -p "$OUT"

run_one() {
  local gpu=$1
  local variants=$2
  local log="$OUT/gpu${gpu}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n fast_dllm python run_math500_topk_prune.py \
    --n 100 \
    --seed 1234 \
    --max-new-tokens 1024 \
    --device cuda:0 \
    --variants "${variants}" \
    --out-dir "${OUT}/gpu${gpu}" \
    > "${log}" 2>&1 &
  echo "GPU ${gpu}: ${variants} → ${log} (pid $!)"
  echo "$!"
}

PID4=$(run_one 4 "top128_fp,top256_kivi8")
PID6=$(run_one 6 "top512_kivi4,top1024_kivi2")

echo "Waiting for pids ${PID4} and ${PID6}..."
wait "${PID4}" "${PID6}"

echo "Merging summaries..."
python - <<'PY'
import json
from pathlib import Path

out = Path("../checkpoints/math500_topk_prune_n100")
merged = {"meta": None}
for gpu in ("gpu4", "gpu6"):
    p = out / gpu / "summary.json"
    if not p.exists():
        print(f"WARNING: missing {p}")
        continue
    data = json.loads(p.read_text())
    if merged["meta"] is None:
        merged["meta"] = data.get("meta", {})
        merged["meta"]["parallel_gpus"] = [4, 6]
    for k, v in data.items():
        if k == "meta":
            continue
        merged[k] = v

(out / "summary.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False))
print(f"Merged → {out / 'summary.json'}")
for k in merged:
    if k == "meta":
        continue
    print(f"  {k}: {merged[k]['accuracy']:.1%}")
PY

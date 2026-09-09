#!/usr/bin/env bash
# The button: 15 GSM8K + 15 LongBench (5 each of 2wikimqa/qmsum/repobench-p),
# all 4 variants (dense, sparse fp16 all, sparse fp16 middle, sparse K4/V4 all),
# quality + a separate coverage-diagnostic pass, then a summary table.
#
# Override anything via env vars, e.g.:
#   DEVICE=cuda:2 GSM8K_N=30 bash scripts/run_suite.sh
#   VARIANTS=dense,sparse_k4v4_all bash scripts/run_suite.sh   # just two variants
#
# Expect on the order of an hour on one A100 for the full default run (7
# variant/pass combinations x 30 examples, up to 2048 generated tokens on
# GSM8K); --smoke below finishes in a couple of minutes and is the thing to
# run first on a new machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Point this at whatever interpreter has the project's dependencies (the env
# `scripts/setup.sh` creates, conda-activated, or any other with a matching
# transformers). Defaults to whatever `python3` resolves to, which on a fresh
# shell is very often the WRONG environment - the preflight check below exists
# because that failure mode is a deep, cryptic traceback from inside
# transformers (a rope-init KeyError on this specific model), not a clear
# "wrong env" message.
PYTHON="${PYTHON:-python3}"

echo "python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
"$PYTHON" - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("torch is not installed in this interpreter - activate the "
              "environment from `bash scripts/setup.sh`, or set PYTHON=/path/to/python")
try:
    import transformers
except ImportError:
    sys.exit("transformers is not installed in this interpreter - see scripts/setup.sh")
if transformers.__version__ != "4.53.1":
    sys.exit(
        f"transformers=={transformers.__version__} is active, but this model's "
        "remote code (Fast_dLLM_v2's RoPE init) needs exactly transformers==4.53.1 "
        "and fails with an unrelated-looking KeyError on anything newer. "
        "Run `bash scripts/setup.sh` or `pip install transformers==4.53.1`, "
        "or set PYTHON=/path/to/the/right/python."
    )
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False on this interpreter/machine")
print(f"torch {torch.__version__}, transformers {transformers.__version__}, "
      f"{torch.cuda.device_count()} GPU(s) visible - OK")
PY

DEVICE="${DEVICE:-cuda:0}"
MODEL="${MODEL:-Efficient-Large-Model/Fast_dLLM_v2_7B}"
MODEL_REVISION="${MODEL_REVISION:-0661abf5f9f0ee338970d091052a26c8efa51974}"
GSM8K_N="${GSM8K_N:-15}"
LONGBENCH_TASKS="${LONGBENCH_TASKS:-2wikimqa,qmsum,repobench-p}"
LONGBENCH_N="${LONGBENCH_N:-5}"
VARIANTS="${VARIANTS:-dense,sparse_fp16_all,sparse_fp16_middle,sparse_k4v4_all}"
TOPK_PCT="${TOPK_PCT:-5.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/suite_run_$(date +%Y%m%d_%H%M%S)}"

echo "=== BitSieve suite ==="
echo "device=$DEVICE  model=$MODEL@$MODEL_REVISION"
echo "gsm8k_n=$GSM8K_N  longbench_tasks=$LONGBENCH_TASKS  longbench_n=$LONGBENCH_N"
echo "variants=$VARIANTS  topk_pct=$TOPK_PCT"
echo "output_root=$OUTPUT_ROOT"
echo

"$PYTHON" scripts/run_suite.py \
  --device "$DEVICE" \
  --model "$MODEL" \
  --revision "$MODEL_REVISION" \
  --gsm8k-n "$GSM8K_N" \
  --longbench-tasks "$LONGBENCH_TASKS" \
  --longbench-n "$LONGBENCH_N" \
  --variants "$VARIANTS" \
  --topk-pct "$TOPK_PCT" \
  --output-root "$OUTPUT_ROOT" \
  "$@"

echo
echo "Results:  $OUTPUT_ROOT/*.jsonl"
echo "Summary:  $OUTPUT_ROOT/summary.json"

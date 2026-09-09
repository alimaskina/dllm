#!/usr/bin/env bash
# One-button reproducible quality check: two budget profiles over GSM8K,
# QMSum, and RepoBench-P, with 20 examples per benchmark.
# Dataset and model revisions are pinned so independent runs use identical
# prompts, references, and checkpoint weights. Override counts or variants via
# environment variables; each profile can be safely resumed independently.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DEVICE="${DEVICE:-cuda:0}"
export GSM8K_N="${GSM8K_N:-20}"
export LONGBENCH_TASKS="${LONGBENCH_TASKS:-qmsum,repobench-p}"
export LONGBENCH_N="${LONGBENCH_N:-20}"
export VARIANTS="${VARIANTS:-dense,sparse_fp16_all,sparse_fp16_middle,sparse_k4v4_all}"
export MODEL="${MODEL:-Efficient-Large-Model/Fast_dLLM_v2_7B}"
export MODEL_REVISION="${MODEL_REVISION:-0661abf5f9f0ee338970d091052a26c8efa51974}"
export GSM8K_REVISION="${GSM8K_REVISION:-740312add88f781978c0658806c59bc2815b9866}"
export LONG_BENCH_REVISION="${LONG_BENCH_REVISION:-5e628be450b7e67fb7ae6e201bd6d8f7056f7672}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/quality_repro_$(date +%Y%m%d_%H%M%S)}"
CLI_ARGS=("$@")

run_profile() {
    local name="$1"
    local long_budget="$2"
    local math_budget="$3"
    echo
    echo "=== quality profile: ${name} ==="
    TOPK_PCT="$long_budget" OUTPUT_ROOT="$OUTPUT_ROOT/$name" \
        bash scripts/run_suite.sh --skip-coverage --math-topk "$math_budget" "${CLI_ARGS[@]}"
}

run_profile long5_math64 5.0 64
run_profile long20_math128 20.0 128

echo
echo "All quality profiles completed."
echo "Results: $OUTPUT_ROOT/long5_math64/summary.json"
echo "         $OUTPUT_ROOT/long20_math128/summary.json"

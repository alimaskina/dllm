#!/usr/bin/env bash
# Screen every English LongBench task for whether its metric puts the four
# variants in the order the method predicts:
#
#   dense >= sparse_fp16_all ~ sparse_k4v4_all > sparse_fp16_middle
#
# Tasks whose metric is noisy enough to let a weaker selector beat dense are the
# ones to keep out of a headline table - an approximation "beating" the exact
# ceiling reads as a broken experiment to a reviewer, even when it is just
# variance on a fuzzy metric at a modest n.
#
# Runs ONE TASK PER INVOCATION of run_suite.py (rather than one invocation over
# all tasks) so that each task's four-variant comparison lands complete: the
# suite loops variant-major, so a single big invocation would leave every task
# half-measured until the very end. Costs one model load per task (~1 min).
#
# Quality pass only by default: the screen is about score ordering, not selector
# fidelity, and the coverage pass roughly doubles the runtime. Set COVERAGE=1 to
# turn it on - worth it when the budget is tight enough that coverage mass is
# expected to actually separate the selectors.
#
# EXTRA_ARGS is passed through to run_suite.py, e.g.
#   EXTRA_ARGS="--longbench-topk 128" to swap the percent budget for a fixed one.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda:0}"
N="${N:-20}"
TOPK_PCT="${TOPK_PCT:-5.0}"
VARIANTS="${VARIANTS:-dense,sparse_fp16_all,sparse_fp16_middle,sparse_k4v4_all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/task_selection_$(date +%Y%m%d_%H%M%S)}"
COVERAGE="${COVERAGE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ "$COVERAGE" == "1" ]]; then COVERAGE_FLAG=(); else COVERAGE_FLAG=(--skip-coverage); fi
# Unquoted on purpose: EXTRA_ARGS is a caller-supplied flag list, not one word.
# shellcheck disable=SC2206
EXTRA_ARGV=($EXTRA_ARGS)
# Every English task wired into LONG_BENCH_CONFIGS.
TASKS="${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"

echo "=== LongBench task selection screen ==="
echo "device=$DEVICE  n_per_task=$N  topk_pct=$TOPK_PCT  coverage=$COVERAGE"
[[ -n "$EXTRA_ARGS" ]] && echo "extra_args=$EXTRA_ARGS"
echo "variants=$VARIANTS"
echo "output_root=$OUTPUT_ROOT"
echo "tasks=$TASKS"
echo

IFS=',' read -ra TASK_LIST <<< "$TASKS"
for task in "${TASK_LIST[@]}"; do
    echo
    echo "######## $task ########"
    # --gsm8k-n 0 keeps GSM8K (not a LongBench task) out of this screen.
    "$PYTHON" scripts/run_suite.py \
        --device "$DEVICE" \
        --gsm8k-n 0 \
        --longbench-tasks "$task" \
        --longbench-n "$N" \
        --variants "$VARIANTS" \
        --topk-pct "$TOPK_PCT" \
        "${COVERAGE_FLAG[@]}" \
        ${EXTRA_ARGV[@]+"${EXTRA_ARGV[@]}"} \
        --output-root "$OUTPUT_ROOT/$task" \
        || echo "!! $task failed, continuing"
done

echo
echo "All tasks done. Per-task results: $OUTPUT_ROOT/<task>/summary.json"
echo "Rank them with: $PYTHON scripts/rank_task_orderings.py $OUTPUT_ROOT"

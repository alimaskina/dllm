#!/usr/bin/env bash
# One command to finish the official 200-example evaluation on the three tasks
# in the headline table (gov_report, qasper, trec).
#
#     bash scripts/run_official_200.sh
#
# Everything else is automatic: it picks the GPUs, splits the work across them,
# resumes from the 60 examples already committed, waits, prints the report and
# packs what needs to be sent back into a single tarball.
#
# Knobs, none of them required:
#   GPUS=0,1     which GPUs to use (default: every one nvidia-smi reports)
#   PYTHON=...   interpreter (default: python)
#   N=200        example count; leave it alone unless you know why
#   DETACH=0     stay in the foreground instead of detaching (default: detach)
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
N="${N:-200}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/task_selection}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/official_200}"
TASKS_LONG="gov_report"          # ~4.4 h: 512 generated tokens per example
TASKS_SHORT="qasper,trec"        # ~0.9 h together
ALL_TASKS="gov_report qasper trec"

mkdir -p "$LOG_DIR"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight --
say "Preflight"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found; this needs a CUDA machine."

# Checked the way run_suite.py actually imports: it puts ./src on sys.path itself,
# so the package does NOT have to be pip-installed into the environment. Testing
# a bare `import bitsieve_fastdllm` here would reject working setups.
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'PREFLIGHT' || die \
"'$PYTHON' cannot run this project (see the error above).
   Activate the environment first (conda activate bitsieve), or point PYTHON at it:
       PYTHON=/path/to/env/bin/python bash scripts/run_official_200.sh"
import sys
import torch
import bitsieve_fastdllm  # noqa: F401
if not torch.cuda.is_available():
    sys.exit("torch cannot see a GPU in this environment")
print(f"  torch {torch.__version__}, {torch.cuda.device_count()} GPU(s) visible")
PREFLIGHT

# The 60-example results on this branch are valid input, and the run is supposed
# to continue from them. If they are missing, the run would silently recompute
# ~6 GPU-hours of work that is already on disk - so stop instead.
missing=""
for t in $ALL_TASKS; do
    rows=0
    if [[ -f "$OUTPUT_ROOT/$t/dense.jsonl" ]]; then
        rows=$(grep -c '"pass": "quality"' "$OUTPUT_ROOT/$t/dense.jsonl" || true)
    fi
    printf '  %-12s %s existing examples\n' "$t" "$rows"
    (( rows >= 60 )) || missing="$missing $t"
done
[[ -z "$missing" ]] || die \
"no committed results found for:$missing
   Expected at least 60 examples per task under $OUTPUT_ROOT/.
   You are probably on the wrong branch - this needs 'bitsieve-fastdllm-fixes':
       git checkout bitsieve-fastdllm-fixes && git pull
   Running without them would recompute about 6 GPU-hours that are already done."

mapfile -t GPU_LIST < <(
    if [[ -n "${GPUS:-}" ]]; then tr ',' '\n' <<<"$GPUS" | sed '/^$/d'
    else nvidia-smi --query-gpu=index --format=csv,noheader; fi
)
(( ${#GPU_LIST[@]} > 0 )) || die "no GPUs found. Set GPUS=0 to force one."
printf '  GPUs: %s\n' "${GPU_LIST[*]}"

for g in "${GPU_LIST[@]}"; do
    free=$(nvidia-smi --id="$g" --query-gpu=memory.free --format=csv,noheader,nounits)
    printf '  GPU %s: %s MiB free\n' "$g" "$free"
    (( free >= 40000 )) || printf '    WARNING: under 40 GB free; a 32k-token prompt may OOM.\n'
done

# --------------------------------------------------------------- detachment --
# Preflight has passed, so anything that was going to fail fast already has and
# the operator saw it. The remaining ~5 hours should not depend on an ssh
# session staying up, so re-exec detached unless asked not to.
if [[ -z "${_OFFICIAL200_DETACHED:-}" && "${DETACH:-1}" != 0 ]]; then
    MASTER_LOG="$LOG_DIR/${STAMP}_run.log"
    _OFFICIAL200_DETACHED=1 STAMP="$STAMP" \
        setsid nohup bash "$0" >"$MASTER_LOG" 2>&1 </dev/null &
    disown || true
    cat <<EOF

Started in the background; this terminal is free and you can disconnect.

  Watch:     tail -f $MASTER_LOG
  Finished:  the log ends with "Done." and names a .tar.gz to send back
  Stop:      pkill -f run_official_200.sh && pkill -f run_suite.py

EOF
    exit 0
fi

# ------------------------------------------------------------------- launch --
run_group() {   # run_group <gpu> <tasks-csv> <logfile>
    PYTHON="$PYTHON" DEVICE="cuda:$1" N="$N" TASKS="$2" OUTPUT_ROOT="$OUTPUT_ROOT" \
        bash scripts/run_task_selection.sh >"$3" 2>&1
}

pids=()
logs=()
if (( ${#GPU_LIST[@]} >= 2 )); then
    say "Running on 2 GPUs (~4.5 h wall, set by gov_report)"
    l1="$LOG_DIR/${STAMP}_gov_report.log";   run_group "${GPU_LIST[0]}" "$TASKS_LONG"  "$l1" & pids+=($!); logs+=("$l1")
    l2="$LOG_DIR/${STAMP}_qasper_trec.log";  run_group "${GPU_LIST[1]}" "$TASKS_SHORT" "$l2" & pids+=($!); logs+=("$l2")
else
    say "Running on 1 GPU (~5.5 h, sequential)"
    l1="$LOG_DIR/${STAMP}_all.log"
    run_group "${GPU_LIST[0]}" "gov_report,qasper,trec" "$l1" & pids+=($!); logs+=("$l1")
fi

printf '  logs: %s\n' "${logs[@]}"
printf '  follow with:  tail -f %s\n' "${logs[0]}"

# A resume that is not resuming is the one failure worth catching early - it
# costs hours rather than producing a wrong answer. Give the runs a moment to
# print their first progress line, then check it.
sleep 90
for l in "${logs[@]}"; do
    if grep -q "running $N/$N" "$l" 2>/dev/null; then
        for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
        pkill -P $$ -f run_suite.py 2>/dev/null || true
        die "the run started from zero instead of resuming (see $l).
   The committed results were not picked up. Stopped rather than spend the hours."
    fi
done
grep -h "extending the existing run" "${logs[@]}" 2>/dev/null | sed 's/^/  /' || true

say "Working. Safe to disconnect if this is under tmux/nohup; otherwise leave it open."
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
(( status == 0 )) || die "a run group exited non-zero; see the logs above. Nothing was packed."

# run_task_selection.sh deliberately keeps going when one task fails, and exits 0
# regardless - so a zero exit status does NOT mean every task finished. Without
# this check the script would cheerfully pack a bundle with a whole task missing.
if grep -q "failed, continuing" "${logs[@]}" 2>/dev/null; then
    printf '\n'
    grep -h '^!! \|belongs to a different run\|Traceback' "${logs[@]}" 2>/dev/null | sed 's/^/  /'
    die "at least one task failed (above). Fix it and re-run the same command:
   finished examples are kept, so only the missing work is redone. Nothing was packed."
fi

# Every arm of every task must have reached N, or the comparison is between
# different sample sizes and the report will read 'incomplete'.
short=""
for t in $ALL_TASKS; do
    for v in dense sparse_fp16_all sparse_k4v4_all sparse_fp16_middle; do
        f="$OUTPUT_ROOT/$t/$v.jsonl"
        rows=0
        [[ -f "$f" ]] && rows=$(grep -c '"pass": "quality"' "$f" || true)
        (( rows >= N )) || short="$short
  $t/$v: $rows of $N"
    done
done
[[ -z "$short" ]] || die "some arms did not reach $N examples:$short
   Re-run the same command; finished examples are kept. Nothing was packed." 

# ------------------------------------------------------------------- report --
say "Report"
REPORT="$LOG_DIR/${STAMP}_report.txt"
"$PYTHON" scripts/rank_task_orderings.py "$OUTPUT_ROOT" | tee "$REPORT"

say "Packing"
BUNDLE="$ROOT/official_200_${STAMP}.tar.gz"
tar czf "$BUNDLE" \
    -C "$ROOT" "$OUTPUT_ROOT/gov_report" "$OUTPUT_ROOT/qasper" "$OUTPUT_ROOT/trec" \
    --transform "s|^logs/official_200|logs|" \
    "$(realpath --relative-to="$ROOT" "$REPORT")" \
    $(for l in "${logs[@]}"; do realpath --relative-to="$ROOT" "$l"; done)

cat <<EOF

Done.

  Send back:  $BUNDLE
  Report:     $REPORT

Check before sending: gov_report, qasper and trec should all say verdict 'ok',
and each dense score should sit in or near its published band printed beside it.
If a dense score is far outside, say so instead of treating the run as finished -
that means the protocol broke, not that the model is unusual.
EOF

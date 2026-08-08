#!/usr/bin/env bash
# Watchdog: wait for a GPU with enough free memory, run Quant-dLLM, restart on failure.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOGDIR="$ROOT/log"
mkdir -p "$LOGDIR"
WATCHLOG="$LOGDIR/watchdog.log"
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
SEQLEN="${SEQLEN:-2048}"
NSAMPLES="${NSAMPLES:-4}"
PYTHON="${PYTHON:-/home/alimaskina/miniconda3/envs/quant/bin/python}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WATCHLOG"; }

pick_gpu() {
  "$PYTHON" - <<PY
import subprocess
min_free=int("${MIN_FREE_MIB}")
# Skip cards that repeatedly SIGKILL co-tenants under memory pressure.
skip={int(x) for x in "${SKIP_GPUS:-0}".split(",") if x.strip()!=""}
out=subprocess.check_output(
    ["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],
    text=True,
)
best=None
for line in out.strip().splitlines():
    idx, free = line.split(",")
    idx, free = int(idx.strip()), int(free.strip())
    if idx in skip:
        continue
    if free >= min_free and (best is None or free > best[0]):
        best=(free, idx)
print(best[1] if best else "")
PY
}

attempt=0
while true; do
  attempt=$((attempt+1))
  GPU="$(pick_gpu)"
  if [[ -z "$GPU" ]]; then
    log "attempt $attempt: no GPU with >= ${MIN_FREE_MIB}MiB free; sleeping 60s"
    sleep 60
    continue
  fi
  log "attempt $attempt: launching on cuda:$GPU seqlen=$SEQLEN nsamples=$NSAMPLES slim=${SLIM:-0}"
  : > "$LOGDIR/run_active_stdout.log"
  DEVICE="cuda:$GPU" SEQLEN="$SEQLEN" NSAMPLES="$NSAMPLES" SLIM="${SLIM:-0}" ABMP_RATIO="${ABMP_RATIO:-0.05}" \
    "$ROOT/launch_llada_base.sh" >>"$WATCHLOG" 2>&1
  rc=$?
  log "attempt $attempt: exited rc=$rc"
  # Success if final checkpoint directory exists
  if ls -d "$ROOT"/output/LLaDA-8B-Base_c4_arb-rc_*.pt >/dev/null 2>&1; then
    log "checkpoint found; done"
    exit 0
  fi
  # Also treat clean completion with save as success via log marker
  if grep -q "Quantization time:" "$LOGDIR/run_active_stdout.log" 2>/dev/null; then
    log "quantization finished; done"
    exit 0
  fi
  log "no checkpoint yet; retrying in 30s"
  sleep 30
done

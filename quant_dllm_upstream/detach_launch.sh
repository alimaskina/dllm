#!/usr/bin/env bash
# Fully detach Quant-dLLM so Cursor/tool shells cannot reap it.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p log
: > log/run_active_stdout.log
: > log/launch_detach.log
rm -f log/run_active.pid
# Close inherited fds, new session, background
setsid env DEVICE="${DEVICE:-cuda:0}" SEQLEN="${SEQLEN:-2048}" NSAMPLES="${NSAMPLES:-4}" \
  SLIM="${SLIM:-0}" DISABLE_GPTQ="${DISABLE_GPTQ:-1}" ABMP_RATIO="${ABMP_RATIO:-0.05}" \
  "$ROOT/launch_llada_base.sh" </dev/null >>"$ROOT/log/launch_detach.log" 2>&1 &
echo $! > log/detach_wrapper.pid
echo "started wrapper pid=$(cat log/detach_wrapper.pid)"

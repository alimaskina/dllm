#!/bin/bash
# FP16 Sudoku n256, 2 GPU, full traces. Resume-safe (no FRESH by default).
set -euo pipefail
cd "$(dirname "$0")"
export GPUS_2="${GPUS_2:-2,3}"
export MAIN_PORT="${MAIN_PORT:-29500}"
exec bash run_sudoku.sh fp16 fast 2gpu n256

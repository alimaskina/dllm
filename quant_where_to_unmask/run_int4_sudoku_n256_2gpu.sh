#!/bin/bash
# INT4 Sudoku n256, 2 GPU, full traces. Resume-safe (no FRESH by default).
set -euo pipefail
cd "$(dirname "$0")"
export GPUS_2="${GPUS_2:-4,5}"
export MAIN_PORT="${MAIN_PORT:-29501}"
exec bash run_sudoku.sh int4 fast 2gpu n256

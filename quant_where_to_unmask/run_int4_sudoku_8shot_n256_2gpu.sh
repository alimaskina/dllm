#!/bin/bash
# INT4 Sudoku 8-shot n256, 2 GPU, full traces. Resume-safe.
set -euo pipefail
cd "$(dirname "$0")"
export GPUS_2="${GPUS_2:-4,5}"
export MAIN_PORT="${MAIN_PORT:-29501}"
exec bash run_sudoku.sh int4 8shot 2gpu n256

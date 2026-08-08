#!/bin/bash
# FP16 4x4 Sudoku 8-shot n256, 2 GPU. Resume-safe.
set -euo pipefail
cd "$(dirname "$0")"
export GPUS_2="${GPUS_2:-2,3}"
export MAIN_PORT="${MAIN_PORT:-29500}"
exec bash run_sudoku.sh fp16 4x4 2gpu n256

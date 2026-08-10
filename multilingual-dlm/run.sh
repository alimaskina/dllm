#!/usr/bin/env bash
# One-command entry point: setup (if needed) + smoke test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d "$ROOT/.venv" ]]; then
  echo "[run] no .venv found — running setup first"
  bash "$ROOT/setup.sh"
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

python - <<'PY'
import sys
import torch

print(f"[run] python {sys.version.split()[0]}")
print(f"[run] torch {torch.__version__}, cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available. Re-run setup with a CUDA torch wheel:\n"
        "  TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash setup.sh"
    )

from common import pick_cuda_device

device = pick_cuda_device()
free_gb = torch.cuda.mem_get_info(int(device.split(':')[1]))[0] / (1024 ** 3)
print(f"[run] using {device} ({free_gb:.1f} GB free)")
PY

echo "[run] smoke_test.py"
python smoke_test.py "$@"

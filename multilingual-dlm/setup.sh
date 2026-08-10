#!/usr/bin/env bash
# One-time environment setup for multilingual-dlm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "[setup] python: $($PYTHON --version)"

if [[ ! -d "$VENV" ]]; then
  echo "[setup] creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -U pip wheel

if ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "[setup] installing CUDA torch from $TORCH_INDEX"
  python -m pip install "torch>=2.5.0" --index-url "$TORCH_INDEX"
else
  echo "[setup] torch with CUDA already available: $(python -c 'import torch; print(torch.__version__)')"
fi

echo "[setup] installing Python dependencies"
python -m pip install -r requirements.txt

echo
echo "[setup] done."
echo "  activate: source .venv/bin/activate"
echo "  run:      bash run.sh"

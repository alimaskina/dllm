#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-bitsieve}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
PYTORCH_CUDA="${PYTORCH_CUDA:-auto}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
CACHE_ROOT="${BITSIEVE_CACHE_ROOT:-${HOME}/.cache/bitsieve-fastdllm}"
REQUIRE_GPU=0

while (($#)); do
    case "$1" in
        --env) ENV_NAME="${2:?}"; shift 2 ;;
        --python) PYTHON_VERSION="${2:?}"; shift 2 ;;
        --torch) TORCH_VERSION="${2:?}"; shift 2 ;;
        --pytorch-cuda) PYTORCH_CUDA="${2:?}"; shift 2 ;;
        --torch-index-url) TORCH_INDEX_URL="${2:?}"; shift 2 ;;
        --cache-root) CACHE_ROOT="${2:?}"; shift 2 ;;
        --require-gpu) REQUIRE_GPU=1; shift ;;
        -h|--help)
            printf '%s\n' 'Usage: bash scripts/setup.sh [--env NAME] [--python VERSION] [--torch VERSION] [--pytorch-cuda auto|cu118|cu126|cu128] [--torch-index-url URL] [--cache-root PATH] [--require-gpu]'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ "$(uname -s)" == Linux ]] || { echo 'Linux is required.' >&2; exit 1; }
CONDA_BIN="${CONDA_EXE:-}"
if [[ ! -x "$CONDA_BIN" ]]; then
    CONDA_BIN="$(type -P conda || true)"
fi
if [[ ! -x "$CONDA_BIN" ]]; then
    for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda; do
        if [[ -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
    done
fi
[[ -x "$CONDA_BIN" ]] || { echo 'Install Conda first or set CONDA_EXE.' >&2; exit 1; }

version_ge() {
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]
}

if [[ "$PYTORCH_CUDA" == auto ]]; then
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n 1 || true)"
    if [[ -n "$DRIVER_CUDA" ]] && version_ge "$DRIVER_CUDA" 12.8; then
        PYTORCH_CUDA=cu128
    elif [[ -n "$DRIVER_CUDA" ]] && version_ge "$DRIVER_CUDA" 12.6; then
        PYTORCH_CUDA=cu126
    elif [[ -n "$DRIVER_CUDA" ]]; then
        PYTORCH_CUDA=cu118
    else
        PYTORCH_CUDA=cu126
        echo 'Driver not visible; using cu126. Override --pytorch-cuda for the target GPU.' >&2
    fi
fi
case "$PYTORCH_CUDA" in
    cu118|cu126|cu128) ;;
    *) [[ -n "$TORCH_INDEX_URL" ]] || { echo 'Unsupported wheel family.' >&2; exit 2; } ;;
esac
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$PYTORCH_CUDA}"

if ! "$CONDA_BIN" run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
    "$CONDA_BIN" create -y -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" pip 'setuptools>=69' wheel
fi
run_env() { "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" "$@"; }
run_env python -m pip install --upgrade pip 'setuptools>=69' wheel
run_env python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
run_env python -m pip install 'numpy>=1.24,<2' -e "$ROOT"
run_env python -m pip check

PREFIX="$(run_env python -c 'import sys; print(sys.prefix)' | tail -n 1)"
mkdir -p "$PREFIX/etc/conda/activate.d" "$CACHE_ROOT"
{
    printf 'if [[ -z "${BITSIEVE_CACHE_ROOT:-}" ]]; then export BITSIEVE_CACHE_ROOT=%q; fi\n' "$CACHE_ROOT"
    cat <<'EOF'
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${BITSIEVE_CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${BITSIEVE_CACHE_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${BITSIEVE_CACHE_ROOT}/triton}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}"
EOF
} > "$PREFIX/etc/conda/activate.d/bitsieve-fastdllm.sh"

REQUIRE_GPU="$REQUIRE_GPU" run_env python - <<'PY'
import json
import os
import torch
from bitsieve_fastdllm.kernels.cuda_fast import install_info
print(json.dumps(install_info(), indent=2))
if os.environ.get('REQUIRE_GPU') == '1':
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not visible in this environment.')
    if not install_info()['eligible']:
        raise SystemExit('The optimized path requires a supported CUDA/Triton build and SM80+.')
PY
printf '\nReady: conda activate %s\n' "$ENV_NAME"

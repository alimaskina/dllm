#!/usr/bin/env bash
# One-shot setup for dllm research environments on a fresh GPU server.
#
# Usage:
#   bash scripts/setup_environment.sh              # both conda envs
#   bash scripts/setup_environment.sh fast_dllm    # Fast-dLLM only
#   bash scripts/setup_environment.sh llada_quant  # LLaDA only
#   bash scripts/setup_environment.sh models       # download weights only
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

log() { echo "[setup] $*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 1
  }
}

create_fast_dllm() {
  log "Creating conda env: fast_dllm (Python 3.10, torch 2.6+cu124)"
  if conda env list | awk '{print $1}' | grep -qx fast_dllm; then
    log "fast_dllm already exists — skipping create"
  else
    conda create -n fast_dllm python=3.10 -y
  fi
  conda run -n fast_dllm pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url "$TORCH_INDEX"
  conda run -n fast_dllm pip install -r "$REPO_ROOT/environment/fast_dllm.txt"
  conda run -n fast_dllm python -c \
    "import torch, transformers; print('fast_dllm OK:', torch.__version__, transformers.__version__, 'cuda=', torch.cuda.is_available())"
}

create_llada_quant() {
  log "Creating conda env: llada_quant (Python 3.10, torch 2.5+cu124)"
  if conda env list | awk '{print $1}' | grep -qx llada_quant; then
    log "llada_quant already exists — skipping create"
  else
    conda create -n llada_quant python=3.10 -y
  fi
  conda run -n llada_quant pip install \
    torch==2.5.1 \
    --index-url "$TORCH_INDEX"
  conda run -n llada_quant pip install -r "$REPO_ROOT/environment/llada_quant.txt"
  conda run -n llada_quant python -c \
    "import torch, transformers, bitsandbytes; print('llada_quant OK:', torch.__version__, transformers.__version__, 'cuda=', torch.cuda.is_available())"
}

download_models() {
  need_cmd huggingface-cli
  mkdir -p "$REPO_ROOT/model"
  export HF_HOME

  download_one() {
    local repo="$1"
    local name="$2"
    local target="$REPO_ROOT/model/$name"
    if [[ -e "$target" ]]; then
      log "model/$name already exists — skip"
      return
    fi
    log "Downloading $repo → model/$name"
    huggingface-cli download "$repo" --local-dir "$target"
  }

  # LLaDA (quant / multi_language / causal_pilot)
  download_one GSAI-ML/LLaDA-8B-Base LLaDA-8B-Base

  # Fast-dLLM v2 goes to HF cache by default (transformers from_pretrained).
  # Optional local mirror:
  # download_one Efficient-Large-Model/Fast_dLLM_v2_7B Fast_dLLM_v2_7B
}

setup_all() {
  need_cmd conda
  create_fast_dllm
  create_llada_quant
  log "Done. Activate with: conda activate fast_dllm  |  conda activate llada_quant"
}

case "${1:-all}" in
  fast_dllm) need_cmd conda; create_fast_dllm ;;
  llada_quant) need_cmd conda; create_llada_quant ;;
  models) download_models ;;
  all) setup_all ;;
  *)
    echo "Usage: $0 [all|fast_dllm|llada_quant|models]" >&2
    exit 1
    ;;
esac

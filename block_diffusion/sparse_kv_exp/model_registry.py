"""Model presets for sparse_kv_exp handoff sweeps."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_V2 = _REPO_ROOT / "_fast_dllm_upstream" / "v2"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    conda_env: str
    backend: str
    notes: str = ""


MODEL_PRESETS: dict[str, ModelSpec] = {
    "fast_dllm_v2_7b": ModelSpec(
        name="fast_dllm_v2_7b",
        hf_id="Efficient-Large-Model/Fast_dLLM_v2_7B",
        conda_env="fast_dllm",
        backend="fast_dllm",
        notes="Fully supported: block-diffusion sparse KV hooks.",
    ),
    "llada2_mini_16b": ModelSpec(
        name="llada2_mini_16b",
        hf_id="inclusionAI/LLaDA2.0-mini",
        conda_env="llada_quant",
        backend="llada2",
        notes=(
            "Experimental: loads via HF trust_remote_code. Sparse KV requires "
            "mdm_sample-compatible API (see HANDOFF.md). Test with --smoke first."
        ),
    ),
}


def list_models() -> list[str]:
    return list(MODEL_PRESETS.keys())


def get_model_spec(preset: str) -> ModelSpec:
    if preset not in MODEL_PRESETS:
        raise KeyError(f"Unknown model preset {preset!r}. Choose from: {list_models()}")
    return MODEL_PRESETS[preset]


def _ensure_fast_dllm_upstream() -> None:
    if str(_UPSTREAM_V2) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM_V2))


def load_model_and_tokenizer(
    preset: str,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.nn.Module, AutoTokenizer, object | None]:
    """Returns (model, tokenizer, upstream_batch_sample or None)."""
    spec = get_model_spec(preset)
    print(f"Loading {spec.hf_id} ({spec.backend}) on {device} ...")

    if spec.backend == "fast_dllm":
        _ensure_fast_dllm_upstream()
        import generation_functions as upstream_gen  # noqa: WPS433

        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
        upstream = upstream_gen.Fast_dLLM_QwenForCausalLM.batch_sample
        return model, tokenizer, upstream

    if spec.backend == "llada2":
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
        upstream = getattr(model, "mdm_sample", None) or getattr(model, "batch_sample", None)
        if upstream is None:
            raise RuntimeError(
                f"{spec.hf_id} has no mdm_sample/batch_sample. "
                "Wire LLaDA2 generation in model_registry.py or use fast_dllm_v2_7b."
            )
        return model, tokenizer, upstream

    raise ValueError(f"unsupported backend {spec.backend}")

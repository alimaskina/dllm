"""Load diffusion LMs despite transformers / remote-code version skew."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from tokenization_audit import TOKENIZER_PRESETS

FAST_DLLM_MASK_ID = 151665


@dataclass(frozen=True)
class DLMSpec:
    model_id: str
    mask_id: int | None
    ar_shift: bool
    loader: str  # llada | auto | causal


DLM_SPECS: dict[str, DLMSpec] = {
    "llada": DLMSpec(TOKENIZER_PRESETS["llada"], 126336, False, "llada"),
    "dream": DLMSpec(TOKENIZER_PRESETS["dream"], 151666, True, "auto"),
    "fast_dllm": DLMSpec(TOKENIZER_PRESETS["fast_dllm"], FAST_DLLM_MASK_ID, False, "causal"),
}


def _patch_pretrained_base() -> None:
    if getattr(_patch_pretrained_base, "_done", False):
        return

    def _all_tied_get(self):
        keys = getattr(self, "_tied_weights_keys", None)
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        return {k: k for k in keys}

    def _all_tied_set(self, value):
        self._tied_weights_keys = value

    PreTrainedModel.all_tied_weights_keys = property(_all_tied_get, _all_tied_set)
    _patch_pretrained_base._done = True


def _patch_rope_default() -> None:
    if getattr(_patch_rope_default, "_done", False):
        return
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, _compute_proportional_rope_parameters

    ROPE_INIT_FUNCTIONS.setdefault("default", _compute_proportional_rope_parameters)
    _patch_rope_default._done = True


def _patch_remote_llada(model_id: str) -> None:
    AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_cls = get_class_from_dynamic_module("modeling_llada.LLaDAModelLM", model_id)
    if model_cls is None:
        return

    if not hasattr(model_cls, "all_tied_weights_keys"):
        model_cls.all_tied_weights_keys = property(
            lambda self: getattr(self, "_tied_weights_keys", {}) or {}
        )

    if not getattr(model_cls, "_ml_tie_patched", False):
        _orig = model_cls.tie_weights

        def _tie(self, *args, **kwargs):
            kwargs.pop("missing_keys", None)
            kwargs.pop("recompute_mapping", None)
            try:
                return _orig(self)
            except TypeError:
                return _orig(self, *args)

        model_cls.tie_weights = _tie
        model_cls._ml_tie_patched = True


def _patch_dream_generation_config(model_id: str) -> None:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    try:
        cls = get_class_from_dynamic_module("generation_utils.DreamGenerationConfig", model_id)
    except Exception:
        return
    if getattr(cls, "_ml_validate_patched", False):
        return

    def _validate(self, *args, **kwargs):
        kwargs.pop("user_set_attributes", None)
        return None

    cls.validate = _validate
    cls._ml_validate_patched = True


def resolve_mask_id(preset: str, tokenizer, spec: DLMSpec) -> int:
    if spec.mask_id is not None:
        return spec.mask_id
    mid = getattr(tokenizer, "mask_token_id", None)
    if mid is None:
        raise RuntimeError(f"No mask_token_id for preset={preset}")
    return int(mid)


def model_logits(model, x: torch.Tensor, *, ar_shift: bool) -> torch.Tensor:
    logits = model(x).logits
    if ar_shift:
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


def load_llada(model_id: str, device: str = "cuda", dtype=torch.float16):
    _patch_pretrained_base()
    _patch_remote_llada(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model.to(device)


def load_causal(model_id: str, device: str = "cuda", dtype=torch.bfloat16):
    _patch_pretrained_base()
    _patch_rope_default()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    return model.to(device)


def load_dlm(preset: str, device: str = "cuda", dtype=torch.float16):
    """Load DLM by preset name (llada, dream, fast_dllm)."""
    if preset not in DLM_SPECS:
        raise KeyError(f"Unknown DLM preset: {preset}")
    spec = DLM_SPECS[preset]
    _patch_pretrained_base()
    _patch_rope_default()
    if spec.loader == "llada":
        return load_llada(spec.model_id, device, dtype=dtype)
    if spec.loader == "auto":
        _patch_dream_generation_config(spec.model_id)
        model = AutoModel.from_pretrained(
            spec.model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        model.config.use_cache = False
        return model.to(device)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False
    return model.to(device)


def get_dlm_spec(preset: str) -> DLMSpec:
    return DLM_SPECS[preset]

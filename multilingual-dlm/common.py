"""Load LLaDA / Dream for inference (minimal, no generic framework)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

MODEL_IDS = {
    "llada": "GSAI-ML/LLaDA-8B-Base",
    "dream": "Dream-org/Dream-v0-Base-7B",
}

MASK_IDS = {
    "llada": 126336,
    "dream": 151666,
}

AR_SHIFT = {
    "llada": False,
    "dream": True,
}

MIN_GPU_GB = 14


def pick_cuda_device(min_free_gb: float = MIN_GPU_GB) -> str:
    """Pick the CUDA device with the most free memory."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        free_bytes, _total = torch.cuda.mem_get_info(idx)
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx

    need = int(min_free_gb * (1024**3))
    if best_free < need:
        free_gb = best_free / (1024**3)
        raise RuntimeError(
            f"Need >= {min_free_gb:.0f} GB free GPU memory; "
            f"best GPU cuda:{best_idx} has {free_gb:.1f} GB free"
        )
    return f"cuda:{best_idx}"


@dataclass
class LoadedModel:
    """Everything needed for DLM inference and oracle trajectories."""

    preset: str
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    mask_token_id: int
    device: str
    dtype: torch.dtype
    ar_shift: bool
    model_id: str


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


def _load_llada(model_id: str, device: str, dtype: torch.dtype) -> torch.nn.Module:
    _patch_pretrained_base()
    _patch_remote_llada(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=False,
    )
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model.to(device)


def _load_dream(model_id: str, device: str, dtype: torch.dtype) -> torch.nn.Module:
    _patch_pretrained_base()
    _patch_rope_default()
    _patch_dream_generation_config(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False
    return model.to(device)


def load_model(
    preset: str,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> LoadedModel:
    """
    Load LLaDA or Dream with compatibility patches applied.

    Example:
        bundle = load_model("llada")
        bundle = load_model("dream", device="cuda:0")
    """
    preset = preset.lower()
    if preset not in MODEL_IDS:
        raise KeyError(f"Unknown preset {preset!r}; expected one of {sorted(MODEL_IDS)}")

    if device is None:
        device = pick_cuda_device() if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model_id = MODEL_IDS[preset]
    if preset == "llada":
        model = _load_llada(model_id, device, dtype)
    else:
        model = _load_dream(model_id, device, dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mask_token_id = MASK_IDS[preset]
    if getattr(tokenizer, "mask_token_id", None) is not None:
        mask_token_id = int(tokenizer.mask_token_id)

    return LoadedModel(
        preset=preset,
        model=model,
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        device=device,
        dtype=dtype,
        ar_shift=AR_SHIFT[preset],
        model_id=model_id,
    )


def model_logits(bundle: LoadedModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward pass with optional Dream AR-shift on logits."""
    logits = bundle.model(input_ids).logits
    if bundle.ar_shift:
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits

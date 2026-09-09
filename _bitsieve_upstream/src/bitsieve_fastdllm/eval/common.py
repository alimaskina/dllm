from __future__ import annotations

from typing import Any

import torch


def parse_dtype(name: str) -> torch.dtype:
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown dtype: {name}") from exc


def load_fast_dllm(
    model_id: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    revision: str | None = None,
    attn_implementation: str | None = None,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers is required; install this project in its evaluation env") from exc

    # torch.cuda.current_device() stays 0 until something sets it - creating
    # tensors on cuda:N (N != 0) via device= or .to() does NOT move it. Every
    # packed/selector kernel launch in kernels/ops.py then runs against
    # device 0's context while the tensors it is handed live on device N,
    # which Triton reports as "Pointer argument (at 0) cannot be accessed from
    # Triton (cpu tensor?)" - a deterministic crash on every call, not a CPU
    # tensor at all. Only avoided previously by restricting visibility with
    # CUDA_VISIBLE_DEVICES so the requested index was always 0.
    if device not in ("auto", "cpu") and torch.cuda.is_available():
        torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "revision": revision,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def encode_prompt(
    tokenizer,
    prompt: str,
    *,
    max_input_tokens: int,
    use_chat_template: bool = True,
    device: torch.device | str,
) -> torch.Tensor:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    else:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    if ids.shape[1] > max_input_tokens:


        first = max_input_tokens // 2
        last = max_input_tokens - first
        ids = torch.cat([ids[:, :first], ids[:, -last:]], dim=1)
    return ids.to(device)

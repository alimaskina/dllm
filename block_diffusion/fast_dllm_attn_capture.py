"""Capture Fast-dLLM v2 attention weights during block-diffusion forwards."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_CAPTURED: dict[int, torch.Tensor] = {}
_ACTIVE_LAYERS: set[int] | None = None
_ORIG_SDPA = None


def _expand_kv_for_gqa(
    key: torch.Tensor, value: torch.Tensor, num_q_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    num_kv = key.size(1)
    if num_kv == num_q_heads:
        return key, value
    n_rep = num_q_heads // num_kv
    key = key.repeat_interleave(n_rep, dim=1, output_size=num_q_heads)
    value = value.repeat_interleave(n_rep, dim=1, output_size=num_q_heads)
    return key, value


def _compute_attn_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """Return head-mean attention weights [B, T_q, T_k]."""
    key, _ = _expand_kv_for_gqa(key, key, query.size(1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + attn_mask
    weights = F.softmax(scores, dim=-1)
    return weights.mean(dim=1)


def _capturing_sdpa(module, query, key, value, attn_mask, **kwargs):
    layer_idx = getattr(module, "layer_idx", None)
    if _ACTIVE_LAYERS is None or layer_idx in _ACTIVE_LAYERS:
        scaling = kwargs.get("scaling", query.size(-1) ** -0.5)
        _CAPTURED[int(layer_idx)] = _compute_attn_weights(
            query, key, attn_mask, scaling
        ).detach()
    return _ORIG_SDPA(module, query, key, value, attn_mask, **kwargs)


def _ensure_patch() -> None:
    global _ORIG_SDPA
    if _ORIG_SDPA is None:
        _ORIG_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _capturing_sdpa


def get_layers(model):
    return model.model.layers


@contextmanager
def capture_attention(layer_ids: set[int] | None = None) -> Iterator[dict[int, torch.Tensor]]:
    """Capture head-mean attention per layer during model.forward."""
    global _ACTIVE_LAYERS
    _ensure_patch()
    _CAPTURED.clear()
    _ACTIVE_LAYERS = layer_ids  # None → all layers
    try:
        yield _CAPTURED
    finally:
        _ACTIVE_LAYERS = None
        _CAPTURED.clear()


def aggregate_layers(captured: dict[int, torch.Tensor]) -> torch.Tensor:
    """Mean over layers → [T_q, T_k]."""
    if not captured:
        raise ValueError("no attention captured")
    layers = sorted(captured)
    stacked = torch.stack([captured[l][0].float() for l in layers], dim=0)
    return stacked.mean(dim=0)


def block_saliency(
    attn: torch.Tensor,
    *,
    query_start: int,
    query_end: int,
    key_start: int,
    key_end: int,
) -> list[float]:
    """Aggregate attention from query block onto each key token in target block."""
    q_slice = attn[query_start:query_end, key_start:key_end]
    if q_slice.numel() == 0:
        return [0.0] * (key_end - key_start)
    return q_slice.sum(dim=0).tolist()

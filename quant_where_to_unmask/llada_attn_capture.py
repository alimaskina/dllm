"""Capture LLaDA attention weights (output_attentions is unsupported upstream)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F

_CAPTURED: dict[int, torch.Tensor] = {}
_ACTIVE_LAYERS: set[int] = set()
_ORIG_SDPA = None
_PATCHED_CLS: set[type] = set()


def _compute_attn_weights(q: torch.Tensor, k: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
    """Return head-mean attention weights [B, T, T]."""
    num_q_heads = q.size(1)
    if num_q_heads != k.size(1):
        k = k.repeat_interleave(num_q_heads // k.size(1), dim=1, output_size=num_q_heads)

    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    return F.softmax(scores, dim=-1).mean(dim=1)


def _make_patched_sdpa(orig):
    def patched(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        layer_id = getattr(self, "layer_id", None)
        if layer_id is not None and layer_id in _ACTIVE_LAYERS:
            _CAPTURED[layer_id] = _compute_attn_weights(q, k, attn_mask).detach()
        return orig(self, q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    return patched


def _ensure_patch(block) -> None:
    global _ORIG_SDPA
    cls = type(block)
    if cls in _PATCHED_CLS:
        return
    _ORIG_SDPA = cls._scaled_dot_product_attention
    cls._scaled_dot_product_attention = _make_patched_sdpa(_ORIG_SDPA)
    _PATCHED_CLS.add(cls)


def get_blocks(model):
    return model.model.transformer.blocks


@contextmanager
def capture_attention(model, layer_ids: set[int]) -> Iterator[dict[int, torch.Tensor]]:
    global _ACTIVE_LAYERS
    blocks = get_blocks(model)
    if blocks:
        _ensure_patch(blocks[0])
    _CAPTURED.clear()
    _ACTIVE_LAYERS = set(layer_ids)
    try:
        yield _CAPTURED
    finally:
        _ACTIVE_LAYERS = set()
        _CAPTURED.clear()


def forward_attn(model, x: torch.Tensor, layer_ids: set[int]) -> dict[int, torch.Tensor]:
    with torch.inference_mode():
        with capture_attention(model, layer_ids) as captured:
            model(x, attention_mask=torch.ones_like(x))
            return {lid: captured[lid][0].float().cpu() for lid in layer_ids if lid in captured}
    return {}


def attn_mass(attn_row: torch.Tensor, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(attn_row[torch.tensor(indices, dtype=torch.long)].sum().item())

#!/usr/bin/env python3
"""Capture Dream attention weights via eager attention or SDPA hook."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F

_CAPTURED: dict[int, torch.Tensor] = {}
_ACTIVE_LAYERS: set[int] = set()
_ORIG_SDPA = None


def _compute_attn_weights(q: torch.Tensor, k: torch.Tensor, attn_mask) -> torch.Tensor:
    """q,k: [B, H, T, D] → head-mean [B, T, T]."""
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None and isinstance(attn_mask, torch.Tensor):
        scores = scores + attn_mask
    return F.softmax(scores, dim=-1).mean(dim=1)


def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    # DreamSdpaAttention calls F.scaled_dot_product_attention; we intercept when layers active.
    # layer_id is set via thread-local-ish: we can't get layer from F.sdpa easily.
    # Prefer eager path instead.
    return _ORIG_SDPA(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )


def get_layers(model):
    return model.model.layers


@contextmanager
def capture_attention_eager(model, layer_ids: set[int]) -> Iterator[dict[int, torch.Tensor]]:
    """Monkeypatch DreamAttention.forward (eager parent) to save weights when output_attentions."""
    global _ACTIVE_LAYERS, _CAPTURED
    layers = get_layers(model)
    # Ensure eager attention modules
    from transformers.models.dream.modeling_dream import DreamAttention, DreamSdpaAttention  # type: ignore

    # Actually Dream is trust_remote_code — import from model module
    attn_cls = type(layers[0].self_attn)
    orig_forward = attn_cls.forward

    def wrapped_forward(self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        cache_position=None, position_embeddings=None, **kwargs):
        # Force attentions if this layer is active
        want = getattr(self, "layer_idx", None) in _ACTIVE_LAYERS
        out = orig_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=want or output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # out: (attn_output, attn_weights, past)
        if want and len(out) >= 2 and out[1] is not None:
            # attn_weights [B, H, T, T]
            _CAPTURED[int(self.layer_idx)] = out[1].detach().mean(dim=1)
        return out

    _CAPTURED.clear()
    _ACTIVE_LAYERS = set(layer_ids)
    attn_cls.forward = wrapped_forward
    try:
        yield _CAPTURED
    finally:
        attn_cls.forward = orig_forward
        _ACTIVE_LAYERS = set()
        _CAPTURED.clear()


def forward_attn(model, x: torch.Tensor, layer_ids: set[int]) -> dict[int, torch.Tensor]:
    """Return {layer: [T,T] float cpu} head-mean attention."""
    # Prefer loading model with attn_implementation='eager'; fallback wraps SDPA module
    with torch.inference_mode():
        # Use output_attentions=True on whole model if eager
        try:
            out = model(x, attention_mask="full", output_attentions=True)
            if out.attentions is not None:
                return {
                    lid: out.attentions[lid][0].float().mean(dim=0).cpu()
                    for lid in layer_ids
                    if lid < len(out.attentions)
                }
        except Exception:
            pass

        with capture_attention_eager(model, layer_ids) as captured:
            model(x, attention_mask="full", output_attentions=False)
            return {lid: captured[lid][0].float().cpu() for lid in layer_ids if lid in captured}
    return {}


def attn_mass(attn_row: torch.Tensor, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(attn_row[torch.tensor(indices, dtype=torch.long)].sum().item())

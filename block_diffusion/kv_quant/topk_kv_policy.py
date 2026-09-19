"""Top-K cached-token selection by step-0 attention importance."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

from kv_cache_quant import (
    DEFAULT_KIVI_GROUP_SIZE,
    apply_rope_keys,
    inverse_rope_keys,
    quantize_kivi_key_single_token,
    quantize_kivi_value_single_token,
    rope_cos_sin,
)


def aggregate_importance_across_layers(
    per_layer_importance: dict[int, list[float]],
) -> list[float]:
    if not per_layer_importance:
        return []
    n = max(len(v) for v in per_layer_importance.values())
    agg = [0.0] * n
    for imp in per_layer_importance.values():
        for i, v in enumerate(imp):
            agg[i] += float(v)
    return agg


def select_top_k_indices(importance: list[float], k: int) -> set[int]:
    if not importance:
        return set()
    k = min(k, len(importance))
    order = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)
    return set(order[:k])


def clone_dynamic_cache(past_key_values):
    from transformers.cache_utils import DynamicCache

    cloned = DynamicCache()
    for layer_id in range(len(past_key_values)):
        cloned.key_cache.append(past_key_values.key_cache[layer_id].clone())
        cloned.value_cache.append(past_key_values.value_cache[layer_id].clone())
    return cloned


def quantize_keep_tokens_kivi(
    past_key_values,
    keep_indices: set[int],
    bits: int,
    *,
    keys_pre_rope: bool = False,
    model=None,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
) -> None:
    """In-place KIVI round-trip on selected cached token positions."""
    if bits <= 0 or not keep_indices:
        return

    idx_list = sorted(keep_indices)
    cos = sin = None
    if keys_pre_rope:
        if model is None:
            raise ValueError("model is required when keys_pre_rope=True")
        key0 = past_key_values.key_cache[0]
        cos, sin = rope_cos_sin(
            model,
            key0.shape[-2],
            key0.shape[0],
            key0.device,
            key0.dtype,
        )

    for layer_id in range(len(past_key_values)):
        key = past_key_values.key_cache[layer_id]
        value = past_key_values.value_cache[layer_id]
        for tok in idx_list:
            k_tok = key[:, :, tok : tok + 1, :]
            if keys_pre_rope:
                k_tok = inverse_rope_keys(
                    k_tok, cos[:, tok : tok + 1, :], sin[:, tok : tok + 1, :]
                )
                k_tok = quantize_kivi_key_single_token(
                    k_tok, bits, group_size=kivi_group_size
                )
                k_tok = apply_rope_keys(
                    k_tok, cos[:, tok : tok + 1, :], sin[:, tok : tok + 1, :]
                )
            else:
                k_tok = quantize_kivi_key_single_token(
                    k_tok, bits, group_size=kivi_group_size
                )
            key[:, :, tok : tok + 1, :] = k_tok
            value[:, :, tok : tok + 1, :] = quantize_kivi_value_single_token(
                value[:, :, tok : tok + 1, :], bits, group_size=kivi_group_size
            )


@contextmanager
def keep_kv_indices(model, keep_indices: set[int] | None) -> Iterator[None]:
    """Patch eval_mask so attention only sees selected cached token columns."""
    if not keep_indices:
        yield
        return

    orig = model.model.eval_mask

    def patched(seqlen: int, block_size: int, cache_seq_len: int) -> torch.Tensor:
        base = orig(seqlen, block_size, cache_seq_len)
        for k in range(cache_seq_len):
            if k not in keep_indices:
                base[:, k] = False
        return base

    model.model.eval_mask = patched
    try:
        yield
    finally:
        model.model.eval_mask = orig

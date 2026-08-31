"""SDPA hook: sparse old-cache + independent selector/exec Q/K/V precision."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from cost_model import RunCostSummary, StepCost
from kv_store import DualPrecisionCache
from quantization import apply_precision

_ORIG_SDPA = None
# Native sdpa snapshot at module import — used to unconditionally call the true
# backend, avoiding recursion when fast_dllm_attn_capture is chained on top.
_NATIVE_SDPA = ALL_ATTENTION_FUNCTIONS.get("sdpa")


@dataclass
class AttentionContext:
    enabled: bool = False
    cache_len: int = 0
    current_block_len: int = 0
    num_queries: int = 0
    phase: str = "exec"
    sparse: bool = False
    per_layer_keep: dict[int, list[int] | list[list[int]]] = field(default_factory=dict)
    per_head_sparse: bool = True
    q_bits: int = 16
    k_bits: int = 16
    v_bits: int = 16
    quantize_current_kv: bool = False
    kv_store: DualPrecisionCache | None = None
    cost_summary: RunCostSummary | None = None
    num_q_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    num_layers: int = 0
    _recorded_this_forward: bool = False

    def reset_forward(self) -> None:
        self._recorded_this_forward = False


CTX = AttentionContext()


def _apply_sparse_mask(
    attn_mask: torch.Tensor | None,
    keep_indices: list[int],
    cache_len: int,
    total_k: int,
    num_queries: int,
    device: torch.device,
) -> torch.Tensor:
    if cache_len <= 0:
        return attn_mask

    col_mask = torch.zeros(total_k, dtype=torch.bool, device=device)
    if keep_indices:
        keep_t = torch.as_tensor(keep_indices, device=device, dtype=torch.long)
        # clamp defensively; keep_indices are prompt-range but caller may not enforce
        keep_t = keep_t[(keep_t >= 0) & (keep_t < cache_len)]
        col_mask.index_fill_(0, keep_t, True)
    col_mask[cache_len:] = True

    if attn_mask is None:
        return col_mask.unsqueeze(0).expand(num_queries, -1)

    if attn_mask.dtype == torch.bool:
        if attn_mask.dim() == 2:
            return attn_mask & col_mask.unsqueeze(0)
        if attn_mask.dim() == 3:  # [B, T_q, T_k]
            return attn_mask & col_mask.view(1, 1, -1)
        if attn_mask.dim() == 4:  # [B, H, T_q, T_k]
            return attn_mask & col_mask.view(1, 1, 1, -1)
        return attn_mask

    add = torch.zeros(total_k, device=device, dtype=attn_mask.dtype)
    add[~col_mask] = torch.finfo(attn_mask.dtype).min
    if attn_mask.dim() == 2:
        return attn_mask + add.unsqueeze(0)
    if attn_mask.dim() == 3:
        return attn_mask + add.view(1, 1, -1)
    if attn_mask.dim() == 4:
        return attn_mask + add.view(1, 1, 1, -1)
    return attn_mask + add


def _apply_sparse_mask_per_head(
    attn_mask: torch.Tensor | None,
    keep_per_head: list[list[int]],
    cache_len: int,
    total_k: int,
    num_queries: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build [H, T_q, T_k] additive/bool mask — each head keeps its own old-cache columns."""
    if cache_len <= 0:
        return attn_mask

    num_heads = len(keep_per_head)
    col_mask = torch.zeros(num_heads, total_k, dtype=torch.bool, device=device)
    # Vectorized: build a flat [num_pairs, 2] tensor of (head, key_idx) and scatter once.
    head_rows: list[int] = []
    key_cols: list[int] = []
    for h, keep_indices in enumerate(keep_per_head):
        for k in keep_indices:
            if 0 <= k < cache_len:
                head_rows.append(h)
                key_cols.append(k)
    if head_rows:
        h_t = torch.as_tensor(head_rows, device=device, dtype=torch.long)
        k_t = torch.as_tensor(key_cols, device=device, dtype=torch.long)
        col_mask[h_t, k_t] = True
    col_mask[:, cache_len:] = True

    head_qk = col_mask.unsqueeze(1).expand(num_heads, num_queries, total_k)

    if attn_mask is None:
        add = torch.zeros(num_heads, num_queries, total_k, device=device, dtype=dtype)
        add[~head_qk] = torch.finfo(dtype).min
        return add

    if attn_mask.dim() == 2:
        base = attn_mask.unsqueeze(0).expand(num_heads, -1, -1)
    elif attn_mask.dim() == 3:
        base = attn_mask
    else:
        base = attn_mask

    if base.dtype == torch.bool:
        return base & head_qk

    add_cols = torch.zeros(num_heads, total_k, device=device, dtype=base.dtype)
    add_cols[~col_mask] = torch.finfo(base.dtype).min
    return base + add_cols.unsqueeze(1)


def _prepare_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    cache_len = CTX.cache_len
    t_q = query.shape[2]
    q = apply_precision(query, CTX.q_bits, "q_per_token")

    k_old = key[:, :, :cache_len, :] if cache_len > 0 else None
    k_cur = key[:, :, cache_len:, :]
    v_old = value[:, :, :cache_len, :] if cache_len > 0 else None
    v_cur = value[:, :, cache_len:, :]

    if cache_len > 0 and CTX.kv_store is not None:
        store_k, store_v = CTX.kv_store.get_layer_old_kv(
            layer_idx, phase=CTX.phase, sparse=CTX.sparse
        )
        if store_k is not None and store_v is not None:
            k_old = store_k
            v_old = store_v

    if CTX.quantize_current_kv:
        k_cur = apply_precision(k_cur, CTX.k_bits, "k_per_channel")
        v_cur = apply_precision(v_cur, CTX.v_bits, "v_per_token")

    if k_old is not None:
        k = torch.cat([k_old, k_cur], dim=2)
        v = torch.cat([v_old, v_cur], dim=2)
    else:
        k, v = k_cur, v_cur

    return q, k, v, k_old


def _capturing_sdpa(module, query, key, value, attn_mask, **kwargs):
    if not CTX.enabled:
        # Always go to the true native backend, not any prior wrapper. This is
        # what prevents circular chains when fast_dllm_attn_capture is layered
        # on top (capture-outer → hook-inner → native).
        return _NATIVE_SDPA(module, query, key, value, attn_mask, **kwargs)

    layer_idx = int(getattr(module, "layer_idx", -1))
    q, k, v, _ = _prepare_qkv(query, key, value, layer_idx)

    mask = attn_mask
    if CTX.sparse and CTX.cache_len > 0:
        keep = CTX.per_layer_keep.get(layer_idx)
        if keep is None:
            keep = list(range(CTX.cache_len))
        if CTX.per_head_sparse and keep and isinstance(keep[0], list):
            mask = _apply_sparse_mask_per_head(
                mask,
                keep,  # type: ignore[arg-type]
                CTX.cache_len,
                k.shape[2],
                q.shape[2],
                q.device,
                q.dtype,
            )
        else:
            mask = _apply_sparse_mask(
                mask,
                keep if isinstance(keep, list) and (not keep or isinstance(keep[0], int)) else list(range(CTX.cache_len)),  # type: ignore[arg-type]
                CTX.cache_len,
                k.shape[2],
                q.shape[2],
                q.device,
            )

    out = _NATIVE_SDPA(module, q, k, v, mask, **kwargs)

    if (
        not CTX._recorded_this_forward
        and CTX.cost_summary is not None
        and layer_idx == 0
    ):
        sel_k = CTX.cache_len
        if CTX.sparse:
            keeps = CTX.per_layer_keep.values()
            if keeps and CTX.per_head_sparse and isinstance(next(iter(keeps)), list):
                first = next(iter(keeps))
                if first and isinstance(first[0], list):
                    sel_k = len(first[0])
            else:
                sel_k = max((len(v) for v in keeps if isinstance(v, list) and (not v or isinstance(v[0], int))), default=0)
        step = StepCost(
            phase=CTX.phase,
            num_queries=CTX.num_queries,
            old_cache_len=CTX.cache_len,
            selected_k=sel_k if CTX.sparse else CTX.cache_len,
            current_block_len=CTX.current_block_len,
            sparse=CTX.sparse,
            q_bits=CTX.q_bits,
            k_bits=CTX.k_bits,
            v_bits=CTX.v_bits,
            num_q_heads=CTX.num_q_heads or query.size(1),
            num_kv_heads=CTX.num_kv_heads or key.size(1),
            head_dim=CTX.head_dim or query.size(-1),
            num_layers=CTX.num_layers or 1,
        )
        CTX.cost_summary.add(step)
        CTX._recorded_this_forward = True

    return out


def _ensure_patch() -> None:
    """Keep this hook outermost so capture sees post-quant Q/K."""
    global _ORIG_SDPA
    current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if current is not _capturing_sdpa:
        _ORIG_SDPA = current
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _capturing_sdpa


def configure_from_model(model) -> None:
    cfg = model.config
    CTX.num_q_heads = cfg.num_attention_heads
    CTX.num_kv_heads = cfg.num_key_value_heads
    CTX.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    CTX.num_layers = cfg.num_hidden_layers


@contextmanager
def attention_experiment(
    *,
    cache_len: int,
    current_block_len: int,
    num_queries: int,
    phase: str,
    sparse: bool,
    per_layer_keep: dict[int, list[int] | list[list[int]]] | None,
    per_head_sparse: bool = True,
    q_bits: int,
    k_bits: int,
    v_bits: int,
    kv_store: DualPrecisionCache | None,
    cost_summary: RunCostSummary | None,
    enabled: bool = True,
) -> Iterator[None]:
    _ensure_patch()
    prev = AttentionContext(
        enabled=CTX.enabled,
        cache_len=CTX.cache_len,
        current_block_len=CTX.current_block_len,
        num_queries=CTX.num_queries,
        phase=CTX.phase,
        sparse=CTX.sparse,
        per_layer_keep=dict(CTX.per_layer_keep),
        per_head_sparse=CTX.per_head_sparse,
        q_bits=CTX.q_bits,
        k_bits=CTX.k_bits,
        v_bits=CTX.v_bits,
        kv_store=CTX.kv_store,
        cost_summary=CTX.cost_summary,
        num_q_heads=CTX.num_q_heads,
        num_kv_heads=CTX.num_kv_heads,
        head_dim=CTX.head_dim,
        num_layers=CTX.num_layers,
    )
    CTX.enabled = enabled
    CTX.cache_len = cache_len
    CTX.current_block_len = current_block_len
    CTX.num_queries = num_queries
    CTX.phase = phase
    CTX.sparse = sparse
    CTX.per_head_sparse = per_head_sparse
    CTX.per_layer_keep = per_layer_keep or {}
    CTX.q_bits = q_bits
    CTX.k_bits = k_bits
    CTX.v_bits = v_bits
    CTX.kv_store = kv_store
    CTX.cost_summary = cost_summary
    CTX.reset_forward()
    try:
        yield
    finally:
        CTX.enabled = prev.enabled
        CTX.cache_len = prev.cache_len
        CTX.current_block_len = prev.current_block_len
        CTX.num_queries = prev.num_queries
        CTX.phase = prev.phase
        CTX.sparse = prev.sparse
        CTX.per_head_sparse = prev.per_head_sparse
        CTX.per_layer_keep = prev.per_layer_keep
        CTX.q_bits = prev.q_bits
        CTX.k_bits = prev.k_bits
        CTX.v_bits = prev.v_bits
        CTX.kv_store = prev.kv_store
        CTX.cost_summary = prev.cost_summary

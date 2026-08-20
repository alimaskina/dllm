"""Per-layer (and per-head) top-k selection policies for old KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from config import SelectorConfig
from quantization import apply_precision

KeepIndices = list[int] | list[list[int]]


def uniform_query_indices(num_queries: int, n: int) -> list[int]:
    if num_queries <= 0:
        return []
    if n >= num_queries:
        return list(range(num_queries))
    if n == 1:
        return [num_queries // 2]
    step = (num_queries - 1) / (n - 1)
    return sorted({int(round(i * step)) for i in range(n)})


def query_indices_for_mode(
    num_queries: int,
    mode: Literal["all_mean", "middle", "uniform"],
    uniform_n: int = 3,
) -> list[int] | None:
    """Return query positions to aggregate; None means all queries (mean)."""
    if mode == "all_mean":
        return None
    if mode == "middle":
        return [num_queries // 2] if num_queries > 0 else []
    return uniform_query_indices(num_queries, uniform_n)


def importance_from_attn(
    attn: torch.Tensor,
    num_cached: int,
    *,
    query_indices: list[int] | None,
    reduce: str = "mean",
) -> list[float] | torch.Tensor:
    """Head-mean attn [B,T_q,T_k] → importance [cache]. Per-head [B,H,T_q,T_k] → [H,cache]."""
    if attn.dim() == 4:
        a = attn[0, :, :, :num_cached].float()
        if query_indices is not None:
            q = a[:, query_indices, :]
        else:
            q = a
        if q.shape[1] == 0:
            return torch.zeros(a.shape[0], num_cached, dtype=a.dtype, device=a.device)
        if reduce == "mean":
            return q.mean(dim=1)
        return q.sum(dim=1)

    a = attn[0, :, :num_cached].float()
    if query_indices is not None:
        q = a[query_indices]
    else:
        q = a
    if q.shape[0] == 0:
        return [0.0] * num_cached
    if reduce == "mean":
        imp = q.mean(dim=0)
    else:
        imp = q.sum(dim=0)
    return imp.tolist()


def select_top_k(importance: list[float], k: int) -> list[int]:
    if not importance or k <= 0:
        return list(range(len(importance)))
    k = min(k, len(importance))
    order = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)
    return sorted(order[:k])


def select_top_k_batch(importance: torch.Tensor, k: int) -> list[list[int]]:
    """Per-row top-k; importance [H, cache] → list of H index lists."""
    out: list[list[int]] = []
    for h in range(importance.shape[0]):
        out.append(select_top_k(importance[h].tolist(), k))
    return out


def attention_mass_captured(
    importance: list[float],
    selected: list[int],
) -> float:
    total = sum(importance)
    if total <= 0:
        return 0.0
    kept = sum(importance[i] for i in selected)
    return float(kept / total)


def attention_mass_captured_batch(
    importance: torch.Tensor,
    selected_per_head: list[list[int]],
) -> float:
    masses = []
    for h, sel in enumerate(selected_per_head):
        row = importance[h].tolist()
        masses.append(attention_mass_captured(row, sel))
    return float(sum(masses) / len(masses)) if masses else 0.0


def reference_old_cache_mass_per_head(
    attn: torch.Tensor,
    num_cached: int,
    masked_query_indices: list[int],
) -> torch.Tensor:
    """Reference mass on old cache: sum over masked block queries.

    attn: [H, T_q, T_k] dense FP16 probe (full softmax keys).
    Returns [H, num_cached] — not renormalized; current-block keys excluded.
    """
    if not masked_query_indices or num_cached <= 0:
        h = attn.shape[0] if attn.dim() == 3 else 1
        return torch.zeros(h, num_cached, dtype=torch.float32, device=attn.device)
    a = attn.float()
    if a.dim() == 2:
        a = a.unsqueeze(0)
    q = a[:, masked_query_indices, :num_cached]
    return q.sum(dim=1)


def coverage_vs_reference_mass(
    ref_mass: list[float] | torch.Tensor,
    selected: list[int],
) -> float:
    """Fraction of reference old-cache mass retained by selected indices."""
    if isinstance(ref_mass, torch.Tensor):
        ref_mass = ref_mass.tolist()
    total = sum(ref_mass)
    if total <= 0:
        return 0.0
    kept = sum(ref_mass[i] for i in selected if 0 <= i < len(ref_mass))
    return float(kept / total)


def coverage_vs_reference_batch(
    attn: torch.Tensor,
    num_cached: int,
    masked_query_indices: list[int],
    selected_per_head: list[list[int]],
) -> float:
    """Mean coverage vs full-block-masked reference, per head. attn [H,T_q,T_k] or [B,H,...]."""
    if attn.dim() == 4:
        attn = attn[0]
    ref = reference_old_cache_mass_per_head(attn, num_cached, masked_query_indices)
    if not selected_per_head:
        return 0.0
    masses = [
        coverage_vs_reference_mass(ref[h], sel)
        for h, sel in enumerate(selected_per_head)
    ]
    return float(sum(masses) / len(masses))


def rank_overlap_at_k(
    ref_mass: list[float] | torch.Tensor,
    sel_mass: list[float] | torch.Tensor,
    k: int,
) -> float:
    """|top-k(ref) ∩ top-k(sel)| / k."""
    if isinstance(ref_mass, torch.Tensor):
        ref_mass = ref_mass.tolist()
    if isinstance(sel_mass, torch.Tensor):
        sel_mass = sel_mass.tolist()
    if k <= 0 or not ref_mass:
        return 0.0
    k = min(k, len(ref_mass))
    ref_top = set(select_top_k(ref_mass, k))
    sel_top = set(select_top_k(sel_mass, k))
    return len(ref_top & sel_top) / k


def rank_overlap_at_k_batch(
    ref_mass: torch.Tensor,
    sel_mass: torch.Tensor,
    k: int,
) -> float:
    """Mean top-k set overlap between reference and selector importance, per head."""
    if ref_mass.numel() == 0:
        return 0.0
    overlaps = [
        rank_overlap_at_k(ref_mass[h].tolist(), sel_mass[h].tolist(), k)
        for h in range(ref_mass.shape[0])
    ]
    return float(sum(overlaps) / len(overlaps))


def _expand_kv_for_gqa(key: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    num_kv = key.size(1)
    if num_kv == num_q_heads:
        return key
    n_rep = num_q_heads // num_kv
    return key.repeat_interleave(n_rep, dim=1, output_size=num_q_heads)


def recompute_attn_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    scaling: float,
    *,
    num_cached: int,
    old_k_bits: int = 16,
) -> torch.Tensor:
    """Recompute softmax attention; optionally quantize old-cache K only.

    Returns [B, H, T_q, T_k] with the same key span as ``key``.
    """
    k = key.clone()
    if num_cached > 0 and old_k_bits < 16:
        k_old = apply_precision(k[:, :, :num_cached, :], old_k_bits, "k_per_channel")
        k = torch.cat([k_old, k[:, :, num_cached:, :]], dim=2)
    k = _expand_kv_for_gqa(k, query.size(1))
    scores = torch.matmul(query, k.transpose(-2, -1)) * scaling
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            scores = scores + attn_mask.unsqueeze(0)
        elif attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + attn_mask
    return F.softmax(scores, dim=-1)


def quant_probe_selection(
    qk: dict[str, torch.Tensor | float | None],
    *,
    num_cached: int,
    topk: int,
    masked_query_indices: list[int],
    old_k_bits: int,
    per_head: bool,
) -> list[list[int]] | list[int]:
    """Top-k from all-mean over masked queries with quant old-cache K."""
    query = qk["query"]
    assert isinstance(query, torch.Tensor)
    key = qk["key"]
    assert isinstance(key, torch.Tensor)
    mask = qk.get("attn_mask")
    scaling = float(qk["scaling"])  # type: ignore[arg-type]
    attn = recompute_attn_weights(
        query, key, mask if isinstance(mask, torch.Tensor) else None, scaling,
        num_cached=num_cached,
        old_k_bits=old_k_bits,
    )
    imp = importance_from_attn(
        attn,
        num_cached,
        query_indices=masked_query_indices,
        reduce="mean",
    )
    if per_head and isinstance(imp, torch.Tensor):
        return select_top_k_batch(imp, topk)
    imp_list = imp if isinstance(imp, list) else imp.mean(dim=0).tolist()
    return select_top_k(imp_list, topk)


def union_indices(selected_per_head: list[list[int]]) -> list[int]:
    u: set[int] = set()
    for sel in selected_per_head:
        u.update(sel)
    return sorted(u)


@dataclass
class LayerSelection:
    layer_id: int
    old_cache_len: int
    selected_k: int
    selected_indices: KeepIndices
    importance: list[float] | None
    mass_captured: float
    selector_self_coverage: float
    rank_overlap_at_k: float
    quant_probe_coverage: float | None = None
    per_head: bool = False


def select_per_layer(
    captured: dict[int, torch.Tensor],
    *,
    num_cached: int,
    block_size: int,
    selector: SelectorConfig,
    masked_query_indices: list[int] | None = None,
    captured_qk: dict[int, dict[str, torch.Tensor | float | None]] | None = None,
    probe_old_k_bits: int | None = None,
) -> dict[int, LayerSelection]:
    """Per-layer top-k from step-0 probe; coverage always vs full-block→old-cache ref."""
    query_end = block_size
    q_idx = query_indices_for_mode(query_end, selector.mode, selector.uniform_n)
    reduce = "mean" if selector.mode == "all_mean" else "sum"
    ref_queries = (
        masked_query_indices
        if masked_query_indices is not None
        else list(range(query_end))
    )
    out: dict[int, LayerSelection] = {}

    for layer_id, attn in captured.items():
        k_eff = selector.effective_topk(num_cached)
        imp = importance_from_attn(
            attn,
            num_cached,
            query_indices=q_idx,
            reduce=reduce,
        )
        attn_h = attn[0].float() if attn.dim() == 4 else attn.float().unsqueeze(0)
        ref_mass = reference_old_cache_mass_per_head(attn_h, num_cached, ref_queries)

        quant_cov: float | None = None
        if (
            probe_old_k_bits is not None
            and probe_old_k_bits < 16
            and captured_qk is not None
            and layer_id in captured_qk
        ):
            q_sel = quant_probe_selection(
                captured_qk[layer_id],
                num_cached=num_cached,
                topk=k_eff,
                masked_query_indices=ref_queries,
                old_k_bits=probe_old_k_bits,
                per_head=selector.per_head,
            )
            if selector.per_head and isinstance(q_sel[0], list):
                quant_cov = coverage_vs_reference_batch(
                    attn_h, num_cached, ref_queries, q_sel  # type: ignore[arg-type]
                )
            else:
                quant_cov = coverage_vs_reference_mass(
                    ref_mass.mean(dim=0), q_sel  # type: ignore[arg-type]
                )

        if selector.per_head and isinstance(imp, torch.Tensor):
            selected_ph = select_top_k_batch(imp, k_eff)
            out[int(layer_id)] = LayerSelection(
                layer_id=int(layer_id),
                old_cache_len=num_cached,
                selected_k=k_eff,
                selected_indices=selected_ph,
                importance=None,
                mass_captured=coverage_vs_reference_batch(
                    attn_h, num_cached, ref_queries, selected_ph
                ),
                selector_self_coverage=attention_mass_captured_batch(imp, selected_ph),
                rank_overlap_at_k=rank_overlap_at_k_batch(ref_mass, imp, k_eff),
                quant_probe_coverage=quant_cov,
                per_head=True,
            )
        else:
            imp_list = imp if isinstance(imp, list) else imp.mean(dim=0).tolist()
            selected = select_top_k(imp_list, k_eff)
            ref_flat = ref_mass.mean(dim=0).tolist()
            out[int(layer_id)] = LayerSelection(
                layer_id=int(layer_id),
                old_cache_len=num_cached,
                selected_k=k_eff,
                selected_indices=selected,
                importance=imp_list,
                mass_captured=coverage_vs_reference_mass(ref_flat, selected),
                selector_self_coverage=attention_mass_captured(imp_list, selected),
                rank_overlap_at_k=rank_overlap_at_k(ref_flat, imp_list, k_eff),
                quant_probe_coverage=quant_cov,
                per_head=False,
            )
    return out

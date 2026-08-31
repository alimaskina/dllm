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
    masked_query_indices: list[int] | None = None,
) -> list[int] | None:
    """Return query positions to aggregate; None means all block queries (mean)."""
    if mode == "all_mean":
        if masked_query_indices is not None:
            return masked_query_indices
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


def cache_token_groups(num_cached: int, group_size: int) -> list[list[int]]:
    """Partition old-cache indices into contiguous token groups."""
    if num_cached <= 0:
        return []
    if group_size <= 1:
        return [[i] for i in range(num_cached)]
    groups: list[list[int]] = []
    for start in range(0, num_cached, group_size):
        groups.append(list(range(start, min(start + group_size, num_cached))))
    return groups


def group_importance(
    importance: list[float],
    groups: list[list[int]],
    *,
    reduce: str = "sum",
) -> list[float]:
    out: list[float] = []
    for group in groups:
        vals = [importance[i] for i in group if 0 <= i < len(importance)]
        if not vals:
            out.append(0.0)
        elif reduce == "max":
            out.append(max(vals))
        else:
            out.append(sum(vals))
    return out


def select_top_k_token_groups(
    importance: list[float],
    *,
    num_cached: int,
    k_tokens: int,
    token_group_size: int,
    group_reduce: str = "sum",
) -> tuple[list[int], int]:
    """Pick top whole token groups; return flat sorted indices and groups kept."""
    if num_cached <= 0 or k_tokens <= 0:
        return [], 0
    if token_group_size <= 1:
        return select_top_k(importance, k_tokens), k_tokens

    groups = cache_token_groups(num_cached, token_group_size)
    if not groups:
        return [], 0

    g_imp = group_importance(importance, groups, reduce=group_reduce)
    n_pick = min(len(groups), max(1, (k_tokens + token_group_size - 1) // token_group_size))
    order = sorted(range(len(groups)), key=lambda i: g_imp[i], reverse=True)
    selected: list[int] = []
    for gi in order[:n_pick]:
        selected.extend(groups[gi])
    return sorted(selected), n_pick


def select_top_k_token_groups_batch(
    importance: torch.Tensor,
    *,
    num_cached: int,
    k_tokens: int,
    token_group_size: int,
    group_reduce: str = "sum",
) -> tuple[list[list[int]], int]:
    """Per-head whole-group top-k; returns per-head index lists and n_groups picked."""
    out: list[list[int]] = []
    n_groups = 0
    for h in range(importance.shape[0]):
        sel, n_groups = select_top_k_token_groups(
            importance[h].tolist(),
            num_cached=num_cached,
            k_tokens=k_tokens,
            token_group_size=token_group_size,
            group_reduce=group_reduce,
        )
        out.append(sel)
    return out, n_groups


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
    kivi_group_size: int = 32,
    kivi_residual_length: int = 32,
    old_k_quant_scheme: str = "kivi",
) -> torch.Tensor:
    """Recompute softmax attention; optionally quantize old-cache K only.

    Returns [B, H, T_q, T_k] with the same key span as ``key``.
    """
    k = key.clone()
    if num_cached > 0 and old_k_bits < 16:
        import sys
        from pathlib import Path

        _kvq = Path(__file__).resolve().parent.parent / "kv_quant"
        if str(_kvq) not in sys.path:
            sys.path.insert(0, str(_kvq))
        from kv_cache_quant import quantize_key_roundtrip

        k_old = quantize_key_roundtrip(
            k[:, :, :num_cached, :],
            old_k_bits,
            scheme=old_k_quant_scheme,
            group_size=kivi_group_size,
            residual_length=kivi_residual_length,
        )
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


def quant_probe_importance(
    qk: dict[str, torch.Tensor | float | None],
    *,
    num_cached: int,
    masked_query_indices: list[int],
    old_k_bits: int,
    kivi_group_size: int = 32,
    kivi_residual_length: int = 32,
    old_k_quant_scheme: str = "kivi",
) -> list[float] | torch.Tensor:
    """Importance from masked-query mean attn with quant old-cache K only."""
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
        kivi_group_size=kivi_group_size,
        kivi_residual_length=kivi_residual_length,
        old_k_quant_scheme=old_k_quant_scheme,
    )
    return importance_from_attn(
        attn,
        num_cached,
        query_indices=masked_query_indices,
        reduce="mean",
    )


def quant_probe_selection(
    qk: dict[str, torch.Tensor | float | None],
    *,
    num_cached: int,
    topk: int,
    masked_query_indices: list[int],
    old_k_bits: int,
    per_head: bool,
    token_group_size: int = 1,
    token_group_reduce: str = "sum",
    kivi_group_size: int = 32,
    kivi_residual_length: int = 32,
    old_k_quant_scheme: str = "kivi",
) -> list[list[int]] | list[int]:
    """Top-k from all-mean over masked queries with quant old-cache K."""
    imp = quant_probe_importance(
        qk,
        num_cached=num_cached,
        masked_query_indices=masked_query_indices,
        old_k_bits=old_k_bits,
        kivi_group_size=kivi_group_size,
        kivi_residual_length=kivi_residual_length,
        old_k_quant_scheme=old_k_quant_scheme,
    )
    if per_head and isinstance(imp, torch.Tensor):
        sel, _ = select_top_k_token_groups_batch(
            imp,
            num_cached=num_cached,
            k_tokens=topk,
            token_group_size=token_group_size,
            group_reduce=token_group_reduce,
        )
        return sel
    imp_list = imp if isinstance(imp, list) else imp.mean(dim=0).tolist()
    sel, _ = select_top_k_token_groups(
        imp_list,
        num_cached=num_cached,
        k_tokens=topk,
        token_group_size=token_group_size,
        group_reduce=token_group_reduce,
    )
    return sel


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
    # Coverage / overlap vs an FP16 reference attention map (same Q/mask/scaling,
    # but FP16 old-cache keys). Present only when the caller provides fp16 keys.
    fp16_ref_coverage: float | None = None
    fp16_ref_rank_overlap_at_k: float | None = None
    quant_probe_coverage: float | None = None
    quant_rank_overlap_at_k: float | None = None
    per_head: bool = False
    token_group_size: int = 1
    selected_groups: int | None = None
    actual_tokens_kept: int | None = None


def select_per_layer(
    captured: dict[int, torch.Tensor],
    *,
    num_cached: int,
    block_size: int,
    selector: SelectorConfig,
    masked_query_indices: list[int] | None = None,
    captured_qk: dict[int, dict[str, torch.Tensor | float | None]] | None = None,
    fp16_old_keys: dict[int, torch.Tensor] | None = None,
    probe_old_k_bits: int | None = None,
    kivi_group_size: int = 32,
    kivi_residual_length: int = 32,
    old_k_quant_scheme: str = "kivi",
    keep_from_quant: bool = False,
) -> dict[int, LayerSelection]:
    """Per-layer top-k from step-0 probe; coverage always vs full-block→old-cache ref."""
    query_end = block_size
    q_idx = query_indices_for_mode(
        query_end, selector.mode, selector.uniform_n, masked_query_indices
    )
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

        fp16_ref_mass: torch.Tensor | None = None
        if (
            fp16_old_keys is not None
            and captured_qk is not None
            and layer_id in captured_qk
            and layer_id in fp16_old_keys
            and num_cached > 0
            and ref_queries
        ):
            qk = captured_qk[layer_id]
            query = qk["query"]
            assert isinstance(query, torch.Tensor)
            key = qk["key"]
            assert isinstance(key, torch.Tensor)
            mask = qk.get("attn_mask")
            scaling = float(qk["scaling"])  # type: ignore[arg-type]
            k_fp16_old = fp16_old_keys[layer_id]
            # key is [B, H_kv, T_old+T_cur, D] for this layer (post-hook). Swap the
            # old-cache slice for the FP16 reference keys; keep the current slice.
            k_cur = key[:, :, num_cached:, :]
            key_fp16 = torch.cat([k_fp16_old, k_cur], dim=2)
            attn_fp16 = recompute_attn_weights(
                query,
                key_fp16,
                mask if isinstance(mask, torch.Tensor) else None,
                scaling,
                num_cached=num_cached,
                old_k_bits=16,
                kivi_group_size=kivi_group_size,
                kivi_residual_length=kivi_residual_length,
                old_k_quant_scheme=old_k_quant_scheme,
            )
            fp16_ref_mass = reference_old_cache_mass_per_head(
                attn_fp16[0], num_cached, ref_queries
            )

        quant_cov: float | None = None
        quant_overlap: float | None = None
        q_sel: list | torch.Tensor | None = None
        q_imp_used: list[float] | torch.Tensor | None = None
        if (
            probe_old_k_bits is not None
            and probe_old_k_bits < 16
            and captured_qk is not None
            and layer_id in captured_qk
        ):
            q_imp = quant_probe_importance(
                captured_qk[layer_id],
                num_cached=num_cached,
                masked_query_indices=ref_queries,
                old_k_bits=probe_old_k_bits,
                kivi_group_size=kivi_group_size,
                kivi_residual_length=kivi_residual_length,
                old_k_quant_scheme=old_k_quant_scheme,
            )
            if selector.per_head and isinstance(q_imp, torch.Tensor):
                q_sel, _ = select_top_k_token_groups_batch(
                    q_imp,
                    num_cached=num_cached,
                    k_tokens=k_eff,
                    token_group_size=selector.token_group_size,
                    group_reduce=selector.token_group_reduce,
                )
                quant_overlap = rank_overlap_at_k_batch(ref_mass, q_imp, k_eff)
                quant_cov = coverage_vs_reference_batch(
                    attn_h, num_cached, ref_queries, q_sel
                )
                q_imp_used = q_imp
            else:
                q_imp_list = q_imp if isinstance(q_imp, list) else q_imp.mean(dim=0).tolist()
                ref_flat = ref_mass.mean(dim=0).tolist()
                q_sel, _ = select_top_k_token_groups(
                    q_imp_list,
                    num_cached=num_cached,
                    k_tokens=k_eff,
                    token_group_size=selector.token_group_size,
                    group_reduce=selector.token_group_reduce,
                )
                quant_overlap = rank_overlap_at_k(ref_flat, q_imp_list, k_eff)
                quant_cov = coverage_vs_reference_mass(ref_flat, q_sel)
                q_imp_used = q_imp_list

        if selector.per_head and isinstance(imp, torch.Tensor):
            selected_ph, n_groups = select_top_k_token_groups_batch(
                imp,
                num_cached=num_cached,
                k_tokens=k_eff,
                token_group_size=selector.token_group_size,
                group_reduce=selector.token_group_reduce,
            )
            if keep_from_quant and q_sel is not None:
                selected_ph = q_sel  # type: ignore[assignment]
                n_groups = None
            actual = len(union_indices(selected_ph))
            fp16_cov: float | None = None
            fp16_overlap: float | None = None
            if fp16_ref_mass is not None:
                fp16_cov = float(
                    sum(
                        coverage_vs_reference_mass(fp16_ref_mass[h], sel)
                        for h, sel in enumerate(selected_ph)
                    )
                    / max(1, len(selected_ph))
                )
                used_mass = (
                    q_imp_used
                    if keep_from_quant and isinstance(q_imp_used, torch.Tensor)
                    else imp
                )
                if isinstance(used_mass, torch.Tensor):
                    fp16_overlap = rank_overlap_at_k_batch(fp16_ref_mass, used_mass, k_eff)
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
                fp16_ref_coverage=fp16_cov,
                fp16_ref_rank_overlap_at_k=fp16_overlap,
                quant_probe_coverage=quant_cov,
                quant_rank_overlap_at_k=quant_overlap,
                per_head=True,
                token_group_size=selector.token_group_size,
                selected_groups=n_groups if selector.token_group_size > 1 else None,
                actual_tokens_kept=actual,
            )
        else:
            imp_list = imp if isinstance(imp, list) else imp.mean(dim=0).tolist()
            selected, n_groups = select_top_k_token_groups(
                imp_list,
                num_cached=num_cached,
                k_tokens=k_eff,
                token_group_size=selector.token_group_size,
                group_reduce=selector.token_group_reduce,
            )
            if keep_from_quant and q_sel is not None:
                selected = q_sel  # type: ignore[assignment]
                n_groups = None
            ref_flat = ref_mass.mean(dim=0).tolist()
            fp16_cov = None
            fp16_overlap = None
            if fp16_ref_mass is not None:
                fp16_flat = fp16_ref_mass.mean(dim=0).tolist()
                fp16_cov = coverage_vs_reference_mass(fp16_flat, selected)
                used_mass = (
                    q_imp_used
                    if keep_from_quant and isinstance(q_imp_used, list)
                    else imp_list
                )
                if isinstance(used_mass, list):
                    fp16_overlap = rank_overlap_at_k(fp16_flat, used_mass, k_eff)
            out[int(layer_id)] = LayerSelection(
                layer_id=int(layer_id),
                old_cache_len=num_cached,
                selected_k=k_eff,
                selected_indices=selected,
                importance=imp_list,
                mass_captured=coverage_vs_reference_mass(ref_flat, selected),
                selector_self_coverage=attention_mass_captured(imp_list, selected),
                rank_overlap_at_k=rank_overlap_at_k(ref_flat, imp_list, k_eff),
                fp16_ref_coverage=fp16_cov,
                fp16_ref_rank_overlap_at_k=fp16_overlap,
                quant_probe_coverage=quant_cov,
                quant_rank_overlap_at_k=quant_overlap,
                per_head=False,
                token_group_size=selector.token_group_size,
                selected_groups=n_groups if selector.token_group_size > 1 else None,
                actual_tokens_kept=len(selected),
            )
    return out

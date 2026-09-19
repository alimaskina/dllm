"""Grad-enabled block-diffusion forward that reads a degraded prefix.

Why this is hand-written
------------------------
Fast-dLLM-v2 offers two forwards and neither can be used here. With
``model.training == True`` it applies its *own* random masking (teacher and
student would see different noise) and runs ``flex_attention`` under
``torch.compile(fullgraph=True)``, which is not interceptable. With
``model.training == False`` it is a block-causal pass over a single sequence,
where block *j*'s context would be the *noisy* tokens of blocks ``< j`` — at
inference those blocks are already decoded and clean.

So this reimplements the doubled ``[x_t ; x_0]`` formulation on the model's own
submodules. An ``x_t`` query in block *j* attends to its own noisy block plus
the **clean** ``x_0`` of blocks ``< j`` — exactly the tokens that live in the
packed cache at decode time, which is what makes them interceptable.

Matching BitSieve
-----------------
* ``residual_tokens: 0`` — nothing is held in bf16, so *every* visible old block
  is degraded. There is no clean tail to carve out.
* Selection is **per KV head**, not per query head: ``selector_importance_reference``
  averages the softmax mass over the ``Hq/Hkv`` query heads sharing a KV head.
* Query representatives follow ``SelectorConfig.query_indices`` — for the shipped
  configs, five masked positions spread across the block.
* ``domain="prefix"``: the softmax runs over old-cache keys only.
* Error compounds. At inference ``_sparse_forward`` calls ``session.attend``
  (degraded read) and then ``stage_if_committing``, so a block's K/V are computed
  from hidden states that already went through a degraded prefix. ``x_0 -> x_0``
  reads are therefore degraded too, by default.

Step 0 of each block runs *dense* at inference (semantic A schedules selection
and attends densely); this forward models the sparse steps, which are the
majority. See ``docs/recovery_training.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import SelectorConfig
from .degrade import CacheDegradation, degrade_keys, degrade_values

# The only fp32 matmuls here are the attention scores below (weights are bf16,
# so this does not touch them). TF32 keeps the manual kernel at least as tight
# against an fp32 reference as fused SDPA is.
torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class BlockDiffMasks:
    """Boolean [S, S] masks over the doubled sequence, S = 2L."""

    visible: torch.Tensor      # upstream block_diff_mask
    old_pair: torch.Tensor     # reads that come out of the packed cache
    block_of: torch.Tensor     # [S] block index per position
    is_xt: torch.Tensor        # [S] True on the x_t half


def build_masks(
    seq_len: int,
    block_size: int,
    device,
    *,
    propagate_to_x0: bool = True,
) -> BlockDiffMasks:
    """Upstream block-diffusion mask plus the set of reads that hit the cache.

    ``seq_len`` is L; returned masks are [2L, 2L]. ``propagate_to_x0`` also
    degrades the x_0 -> x_0 reads, which is what BitSieve does at inference
    (``attend`` before ``stage_if_committing``); turning it off isolates the
    single-block effect but no longer matches the decoder.
    """
    n = seq_len
    idx = torch.arange(2 * n, device=device)
    is_x0 = idx >= n
    blk = torch.where(is_x0, (idx - n) // block_size, idx // block_size)

    bq, bk = blk[:, None], blk[None, :]
    x0_q, x0_k = is_x0[:, None], is_x0[None, :]

    block_diagonal = (bq == bk) & (x0_q == x0_k)
    offset_block_causal = (bq > bk) & x0_k & ~x0_q
    block_causal = (bq >= bk) & x0_k & x0_q
    visible = block_diagonal | offset_block_causal | block_causal

    old_pair = offset_block_causal.clone()
    if propagate_to_x0:
        old_pair = old_pair | ((bq > bk) & x0_k & x0_q)

    return BlockDiffMasks(visible=visible, old_pair=old_pair, block_of=blk, is_xt=~is_x0)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


@torch.no_grad()
def select_keep_mask(
    query: torch.Tensor,
    key_deg: torch.Tensor,
    masks: BlockDiffMasks,
    is_masked_token: torch.Tensor,
    selector: SelectorConfig,
    scaling: float,
    n_rep: int,
) -> torch.Tensor:
    """Per-(KV head, query block) top-k over the prefix → [Hkv, S, S] keep mask.

    Reproduces ``reference.selector_importance_reference`` with
    ``domain="prefix"`` and ``score="softmax"``: softmax the selector queries'
    logits over the old-cache keys, average over the query heads sharing a KV
    head and over the representatives, then take the top-k per KV head.
    """
    b, hkv, s, _ = key_deg.shape
    device = query.device
    keep = torch.ones(hkv, s, s, dtype=torch.bool, device=device)
    q_all = query.reshape(b, hkv, n_rep, s, query.shape[-1])

    for j in torch.unique(masks.block_of[masks.is_xt]).tolist():
        rows = (masks.block_of == j) & masks.is_xt
        if not bool(rows.any()):
            continue
        row_idx = torch.nonzero(rows, as_tuple=True)[0]
        block_start = int(row_idx[0])

        cols = masks.old_pair[block_start]
        n_old = int(cols.sum())
        if n_old == 0:
            continue
        k_eff = selector.effective_topk(n_old)
        if k_eff >= n_old:
            continue

        # Representatives: the masked positions this block's selector would use.
        masked_rel = [
            int(p) - block_start
            for p in row_idx.tolist()
            if bool(is_masked_token[p])
        ]
        if not masked_rel:
            masked_rel = list(range(len(row_idx)))
        rel = selector.query_indices(masked_rel, len(row_idx))
        if not rel:
            continue
        sel_rows = row_idx[torch.as_tensor(rel, device=device, dtype=torch.long)]
        col_idx = torch.nonzero(cols, as_tuple=True)[0]

        q = q_all[:, :, :, sel_rows, :].float()                  # [b,hkv,g,m,d]
        k_old = key_deg[:, :, col_idx, :].float()                # [b,hkv,n_old,d]
        logits = torch.einsum("bhgmd,bhnd->bhgmn", q, k_old) * scaling
        importance = torch.softmax(logits, dim=-1).mean(dim=(2, 3))[0]   # [hkv,n_old]

        top = importance.topk(k_eff, dim=-1).indices
        block_keep = torch.zeros(hkv, n_old, dtype=torch.bool, device=device)
        block_keep.scatter_(1, top, True)

        tmp = keep[:, row_idx]                               # [hkv, rows, S]
        tmp[:, :, col_idx] = tmp[:, :, col_idx] & block_keep[:, None, :]
        keep[:, row_idx] = tmp
    return keep


def blockdiff_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    masks: BlockDiffMasks,
    scaling: float,
    n_rep: int,
    degrade: CacheDegradation,
    selector: SelectorConfig | None,
    is_masked_token: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Attention where reads that would hit the packed cache use a degraded copy.

    ``query`` [B, Hq, S, D]; ``key``/``value`` [B, Hkv, S, D]; S = 2L.
    With no degradation and no selector this is plain SDPA under the upstream mask.
    """
    if not degrade.active and selector is None:
        out = F.scaled_dot_product_attention(
            query, _repeat_kv(key, n_rep), _repeat_kv(value, n_rep),
            attn_mask=masks.visible[None, None], scale=scaling, is_causal=False,
        )
        return out.transpose(1, 2).contiguous()

    q32 = query.float()
    scores = torch.matmul(q32, _repeat_kv(key, n_rep).float().transpose(-2, -1)) * scaling

    key_deg = key
    if degrade.degrades_keys and bool(masks.old_pair.any()):
        key_deg = degrade_keys(key, degrade, generator=generator)
        scores_c = torch.matmul(
            q32, _repeat_kv(key_deg, n_rep).float().transpose(-2, -1)
        ) * scaling
        scores = torch.where(masks.old_pair[None, None], scores_c, scores)
        del scores_c

    scores = scores.masked_fill(~masks.visible[None, None], float("-inf"))

    if selector is not None and bool(masks.old_pair.any()):
        keep = select_keep_mask(
            query, key_deg, masks, is_masked_token, selector, scaling, n_rep
        )
        # Selection is per KV head; expand it the same way repeat_kv expands K/V.
        keep_q = keep.repeat_interleave(n_rep, dim=0)[None]   # [1, Hq, S, S]
        scores = scores.masked_fill(~keep_q, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    del scores
    v_ref = _repeat_kv(value, n_rep).float()

    if degrade.degrades_values and bool(masks.old_pair.any()):
        v_deg = _repeat_kv(degrade_values(value, degrade, generator=generator), n_rep).float()
        m = masks.old_pair[None, None].to(probs.dtype)
        out = torch.matmul(probs * (1.0 - m), v_ref) + torch.matmul(probs * m, v_deg)
    else:
        out = torch.matmul(probs, v_ref)

    return out.to(query.dtype).transpose(1, 2).contiguous()


# --------------------------------------------------------------------------
# model forward
# --------------------------------------------------------------------------
def _rope_fn(base_model):
    """``apply_rotary_pos_emb`` from the checkpoint's own remote-code module."""
    import sys

    return sys.modules[type(base_model.model.layers[0].self_attn).__module__].apply_rotary_pos_emb


def _attention_forward(attn, hidden_states, cos, sin, masks, degrade, selector,
                       is_masked_token, apply_rope, generator):
    """Reimplementation of ``Fast_dLLM_QwenAttention.forward`` (training geometry).

    RoPE is applied to the two halves of the doubled sequence independently with
    the same cos/sin, so ``x_t[p]`` and ``x_0[p]`` share position p — as upstream
    does in its own training path.
    """
    b, s, _ = hidden_states.shape
    hd = attn.head_dim
    q = attn.q_proj(hidden_states).view(b, s, -1, hd).transpose(1, 2)
    k = attn.k_proj(hidden_states).view(b, s, -1, hd).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(b, s, -1, hd).transpose(1, 2)

    half = s // 2
    q1, k1 = apply_rope(q[:, :, :half], k[:, :, :half], cos, sin)
    q2, k2 = apply_rope(q[:, :, half:], k[:, :, half:], cos, sin)
    q = torch.cat((q1, q2), dim=-2)
    k = torch.cat((k1, k2), dim=-2)

    out = blockdiff_attention(
        q, k, v, masks, attn.scaling, attn.num_key_value_groups,
        degrade, selector, is_masked_token, generator,
    )
    return attn.o_proj(out.reshape(b, s, -1))


def _layer_forward(layer, hidden_states, cos, sin, masks, degrade, selector,
                   is_masked_token, apply_rope, generator):
    residual = hidden_states
    h = _attention_forward(
        layer.self_attn, layer.input_layernorm(hidden_states), cos, sin,
        masks, degrade, selector, is_masked_token, apply_rope, generator,
    )
    hidden_states = residual + h
    return hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))


def blockdiff_logits(
    model,
    input_ids: torch.Tensor,
    *,
    masks: BlockDiffMasks,
    degrade: CacheDegradation,
    selector: SelectorConfig | None,
    is_masked_token: torch.Tensor,
    gradient_checkpointing: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Logits over the x_t half of a doubled ``[x_t ; x_0]`` sequence.

    ``input_ids``: [B, 2L] -> [B, L, vocab]. The caller applies the upstream
    one-position shift (token p is predicted from the logit at p-1).
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    half = input_ids.shape[1] // 2
    apply_rope = _rope_fn(base)

    h = base.model.embed_tokens(input_ids)
    position_ids = torch.arange(half, device=input_ids.device).unsqueeze(0)
    cos, sin = base.model.rotary_emb(h, position_ids)

    for layer in base.model.layers:
        args = (layer, h, cos, sin, masks, degrade, selector,
                is_masked_token, apply_rope, generator)
        if gradient_checkpointing and torch.is_grad_enabled():
            h = checkpoint(_layer_forward, *args, use_reentrant=False)
        else:
            h = _layer_forward(*args)

    return base.lm_head(base.model.norm(h)[:, :half])

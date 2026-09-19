"""Training forward for long contexts: a cached prefix plus one answer window.

LongBench prompts run 5-64k tokens while the answers are 4-76. The doubled
``[x_t ; x_0]`` forward in ``blockdiff.py`` is quadratic in the whole sequence,
which at 17k tokens would be ~130 GB of attention scores per layer -- and it
would spend all of it on positions that carry no supervision.

So the long-context path takes the shape inference already has: encode the
prompt once into a cache, then decode the answer window against it. Concretely

  1. the prefix is forwarded under ``no_grad`` with the model's own efficient
     path, giving per-layer K/V exactly as the commit forwards would;
  2. those are degraded the way the packed cache degrades them;
  3. only the answer window runs with gradient, attending to
     ``[prefix ; x_t ; x_0]``.

The prefix is therefore **detached**: the adapter learns to *read* a degraded
cache, not to *write* a more robust one. That is a real limitation of this path
and the reason the MATH track keeps the full doubled forward, where gradient
does reach the prefix. Scores here are [2A, P + 2A] -- 122 MB at A=32, P=17k.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from ..config import SelectorConfig
from .blockdiff import _repeat_kv, _rope_fn
from .degrade import CacheDegradation, degrade_keys, degrade_values


@dataclass
class PrefixMasks:
    """Boolean masks over queries [2A] x keys [P + 2A]."""

    visible: torch.Tensor
    old_pair: torch.Tensor     # reads served by the packed cache
    block_of: torch.Tensor     # [2A] answer-block index per query
    is_xt: torch.Tensor        # [2A]
    prefix_len: int
    answer_len: int


def build_prefix_masks(
    answer_len: int,
    block_size: int,
    prefix_len: int,
    device,
    *,
    propagate_to_x0: bool = True,
) -> PrefixMasks:
    """Block-diffusion mask for an answer window sitting on a cached prefix.

    The prefix is one contiguous run of already-decoded tokens, so it is
    strictly earlier than every answer block and every query may read it --
    always out of the cache, hence always degraded.
    """
    a, p = answer_len, prefix_len
    qi = torch.arange(2 * a, device=device)
    is_x0_q = qi >= a
    bq = torch.where(is_x0_q, (qi - a) // block_size, qi // block_size)

    ki = torch.arange(p + 2 * a, device=device)
    is_prefix_k = ki < p
    rel = ki - p
    is_x0_k = (~is_prefix_k) & (rel >= a)
    bk = torch.where(is_x0_k, (rel - a) // block_size, rel // block_size)

    BQ, BK = bq[:, None], bk[None, :]
    X0Q, X0K, PK = is_x0_q[:, None], is_x0_k[None, :], is_prefix_k[None, :]

    own_block = (BQ == BK) & ~PK & (X0Q == X0K)
    xt_reads_x0 = (BQ > BK) & X0K & ~X0Q
    x0_reads_x0 = (BQ >= BK) & X0K & X0Q
    visible = PK | own_block | xt_reads_x0 | x0_reads_x0

    old_pair = PK.expand(2 * a, p + 2 * a).clone() | xt_reads_x0
    if propagate_to_x0:
        old_pair = old_pair | ((BQ > BK) & X0K & X0Q)

    return PrefixMasks(
        visible=visible,
        old_pair=old_pair,
        block_of=bq,
        is_xt=~is_x0_q,
        prefix_len=p,
        answer_len=a,
    )


@torch.no_grad()
def select_prefix_keep(
    query: torch.Tensor,
    key_deg: torch.Tensor,
    masks: PrefixMasks,
    is_masked_token: torch.Tensor,
    selector: SelectorConfig,
    scaling: float,
    n_rep: int,
    block_size: int,
) -> torch.Tensor:
    """Per-(KV head, answer block) top-k over everything the cache serves.

    Same rule as the short-context path and as
    ``reference.selector_importance_reference`` with ``domain="prefix"``; the
    only difference is that the candidate set now spans the cached prompt as
    well as the answer blocks already decoded.
    """
    b, hkv, n_keys, d = key_deg.shape
    device = query.device
    n_q = masks.visible.shape[0]
    keep = torch.ones(hkv, n_q, n_keys, dtype=torch.bool, device=device)
    q_all = query.reshape(b, hkv, n_rep, n_q, d)

    for j in torch.unique(masks.block_of[masks.is_xt]).tolist():
        rows = (masks.block_of == j) & masks.is_xt
        row_idx = torch.nonzero(rows, as_tuple=True)[0]
        if row_idx.numel() == 0:
            continue
        start = int(row_idx[0])

        cols = masks.old_pair[start] & masks.visible[start]
        n_old = int(cols.sum())
        if n_old == 0:
            continue
        k_eff = selector.effective_topk(n_old)
        if k_eff >= n_old:
            continue

        masked_rel = [int(p) - start for p in row_idx.tolist() if bool(is_masked_token[p])]
        if not masked_rel:
            masked_rel = list(range(row_idx.numel()))
        rel = selector.query_indices(masked_rel, block_size)
        if not rel:
            continue
        sel_rows = row_idx[torch.as_tensor(rel, device=device, dtype=torch.long)]
        col_idx = torch.nonzero(cols, as_tuple=True)[0]

        q = q_all[:, :, :, sel_rows, :].float()
        k_old = key_deg[:, :, col_idx, :].float()
        logits = torch.einsum("bhgmd,bhnd->bhgmn", q, k_old) * scaling
        importance = torch.softmax(logits, dim=-1).mean(dim=(2, 3))[0]

        top = importance.topk(k_eff, dim=-1).indices
        block_keep = torch.zeros(hkv, n_old, dtype=torch.bool, device=device)
        block_keep.scatter_(1, top, True)

        tmp = keep[:, row_idx]
        tmp[:, :, col_idx] = tmp[:, :, col_idx] & block_keep[:, None, :]
        keep[:, row_idx] = tmp
    return keep


def _degrade_cache_keys(
    k_prefix: torch.Tensor,
    k_answer: torch.Tensor,
    answer_len: int,
    degrade: CacheDegradation,
) -> torch.Tensor:
    """Degrade the parts the cache serves, each on its own group alignment.

    Quantization groups run along the token axis from the start of the cache.
    Degrading the concatenation would straddle x_t tokens -- which never enter
    the cache -- into the prefix's groups, so the two runs are degraded apart
    and only then joined.
    """
    k_pre = degrade_keys(k_prefix, degrade)
    k_x0 = degrade_keys(k_answer[:, :, answer_len:, :], degrade)
    return torch.cat([k_pre, k_answer[:, :, :answer_len, :], k_x0], dim=2)


def _degrade_cache_values(
    v_prefix: torch.Tensor,
    v_answer: torch.Tensor,
    answer_len: int,
    degrade: CacheDegradation,
) -> torch.Tensor:
    v_pre = degrade_values(v_prefix, degrade)
    v_x0 = degrade_values(v_answer[:, :, answer_len:, :], degrade)
    return torch.cat([v_pre, v_answer[:, :, :answer_len, :], v_x0], dim=2)


def cached_prefix_attention(
    query: torch.Tensor,
    key_answer: torch.Tensor,
    value_answer: torch.Tensor,
    key_prefix: torch.Tensor,
    value_prefix: torch.Tensor,
    masks: PrefixMasks,
    scaling: float,
    n_rep: int,
    degrade: CacheDegradation,
    selector: SelectorConfig | None,
    is_masked_token: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Answer-window attention over ``[prefix ; x_t ; x_0]``."""
    a = masks.answer_len
    k_all = torch.cat([key_prefix, key_answer], dim=2)
    v_all = torch.cat([value_prefix, value_answer], dim=2)

    q32 = query.float()
    scores = torch.matmul(q32, _repeat_kv(k_all, n_rep).float().transpose(-2, -1)) * scaling

    k_for_select = k_all
    if degrade.degrades_keys:
        k_for_select = _degrade_cache_keys(key_prefix, key_answer, a, degrade)
        scores_c = torch.matmul(
            q32, _repeat_kv(k_for_select, n_rep).float().transpose(-2, -1)
        ) * scaling
        scores = torch.where(masks.old_pair[None, None], scores_c, scores)
        del scores_c

    scores = scores.masked_fill(~masks.visible[None, None], float("-inf"))

    if selector is not None:
        keep = select_prefix_keep(
            query, k_for_select, masks, is_masked_token, selector,
            scaling, n_rep, block_size,
        )
        scores = scores.masked_fill(~keep.repeat_interleave(n_rep, dim=0)[None], float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    del scores
    v_ref = _repeat_kv(v_all, n_rep).float()

    if degrade.degrades_values:
        v_deg = _repeat_kv(
            _degrade_cache_values(value_prefix, value_answer, a, degrade), n_rep
        ).float()
        m = masks.old_pair[None, None].to(probs.dtype)
        out = torch.matmul(probs * (1.0 - m), v_ref) + torch.matmul(probs * m, v_deg)
    else:
        out = torch.matmul(probs, v_ref)

    return out.to(query.dtype).transpose(1, 2).contiguous()


# --------------------------------------------------------------------------
# model forward
# --------------------------------------------------------------------------
@torch.no_grad()
def encode_prefix(model, prefix_ids: torch.Tensor, block_size: int) -> list[tuple]:
    """Per-layer (K, V) for the prompt, via the model's own efficient forward.

    This is the commit path: the same block-causal encoding the decoder uses to
    put the prompt into the cache. Run without gradient -- see the module
    docstring for what that costs.
    """
    if prefix_ids.shape[1] % block_size:
        raise ValueError(
            f"prefix length {prefix_ids.shape[1]} must be a multiple of block_size {block_size}"
        )
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    out = base.forward(
        input_ids=prefix_ids, use_cache=True,
        update_past_key_values=True, block_size=block_size,
    )
    pkv = out.past_key_values
    return [
        (pkv.key_cache[i].detach(), pkv.value_cache[i].detach())
        for i in range(len(pkv.key_cache))
    ]


def _layer_forward(
    layer, hidden_states, cos, sin, k_prefix, v_prefix, masks,
    degrade, selector, is_masked_token, apply_rope, block_size,
):
    residual = hidden_states
    h = layer.input_layernorm(hidden_states)

    attn = layer.self_attn
    b, s, _ = h.shape
    hd = attn.head_dim
    q = attn.q_proj(h).view(b, s, -1, hd).transpose(1, 2)
    k = attn.k_proj(h).view(b, s, -1, hd).transpose(1, 2)
    v = attn.v_proj(h).view(b, s, -1, hd).transpose(1, 2)

    # x_t[p] and x_0[p] share position p, as upstream's training path does.
    half = s // 2
    q1, k1 = apply_rope(q[:, :, :half], k[:, :, :half], cos, sin)
    q2, k2 = apply_rope(q[:, :, half:], k[:, :, half:], cos, sin)
    q = torch.cat((q1, q2), dim=-2)
    k = torch.cat((k1, k2), dim=-2)

    out = cached_prefix_attention(
        q, k, v, k_prefix, v_prefix, masks, attn.scaling,
        attn.num_key_value_groups, degrade, selector, is_masked_token, block_size,
    )
    hidden_states = residual + attn.o_proj(out.reshape(b, s, -1))
    return hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))


def cached_prefix_logits(
    model,
    answer_ids: torch.Tensor,
    prefix_cache: list[tuple],
    *,
    masks: PrefixMasks,
    degrade: CacheDegradation,
    selector: SelectorConfig | None,
    is_masked_token: torch.Tensor,
    block_size: int,
    gradient_checkpointing: bool = True,
) -> torch.Tensor:
    """Logits over the x_t half of a doubled answer window on a cached prefix.

    ``answer_ids``: [B, 2A] -> [B, A, vocab]. The caller applies the upstream
    one-position shift.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    half = answer_ids.shape[1] // 2
    apply_rope = _rope_fn(base)

    h = base.model.embed_tokens(answer_ids)
    # The answer window sits immediately after the prefix, so it carries the
    # absolute positions the decoder would give it.
    position_ids = torch.arange(
        masks.prefix_len, masks.prefix_len + half, device=answer_ids.device
    ).unsqueeze(0)
    cos, sin = base.model.rotary_emb(h, position_ids)

    for i, layer in enumerate(base.model.layers):
        k_pre, v_pre = prefix_cache[i]
        args = (layer, h, cos, sin, k_pre, v_pre, masks, degrade,
                selector, is_masked_token, apply_rope, block_size)
        if gradient_checkpointing and torch.is_grad_enabled():
            h = checkpoint(_layer_forward, *args, use_reentrant=False)
        else:
            h = _layer_forward(*args)

    return base.lm_head(base.model.norm(h)[:, :half])

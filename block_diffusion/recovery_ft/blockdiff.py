"""Grad-enabled Fast-dLLM-v2 block-diffusion forward with old-cache corruption.

Why a hand-written forward
--------------------------
The upstream model has two mutually exclusive paths:

* ``model.training == True``  — builds the doubled ``[x_t ; x_0]`` sequence with
  its *own* random masking, and runs ``flex_attention`` under
  ``torch.compile(fullgraph=True)``.  We cannot control the noise (teacher and
  student would see different masks) and cannot intercept the KV.
* ``model.training == False`` — plain block-causal ``eval_mask`` over a single
  sequence through ``ALL_ATTENTION_FUNCTIONS["sdpa"]``.  Interceptable, but the
  context of block *j* would be the *noisy* tokens of blocks < j, whereas at
  inference those blocks are already decoded and clean.

This module reimplements the doubled formulation (which is the one that matches
inference: an ``x_t`` query in block *j* sees its own noisy block plus the
**clean** ``x_0`` of blocks < j) on top of the model's own submodules, so the
KV that an ``x_t`` query reads from earlier blocks — exactly the tensors that
live in the quantized/sparse cache at inference — can be corrupted.

Corruption geometry (block_size == bd_size == 32 == KIVI group_size)
-------------------------------------------------------------------
Cache groups are cut from position 0 in steps of ``group_size``, so group *i*
is exactly block *i*.  With ``kivi_residual_length = 32`` the cache seen by
block *j* has blocks ``0 .. j-2`` quantized and block ``j-1`` still in bf16,
which is what ``residual_k_blocks=1`` encodes below.  Values carry
``residual_length = 0`` upstream, so every visible old block is quantized.

Parity against the real decoder is checked by ``test_parity.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from quant_noise import kivi_key_sigma, kivi_value_sigma

_KVQ = Path(__file__).resolve().parent.parent / "kv_quant"
if str(_KVQ) not in sys.path:
    sys.path.insert(0, str(_KVQ))
from kv_cache_quant import quantize_key_roundtrip  # noqa: E402

# The only fp32 matmuls in this pipeline are the attention scores below (model
# weights are bf16, so this flag does not touch them). TF32 makes the manual
# kernel ~10x tighter against an fp32 reference than fused SDPA is, so the
# corruption branches are not paying for a sloppier attention kernel.
torch.backends.cuda.matmul.allow_tf32 = True


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------
def _block_ids(seq_len: int, block_size: int, device) -> torch.Tensor:
    return torch.arange(seq_len, device=device) // block_size


@dataclass
class BlockDiffMasks:
    """Boolean [S, S] masks over the doubled sequence S = 2L."""

    visible: torch.Tensor        # upstream block_diff_mask
    old_pair: torch.Tensor       # x_t query reading x_0 key of a strictly earlier block
    k_corrupt: torch.Tensor      # subset of old_pair whose K is quantized (past residual)
    v_corrupt: torch.Tensor      # subset of old_pair whose V is quantized
    block_q: torch.Tensor        # [S] block index per position
    is_xt: torch.Tensor          # [S] True on the x_t half


def build_masks(
    seq_len: int,
    block_size: int,
    device,
    *,
    residual_k_blocks: int = 1,
    residual_v_blocks: int = 0,
    propagate_to_x0: bool = False,
) -> BlockDiffMasks:
    """Build the upstream block-diffusion mask plus the corruption-role masks.

    ``seq_len`` is L (the un-doubled length); the returned masks are [2L, 2L].
    ``propagate_to_x0=True`` also corrupts the x_0 -> x_0 reads, i.e. lets the
    quantization error compound into the cache itself the way it does in a real
    streaming deployment.
    """
    n = seq_len
    idx = torch.arange(2 * n, device=device)
    is_x0 = idx >= n
    blk = torch.where(is_x0, (idx - n) // block_size, idx // block_size)

    bq = blk[:, None]
    bk = blk[None, :]
    x0_q = is_x0[:, None]
    x0_k = is_x0[None, :]

    block_diagonal = (bq == bk) & (x0_q == x0_k)
    offset_block_causal = (bq > bk) & x0_k & ~x0_q
    block_causal = (bq >= bk) & x0_k & x0_q
    visible = block_diagonal | offset_block_causal | block_causal

    old_pair = offset_block_causal.clone()
    if propagate_to_x0:
        old_pair = old_pair | ((bq > bk) & x0_k & x0_q)

    k_corrupt = old_pair & (bk <= bq - 1 - residual_k_blocks)
    v_corrupt = old_pair & (bk <= bq - 1 - residual_v_blocks)

    return BlockDiffMasks(
        visible=visible,
        old_pair=old_pair,
        k_corrupt=k_corrupt,
        v_corrupt=v_corrupt,
        block_q=blk,
        is_xt=~is_x0,
    )


# --------------------------------------------------------------------------
# corruption / sparsity configuration
# --------------------------------------------------------------------------
@dataclass
class CorruptionConfig:
    """How the old cache is degraded inside the training forward.

    ``mode="noise"``  — additive Gaussian with the measured quantization-error
                        variance (cheap, smooth, the branch-D/E augmentation).
    ``mode="quant"``  — the real KIVI quantize->dequantize, with a
                        straight-through estimator so gradients still flow.
                        Exactly what inference does, and the reference the
                        noise surrogate is validated against.
    """

    mode: str = "noise"
    # Bit width of the degraded cache; None = exact cache.
    k_bits: int | None = None
    v_bits: int | None = None
    kivi_group_size: int = 32
    residual_k_blocks: int = 1
    residual_v_blocks: int = 0
    # Per-head top-k over old-cache tokens (branch E). None = dense old cache.
    topk: int | None = None
    topk_pct: float | None = None
    # Let quantization error compound into the cache (x_0 -> x_0 reads too).
    propagate_to_x0: bool = False
    # Per-tensor correction from calibrate_noise.py (measured/analytic std), so
    # the injected variance equals the *measured* quantization error, not just
    # the uniform-rounding prediction. 1.0 = use the analytic sigma as-is.
    k_noise_scale: float = 1.0
    v_noise_scale: float = 1.0
    # Diagnostics: take the manual attention kernel even with nothing to corrupt.
    force_manual: bool = False

    @property
    def active(self) -> bool:
        return (
            self.force_manual
            or self.k_bits is not None
            or self.v_bits is not None
            or self.topk is not None
            or self.topk_pct is not None
        )

    def effective_topk(self, num_old: int) -> int | None:
        if num_old <= 0:
            return None
        if self.topk_pct is not None:
            return max(1, min(num_old, int(round(num_old * self.topk_pct / 100.0))))
        if self.topk is not None:
            return min(self.topk, num_old)
        return None


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


@torch.no_grad()
def _topk_old_cache_mask(
    scores: torch.Tensor,
    masks: BlockDiffMasks,
    is_masked_token: torch.Tensor,
    cfg: CorruptionConfig,
) -> torch.Tensor:
    """Per-(head, query block) top-k over visible old-cache keys → [H, S, S] bool keep.

    Reproduces the inference selector (``selector.all_mean`` + ``per_head``):
    rank old-cache tokens by the attention they receive, averaged over the
    *masked* queries of the block, then keep the top-k for that head.
    """
    h = scores.shape[1]
    s = scores.shape[-1]
    device = scores.device
    keep = torch.ones(h, s, s, dtype=torch.bool, device=device)

    probs = torch.softmax(scores.float().masked_fill(~masks.visible, float("-inf")), dim=-1)[0]
    blocks = masks.block_q
    xt_blocks = torch.unique(blocks[masks.is_xt])

    for j in xt_blocks.tolist():
        q_rows = (blocks == j) & masks.is_xt & is_masked_token
        if not bool(q_rows.any()):
            q_rows = (blocks == j) & masks.is_xt
            if not bool(q_rows.any()):
                continue
        # Old-cache columns visible to this block (identical for all its queries).
        first_q = int(torch.nonzero(q_rows, as_tuple=True)[0][0])
        cols = masks.old_pair[first_q]
        num_old = int(cols.sum())
        k_eff = cfg.effective_topk(num_old)
        if k_eff is None or k_eff >= num_old:
            continue
        col_idx = torch.nonzero(cols, as_tuple=True)[0]
        imp = probs[:, q_rows][:, :, col_idx].mean(dim=1)          # [H, num_old]
        top = imp.topk(k_eff, dim=-1).indices                      # [H, k]
        block_keep = torch.zeros(h, num_old, dtype=torch.bool, device=device)
        block_keep.scatter_(1, top, True)
        rows = torch.nonzero(q_rows, as_tuple=True)[0]
        sub = keep[:, rows][:, :, col_idx]
        sub &= block_keep[:, None, :]
        tmp = keep[:, rows]
        tmp[:, :, col_idx] = sub
        keep[:, rows] = tmp
    return keep


def _degrade_key(key: torch.Tensor, cfg: CorruptionConfig, generator) -> torch.Tensor:
    """Old-cache keys as the student reads them: KIVI noise surrogate or real quant.

    ``residual_length=0`` here because the residual window (the bf16 tail of the
    cache) is expressed by ``BlockDiffMasks.k_corrupt``, not by the quantizer.
    """
    if cfg.mode == "quant":
        q = quantize_key_roundtrip(
            key.float(), cfg.k_bits, scheme="kivi",
            group_size=cfg.kivi_group_size, residual_length=0,
        ).to(key.dtype)
        return key + (q - key).detach()          # straight-through estimator
    sigma = kivi_key_sigma(
        key, cfg.k_bits, group_size=cfg.kivi_group_size, residual_length=0
    ) * cfg.k_noise_scale
    noise = torch.randn(key.shape, device=key.device, dtype=torch.float32, generator=generator)
    return key + (noise * sigma).to(key.dtype)


def _degrade_value(value: torch.Tensor, cfg: CorruptionConfig, generator) -> torch.Tensor:
    """Old-cache values as the student reads them (per-token quant, no residual)."""
    if cfg.mode == "quant":
        from quantization import apply_precision

        q = apply_precision(value.float(), cfg.v_bits, "v_per_token").to(value.dtype)
        return value + (q - value).detach()      # straight-through estimator
    sigma = kivi_value_sigma(value, cfg.v_bits, residual_length=0) * cfg.v_noise_scale
    noise = torch.randn(value.shape, device=value.device, dtype=torch.float32, generator=generator)
    return value + (noise * sigma).to(value.dtype)


def blockdiff_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    masks: BlockDiffMasks,
    scaling: float,
    n_rep: int,
    cfg: CorruptionConfig,
    is_masked_token: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Attention where old-cache reads may use a corrupted K/V copy.

    ``query`` [B, Hq, S, D]; ``key``/``value`` [B, Hkv, S, D]; S = 2L.
    With ``cfg.active == False`` this is plain SDPA under the upstream mask.
    """
    if not cfg.active:
        out = F.scaled_dot_product_attention(
            query,
            _repeat_kv(key, n_rep),
            _repeat_kv(value, n_rep),
            attn_mask=masks.visible[None, None],
            scale=scaling,
            is_causal=False,
        )
        return out.transpose(1, 2).contiguous()

    q32 = query.float()
    k_rep = _repeat_kv(key, n_rep).float()
    scores = torch.matmul(q32, k_rep.transpose(-2, -1)) * scaling

    if cfg.k_bits is not None and bool(masks.k_corrupt.any()):
        k_bad = _degrade_key(key, cfg, generator)
        scores_c = torch.matmul(q32, _repeat_kv(k_bad, n_rep).float().transpose(-2, -1)) * scaling
        scores = torch.where(masks.k_corrupt[None, None], scores_c, scores)
        del scores_c

    scores = scores.masked_fill(~masks.visible[None, None], float("-inf"))

    if cfg.topk is not None or cfg.topk_pct is not None:
        keep = _topk_old_cache_mask(scores, masks, is_masked_token, cfg)
        scores = scores.masked_fill(~keep[None], float("-inf"))

    # Keep the probabilities in fp32 through the PV product: rounding them to
    # bf16 first costs ~2x accuracy versus fused SDPA, which compounds over 28
    # layers and would show up as a train/inference kernel mismatch.
    probs = torch.softmax(scores, dim=-1)
    del scores
    v_ref = _repeat_kv(value, n_rep).float()

    if cfg.v_bits is not None and bool(masks.v_corrupt.any()):
        v_noisy = _repeat_kv(_degrade_value(value, cfg, generator), n_rep).float()
        vmask = masks.v_corrupt[None, None].to(probs.dtype)
        out = torch.matmul(probs * (1.0 - vmask), v_ref) + torch.matmul(probs * vmask, v_noisy)
    else:
        out = torch.matmul(probs, v_ref)

    return out.to(query.dtype).transpose(1, 2).contiguous()


# --------------------------------------------------------------------------
# model forward
# --------------------------------------------------------------------------
def _rope_fns(model):
    """Grab ``apply_rotary_pos_emb`` from the model's own remote-code module."""
    import sys

    mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
    return mod.apply_rotary_pos_emb


def _attention_forward(
    attn,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    masks: BlockDiffMasks,
    cfg: CorruptionConfig,
    is_masked_token: torch.Tensor,
    apply_rope,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Reimplementation of ``Fast_dLLM_QwenAttention.forward`` (training geometry).

    Matches upstream: RoPE is applied to the two halves of the doubled sequence
    independently, with the same cos/sin, so ``x_t[p]`` and ``x_0[p]`` share
    position p.
    """
    b, s, _ = hidden_states.shape
    hd = attn.head_dim
    q = attn.q_proj(hidden_states).view(b, s, -1, hd).transpose(1, 2)
    k = attn.k_proj(hidden_states).view(b, s, -1, hd).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(b, s, -1, hd).transpose(1, 2)

    half = s // 2
    q1, q2 = q[:, :, :half], q[:, :, half:]
    k1, k2 = k[:, :, :half], k[:, :, half:]
    q1, k1 = apply_rope(q1, k1, cos, sin)
    q2, k2 = apply_rope(q2, k2, cos, sin)
    q = torch.cat((q1, q2), dim=-2)
    k = torch.cat((k1, k2), dim=-2)

    out = blockdiff_attention(
        q, k, v, masks, attn.scaling, attn.num_key_value_groups,
        cfg, is_masked_token, generator,
    )
    return attn.o_proj(out.reshape(b, s, -1))


def _layer_forward(
    layer,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    masks: BlockDiffMasks,
    cfg: CorruptionConfig,
    is_masked_token: torch.Tensor,
    apply_rope,
    generator: torch.Generator | None,
) -> torch.Tensor:
    residual = hidden_states
    h = layer.input_layernorm(hidden_states)
    h = _attention_forward(
        layer.self_attn, h, cos, sin, masks, cfg, is_masked_token, apply_rope, generator
    )
    hidden_states = residual + h
    residual = hidden_states
    h = layer.post_attention_layernorm(hidden_states)
    h = layer.mlp(h)
    return residual + h


def blockdiff_logits(
    model,
    input_ids: torch.Tensor,
    *,
    masks: BlockDiffMasks,
    cfg: CorruptionConfig,
    is_masked_token: torch.Tensor,
    gradient_checkpointing: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Logits over the x_t half of a doubled ``[x_t ; x_0]`` sequence.

    ``input_ids``: [B, 2L].  Returns [B, L, vocab].  Caller applies the upstream
    one-position shift (token p is predicted from the logit at p-1).
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    half = input_ids.shape[1] // 2
    apply_rope = _rope_fns(base)

    h = base.model.embed_tokens(input_ids)
    position_ids = torch.arange(half, device=input_ids.device).unsqueeze(0)
    cos, sin = base.model.rotary_emb(h, position_ids)

    for layer in base.model.layers:
        if gradient_checkpointing and torch.is_grad_enabled():
            h = checkpoint(
                _layer_forward,
                layer, h, cos, sin, masks, cfg, is_masked_token, apply_rope, generator,
                use_reentrant=False,
            )
        else:
            h = _layer_forward(
                layer, h, cos, sin, masks, cfg, is_masked_token, apply_rope, generator
            )

    h = base.model.norm(h)[:, :half]
    return base.lm_head(h)

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class PackedKeys:
    payload: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    bits: int
    token_group: int
    tokens: int


@dataclass(slots=True)
class PackedValues:
    payload: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    bits: int
    channel_group: int
    tokens: int


def values_per_byte(bits: int) -> int:
    if bits not in (2, 4):
        raise ValueError(f"packed bit width must be 2 or 4, got {bits}")
    return 8 // bits


def pack_last_dim(q: torch.Tensor, bits: int) -> torch.Tensor:
    if q.dtype != torch.uint8:
        q = q.to(torch.uint8)
    vpb = values_per_byte(bits)
    if q.shape[-1] % vpb:
        raise ValueError(f"last dimension {q.shape[-1]} is not divisible by {vpb}")
    shape = (*q.shape[:-1], q.shape[-1] // vpb, vpb)
    x = q.reshape(shape).to(torch.int16)
    shifts = torch.arange(vpb, device=q.device, dtype=torch.int16) * bits
    packed = torch.sum(x << shifts, dim=-1)
    return packed.to(torch.uint8)


def unpack_last_dim(payload: torch.Tensor, bits: int, unpacked_size: int) -> torch.Tensor:
    vpb = values_per_byte(bits)
    if unpacked_size != payload.shape[-1] * vpb:
        raise ValueError("unpacked_size does not match payload shape")
    shifts = torch.arange(vpb, device=payload.device, dtype=torch.int16) * bits
    x = payload.to(torch.int16).unsqueeze(-1)
    q = (x >> shifts) & ((1 << bits) - 1)
    return q.reshape(*payload.shape[:-1], unpacked_size).to(torch.uint8)


def quantize_keys(
    key: torch.Tensor,
    *,
    bits: int = 4,
    token_group: int = 32,
    param_dtype: torch.dtype = torch.float16,
) -> PackedKeys:
    if key.ndim != 4:
        raise ValueError("key must have shape [B, Hkv, T, D]")
    if bits not in (2, 4):
        raise ValueError("quantize_keys supports 2- or 4-bit payloads")
    b, h, t, d = key.shape
    if t % token_group:
        raise ValueError(f"T={t} must be divisible by token_group={token_group}")
    if token_group % values_per_byte(bits):
        raise ValueError("token_group is incompatible with bit width")

    ng = t // token_group

    x = key.reshape(b, h, ng, token_group, d).permute(0, 1, 2, 4, 3)
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / float((1 << bits) - 1)).clamp_min(1e-8)
    q = torch.round((x - mn) / scale).clamp_(0, (1 << bits) - 1).to(torch.uint8)
    payload = pack_last_dim(q, bits).contiguous()
    return PackedKeys(
        payload=payload,
        scale=scale.squeeze(-1).to(param_dtype).contiguous(),
        zero=mn.squeeze(-1).to(param_dtype).contiguous(),
        bits=bits,
        token_group=token_group,
        tokens=t,
    )


def dequantize_keys(packed: PackedKeys, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    q = unpack_last_dim(packed.payload, packed.bits, packed.token_group)
    x = q.to(torch.float32) * packed.scale.float().unsqueeze(-1)
    x = x + packed.zero.float().unsqueeze(-1)

    x = x.permute(0, 1, 2, 4, 3).reshape(
        packed.payload.shape[0],
        packed.payload.shape[1],
        packed.tokens,
        packed.scale.shape[-1],
    )
    return x.to(dtype)


def quantize_values(
    value: torch.Tensor,
    *,
    bits: int = 4,
    channel_group: int = 32,
    param_dtype: torch.dtype = torch.float16,
) -> PackedValues:
    if value.ndim != 4:
        raise ValueError("value must have shape [B, Hkv, T, D]")
    if bits not in (2, 4):
        raise ValueError("quantize_values supports 2- or 4-bit payloads")
    b, h, t, d = value.shape
    if d % channel_group:
        raise ValueError(f"D={d} must be divisible by channel_group={channel_group}")
    if channel_group % values_per_byte(bits):
        raise ValueError("channel_group is incompatible with bit width")

    ng = d // channel_group
    x = value.reshape(b, h, t, ng, channel_group)
    mn = x.amin(dim=-1, keepdim=True)
    mx = x.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / float((1 << bits) - 1)).clamp_min(1e-8)
    q = torch.round((x - mn) / scale).clamp_(0, (1 << bits) - 1).to(torch.uint8)
    payload = pack_last_dim(q, bits).contiguous()
    return PackedValues(
        payload=payload,
        scale=scale.squeeze(-1).to(param_dtype).contiguous(),
        zero=mn.squeeze(-1).to(param_dtype).contiguous(),
        bits=bits,
        channel_group=channel_group,
        tokens=t,
    )


def dequantize_values(
    packed: PackedValues,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    q = unpack_last_dim(packed.payload, packed.bits, packed.channel_group)
    x = q.to(torch.float32) * packed.scale.float().unsqueeze(-1)
    x = x + packed.zero.float().unsqueeze(-1)
    b, h, t, ng, cg = x.shape
    return x.reshape(b, h, t, ng * cg).to(dtype)


def repeat_kv(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    if x.shape[1] == num_q_heads:
        return x
    if num_q_heads % x.shape[1]:
        raise ValueError("number of query heads must be divisible by KV heads")
    return x.repeat_interleave(num_q_heads // x.shape[1], dim=1)


def dense_attention_reference(
    query: torch.Tensor,
    old_key: torch.Tensor,
    old_value: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    key = torch.cat([old_key, current_key], dim=2)
    value = torch.cat([old_value, current_value], dim=2)
    key = repeat_kv(key, query.shape[1])
    value = repeat_kv(value, query.shape[1])
    if scale is None:
        scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    prob = torch.softmax(scores, dim=-1)
    return torch.matmul(prob, value.float()).to(query.dtype)


def compact_attention_reference(
    query: torch.Tensor,
    selected_key: torch.Tensor,
    selected_value: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    return dense_attention_reference(
        query,
        selected_key,
        selected_value,
        current_key,
        current_value,
        scale=scale,
    )


def selector_importance_reference(
    query: torch.Tensor,
    old_key: torch.Tensor,
    *,
    query_indices: list[int],
    current_key: torch.Tensor | None = None,
    domain: str = "prefix",
    score_kind: str = "softmax",
    scale: float | None = None,
) -> torch.Tensor:
    if not query_indices:
        raise ValueError("query_indices cannot be empty")
    b, hq, _, d = query.shape
    hkv = old_key.shape[1]
    if hq % hkv:
        raise ValueError("Hq must be divisible by Hkv")
    g = hq // hkv
    q = query[:, :, query_indices, :].reshape(b, hkv, g, len(query_indices), d)
    logits_old = torch.einsum("bhgmd,bhnd->bhgmn", q.float(), old_key.float())
    logits_old = logits_old * (scale if scale is not None else d**-0.5)

    if score_kind == "raw":
        return logits_old.mean(dim=(2, 3))
    if score_kind != "softmax":
        raise ValueError(f"unknown score_kind={score_kind}")

    if domain == "prefix":
        prob_old = torch.softmax(logits_old, dim=-1)
    elif domain == "full":
        if current_key is None:
            raise ValueError("current_key is required for domain='full'")
        logits_cur = torch.einsum("bhgmd,bhnd->bhgmn", q.float(), current_key.float())
        logits_cur = logits_cur * (scale if scale is not None else d**-0.5)
        denom = torch.logsumexp(torch.cat([logits_old, logits_cur], dim=-1), dim=-1)
        prob_old = torch.exp(logits_old - denom.unsqueeze(-1))
    else:
        raise ValueError(f"unknown selector domain={domain}")
    return prob_old.mean(dim=(2, 3))


def select_topk_reference(
    query: torch.Tensor,
    old_key: torch.Tensor,
    *,
    query_indices: list[int],
    topk: int,
    current_key: torch.Tensor | None = None,
    domain: str = "prefix",
    score_kind: str = "softmax",
    sort_indices: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    importance = selector_importance_reference(
        query,
        old_key,
        query_indices=query_indices,
        current_key=current_key,
        domain=domain,
        score_kind=score_kind,
    )
    k = min(topk, old_key.shape[2])
    values, indices = torch.topk(importance, k=k, dim=-1, largest=True, sorted=False)
    if sort_indices:
        indices, order = torch.sort(indices, dim=-1)
        values = torch.gather(values, -1, order)
    return indices, values


def gather_per_kv_head(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if x.shape[:2] != indices.shape[:2]:
        raise ValueError("batch/head dimensions do not match")
    gather_idx = indices.unsqueeze(-1).expand(*indices.shape, x.shape[-1])
    return torch.gather(x, 2, gather_idx)


def sdpa_compact(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    try:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=query.shape[1] != key.shape[1],
        )
    except (RuntimeError, TypeError):
        return F.scaled_dot_product_attention(
            query,
            repeat_kv(key, query.shape[1]),
            repeat_kv(value, query.shape[1]),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

"""KIVI-like K/V/Q quantization (quantize → dequantize, no custom kernels).

Geometry (KIVI):
  K — per-channel: independent min/max scale per head_dim channel (group_size=1)
                  or per group of channels when group_size > 1.
  V — per-token:   one min/max scale per (batch, head, token) over full head_dim.
  Q — per-token:   same as V (used only in attention hook for old-cache QK).

Quantized payloads store scale (+ zero-point for asymmetric) separately from
dequantized tensors used at runtime.  Two cache copies can coexist at different
precisions via ``DualPrecisionCache`` in ``kv_store.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

QuantKind = Literal["k_per_channel", "v_per_token", "q_per_token"]


def _max_int(bits: int) -> int:
    if bits == 16:
        return 0  # no quant
    if bits not in (2, 4, 8):
        raise ValueError(f"unsupported bits={bits}")
    return 2**bits - 1


@dataclass
class QuantParams:
    bits: int
    kind: QuantKind
    scale: torch.Tensor
    zero_point: torch.Tensor

    def nbytes(self) -> int:
        if self.bits >= 16:
            return 0
        elem = self.scale.numel() + self.zero_point.numel()
        q_bits = self.bits  # stored separately in caller
        return elem * 4 + q_bits  # scale/zp as fp32; q payload counted elsewhere


@dataclass
class QuantizedTensor:
    """Round-trip container; ``dequantized`` is what attention uses."""

    dequantized: torch.Tensor
    params: QuantParams | None = None
    q_int: torch.Tensor | None = None

    @property
    def bits(self) -> int:
        return self.params.bits if self.params else 16


def quantize_dequantize(
    tensor: torch.Tensor,
    bits: int | str,
    kind: QuantKind,
    *,
    group_size: int = 1,
) -> QuantizedTensor:
    """Symmetric/asymmetric round-trip quant → dequant."""
    b = _parse_bits(bits)
    if b >= 16:
        return QuantizedTensor(dequantized=tensor.clone(), params=None, q_int=None)

    max_int = _max_int(b)
    x = tensor

    if kind == "k_per_channel":
        # [B, H, T, D] — per-channel along D (optionally grouped)
        bsz, n_heads, seq, dim = x.shape
        if dim % group_size != 0:
            raise ValueError(f"head_dim {dim} not divisible by group_size {group_size}")
        ng = dim // group_size
        xv = x.view(bsz, n_heads, seq, ng, group_size)
        mn = xv.amin(dim=-1, keepdim=True)
        mx = xv.amax(dim=-1, keepdim=True)
        scale = ((mx - mn) / float(max_int)).clamp(min=1e-8)
        q = ((xv - mn) / scale).round().clamp(0, max_int)
        deq = (q * scale + mn).view(bsz, n_heads, seq, dim)
        params = QuantParams(bits=b, kind=kind, scale=scale, zero_point=mn)
        q_int = q.to(torch.uint8)

    elif kind in ("v_per_token", "q_per_token"):
        # [B, H, T, D] — one scale per token over D
        mn = x.amin(dim=-1, keepdim=True)
        mx = x.amax(dim=-1, keepdim=True)
        scale = ((mx - mn) / float(max_int)).clamp(min=1e-8)
        q = ((x - mn) / scale).round().clamp(0, max_int)
        deq = q * scale + mn
        params = QuantParams(bits=b, kind=kind, scale=scale, zero_point=mn)
        q_int = q.to(torch.uint8)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    return QuantizedTensor(dequantized=deq, params=params, q_int=q_int)


def _parse_bits(bits: int | str) -> int:
    s = str(bits).lower()
    if s in ("fp16", "bf16", "16"):
        return 16
    return int(s)


def storage_bytes_kv(
    *,
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    k_bits: int,
    v_bits: int,
    batch_size: int = 1,
) -> dict[str, int]:
    """Analytic KV cache footprint (stored, not read)."""
    kv_elems = batch_size * num_kv_heads * num_tokens * head_dim

    def _payload(bits: int) -> int:
        if bits >= 16:
            return kv_elems * 2  # fp16 bytes
        return (kv_elems * bits) // 8

    def _meta_per_token(bits: int, kind: QuantKind) -> int:
        if bits >= 16:
            return 0
        if kind == "k_per_channel":
            # scale + zp per channel group
            return batch_size * num_kv_heads * num_tokens * head_dim * 4 * 2
        # per-token: 2 fp32 per (b,h,t)
        return batch_size * num_kv_heads * num_tokens * 4 * 2

    k_store = _payload(k_bits) + _meta_per_token(k_bits, "k_per_channel")
    v_store = _payload(v_bits) + _meta_per_token(v_bits, "v_per_token")
    return {
        "k_bytes": k_store,
        "v_bytes": v_store,
        "total_bytes": k_store + v_store,
    }


def apply_precision(
    tensor: torch.Tensor,
    bits: int | str,
    kind: QuantKind,
    *,
    group_size: int = 1,
) -> torch.Tensor:
    """Return dequantized tensor at requested precision."""
    return quantize_dequantize(tensor, bits, kind, group_size=group_size).dequantized

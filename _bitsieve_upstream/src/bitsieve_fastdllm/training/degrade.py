"""How the old KV cache is degraded inside a training forward.

At inference a block's queries read the prefix out of :class:`PackedKVCache`:
keys quantized per channel inside a 32-*token* group, values quantized per token
inside a 32-*channel* group, scale and zero stored in ``param_dtype`` (float16),
and ``residual_tokens: 0`` so *nothing* is held back in bf16. Training has to
hand the student that same grid, or it learns to tolerate a corruption the cache
never produces.

Two modes:

``quant``
    The repository's own ``simulate_*_quantization`` — bit-exact with the packed
    path (``tests/test_simulate_value_quantization.py``) — wrapped in a
    straight-through estimator so gradients still flow. This is the faithful one.

``noise``
    Additive Gaussian with the variance the real rounding error has. Cheaper,
    smooth, and it does not commit the student to one specific rounding grid, so
    it generalizes across bit widths. Its std is derived from the *same* grouping
    the quantizer uses, then corrected by the ratio ``calibrate.py`` measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..reference import simulate_key_quantization, simulate_value_quantization

SQRT12 = math.sqrt(12.0)


def _levels(bits: int) -> float:
    return float((1 << bits) - 1)


def key_quant_sigma(
    key: torch.Tensor,
    bits: int,
    *,
    token_group: int = 32,
) -> torch.Tensor:
    """Per-element std of key rounding error: groups of ``token_group`` tokens,
    min/max per channel inside the group, error uniform on ``[-d/2, d/2]``."""
    b, h, t, d = key.shape
    sigma = torch.zeros_like(key, dtype=torch.float32)
    ng = t // token_group
    if ng == 0:
        return sigma
    head = ng * token_group
    x = key[:, :, :head, :].float().reshape(b, h, ng, token_group, d)
    delta = (x.amax(dim=-2, keepdim=True) - x.amin(dim=-2, keepdim=True)) / _levels(bits)
    sigma[:, :, :head, :] = (
        delta.clamp_min(1e-8).expand_as(x) / SQRT12
    ).reshape(b, h, head, d)
    if head < t:  # ragged tail is its own short group
        tail = key[:, :, head:, :].float()
        delta = (tail.amax(dim=2, keepdim=True) - tail.amin(dim=2, keepdim=True)) / _levels(bits)
        sigma[:, :, head:, :] = delta.clamp_min(1e-8).expand_as(tail) / SQRT12
    return sigma


def value_quant_sigma(
    value: torch.Tensor,
    bits: int,
    *,
    channel_group: int = 32,
) -> torch.Tensor:
    """Per-element std of value rounding error: per token, min/max inside each
    group of ``channel_group`` channels."""
    b, h, t, d = value.shape
    if d % channel_group:
        raise ValueError(
            f"head_dim={d} must be divisible by channel_group={channel_group}"
        )
    ng = d // channel_group
    x = value.float().reshape(b, h, t, ng, channel_group)
    delta = (x.amax(dim=-1, keepdim=True) - x.amin(dim=-1, keepdim=True)) / _levels(bits)
    sigma = (delta.clamp_min(1e-8).expand_as(x) / SQRT12).reshape(b, h, t, d)
    return sigma


@dataclass(slots=True)
class CacheDegradation:
    """What the student's loss forward reads instead of the exact prefix."""

    mode: str = "quant"                 # "quant" | "noise" | "exact"
    k_bits: int = 4
    v_bits: int = 4
    key_token_group: int = 32
    value_channel_group: int = 32
    param_dtype: torch.dtype = torch.float16
    # Correction from calibrate.py (measured std / analytic std); noise mode only.
    k_noise_scale: float = 1.0
    v_noise_scale: float = 1.0

    @property
    def degrades_keys(self) -> bool:
        return self.mode != "exact" and self.k_bits < 16

    @property
    def degrades_values(self) -> bool:
        return self.mode != "exact" and self.v_bits < 16

    @property
    def active(self) -> bool:
        return self.degrades_keys or self.degrades_values


def degrade_keys(
    key: torch.Tensor,
    cfg: CacheDegradation,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not cfg.degrades_keys:
        return key
    if cfg.mode == "quant":
        q = simulate_key_quantization(
            key,
            bits=cfg.k_bits,
            token_group=cfg.key_token_group,
            param_dtype=cfg.param_dtype,
            allow_ragged=True,
        )
        # Straight-through. `key - key.detach()` is exactly zero, so the forward
        # value is bit-exactly `q` -- writing it as `key + (q - key).detach()`
        # would round through bf16 and land off the cache's grid. `q` must be
        # detached too, or the gradient also flows through the quantizer's own
        # min/max and stops being the identity the estimator promises.
        return q.detach() + (key - key.detach())
    sigma = key_quant_sigma(key, cfg.k_bits, token_group=cfg.key_token_group)
    noise = torch.randn(key.shape, device=key.device, dtype=torch.float32, generator=generator)
    return key + (noise * sigma * cfg.k_noise_scale).to(key.dtype)


def degrade_values(
    value: torch.Tensor,
    cfg: CacheDegradation,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not cfg.degrades_values:
        return value
    if cfg.mode == "quant":
        q = simulate_value_quantization(
            value,
            bits=cfg.v_bits,
            channel_group=cfg.value_channel_group,
            param_dtype=cfg.param_dtype,
        )
        return q.detach() + (value - value.detach())   # bit-exact forward, gradient 1
    sigma = value_quant_sigma(value, cfg.v_bits, channel_group=cfg.value_channel_group)
    noise = torch.randn(value.shape, device=value.device, dtype=torch.float32, generator=generator)
    return value + (noise * sigma * cfg.v_noise_scale).to(value.dtype)

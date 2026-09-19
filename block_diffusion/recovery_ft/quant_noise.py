"""Gaussian surrogate for KIVI KV-quantization error.

The real inference path (``sparse_kv_exp``) stores the old cache with

  K — asymmetric per-channel min/max inside a group of ``group_size`` cache
      tokens (``quantize_kivi_key_roundtrip``), last ``residual_length``
      tokens kept in bf16;
  V — asymmetric per-token min/max over ``head_dim`` (``apply_precision``
      with ``v_per_token``), *every* cached token quantized.

Rounding error of an asymmetric b-bit quantizer with step
``delta = (max - min) / (2**b - 1)`` is (to a very good approximation)
uniform on ``[-delta/2, delta/2]``, i.e. zero mean and std ``delta / sqrt(12)``.
``kivi_key_sigma`` / ``kivi_value_sigma`` return exactly that std, computed
from the *same* grouping the real quantizer uses, so the injected noise has
the same per-channel / per-token heteroscedastic structure as the real error.

``calibrate_noise.py`` checks this analytic sigma against the error measured
on real caches; see ``NOISE_CALIBRATION.md`` for the numbers.
"""

from __future__ import annotations

import math

import torch

SQRT12 = math.sqrt(12.0)


def _max_int(bits: int) -> float:
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"bits must be 2, 3, 4 or 8, got {bits}")
    return float(2**bits - 1)


def kivi_key_sigma(
    key: torch.Tensor,
    bits: int,
    *,
    group_size: int = 32,
    residual_length: int = 0,
) -> torch.Tensor:
    """Per-element std of KIVI key-quantization error. ``key``: [B, H, T, D].

    Mirrors ``quantize_kivi_key_roundtrip``: min/max are taken over the token
    axis inside each group of ``group_size`` tokens, independently per channel.
    Tokens in the trailing residual window (and the ragged tail that KIVI leaves
    un-grouped) get sigma 0 because they are stored in bf16.
    """
    b, h, t, d = key.shape
    max_int = _max_int(bits)
    sigma = torch.zeros_like(key, dtype=torch.float32)

    r = min(max(residual_length, 0), t)
    grouped_len = max(0, ((t - r) // group_size) * group_size)
    if grouped_len == 0:
        return sigma

    x = key[..., :grouped_len, :].float().view(b, h, grouped_len // group_size, group_size, d)
    delta = (x.amax(dim=-2, keepdim=True) - x.amin(dim=-2, keepdim=True)) / max_int
    delta = delta.clamp(min=1e-8).expand_as(x)
    sigma[..., :grouped_len, :] = (delta / SQRT12).reshape(b, h, grouped_len, d)
    return sigma


def kivi_value_sigma(
    value: torch.Tensor,
    bits: int,
    *,
    residual_length: int = 0,
) -> torch.Tensor:
    """Per-element std of per-token value-quantization error. ``value``: [B, H, T, D].

    Mirrors ``quantization.apply_precision(..., "v_per_token")``: one min/max
    per (batch, head, token) over ``head_dim``.  The sparse_kv_exp value path
    uses ``residual_length=0`` — every cached token is quantized.
    """
    b, h, t, d = value.shape
    max_int = _max_int(bits)
    sigma = torch.zeros_like(value, dtype=torch.float32)

    r = min(max(residual_length, 0), t)
    prefix = t - r
    if prefix <= 0:
        return sigma

    x = value[..., :prefix, :].float()
    delta = ((x.amax(dim=-1, keepdim=True) - x.amin(dim=-1, keepdim=True)) / max_int).clamp(min=1e-8)
    sigma[..., :prefix, :] = delta.expand_as(x) / SQRT12
    return sigma


def add_quant_noise(
    x: torch.Tensor,
    sigma: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """x + sigma * N(0, 1), kept in x's dtype. Gradients flow through x only."""
    noise = torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=generator)
    return x + (noise * sigma).to(x.dtype)

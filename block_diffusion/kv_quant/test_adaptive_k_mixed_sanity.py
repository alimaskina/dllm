#!/usr/bin/env python3
"""Sanity: all-K2 gather path must match uniform KIVI2 key roundtrip."""

from __future__ import annotations

import torch

from kv_cache_quant import (
    DEFAULT_KIVI_GROUP_SIZE,
    DEFAULT_KIVI_RESIDUAL_LENGTH,
    _kivi_grouped_token_len,
    quantize_kivi_key_mixed_tiers,
    quantize_kivi_key_roundtrip,
)


def main() -> None:
    torch.manual_seed(0)
    b, h, t, d = 1, 8, 160, 128
    key = torch.randn(b, h, t, d, dtype=torch.float32)

    grouped_len = _kivi_grouped_token_len(t, DEFAULT_KIVI_GROUP_SIZE, DEFAULT_KIVI_RESIDUAL_LENGTH)
    bits = [2] * grouped_len

    mixed = quantize_kivi_key_mixed_tiers(key[..., :grouped_len, :], bits)
    uniform = quantize_kivi_key_roundtrip(
        key,
        2,
        group_size=DEFAULT_KIVI_GROUP_SIZE,
        residual_length=DEFAULT_KIVI_RESIDUAL_LENGTH,
    )[..., :grouped_len, :]

    max_diff = (mixed - uniform).abs().max().item()
    ok = torch.allclose(mixed, uniform, atol=1e-5, rtol=1e-5)
    print(f"grouped_len={grouped_len} max_diff={max_diff:.2e} allclose={ok}")
    if not ok:
        raise SystemExit(1)

    # Interleaved K4/K2 should differ from uniform K2
    alt_bits = [4 if i % 2 == 0 else 2 for i in range(grouped_len)]
    alt = quantize_kivi_key_mixed_tiers(key[..., :grouped_len, :], alt_bits)
    alt_diff = (alt - uniform).abs().mean().item()
    print(f"interleaved K4/K2 mean diff from uniform K2: {alt_diff:.4f}")
    print("OK")


if __name__ == "__main__":
    main()

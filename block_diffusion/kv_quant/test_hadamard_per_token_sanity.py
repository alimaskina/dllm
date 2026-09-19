#!/usr/bin/env python3
"""Sanity: FWHT involution, score invariance, per-token vs Hadamard vs KIVI."""

from __future__ import annotations

import math

import torch

from kv_cache_quant import (
    fast_walsh_hadamard,
    quantize_hadamard_per_token_roundtrip,
    quantize_kivi_key_roundtrip,
    quantize_per_token_key_roundtrip,
    quantize_quarot_per_token_roundtrip,
    quantize_qjl_key_roundtrip,
)


def _sylvester_hadamard(n: int) -> torch.Tensor:
    h = torch.tensor([[1.0]])
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)],
            dim=0,
        )
    return h / math.sqrt(n)


def main() -> None:
    torch.manual_seed(0)

    # Explicit H_8 matches FWHT.
    n = 8
    h = _sylvester_hadamard(n)
    x = torch.randn(3, n)
    fw = fast_walsh_hadamard(x)
    mat = x @ h.T
    if not torch.allclose(fw.float(), mat, atol=1e-5, rtol=1e-5):
        raise SystemExit(f"FWHT != H@x: maxΔ={(fw.float() - mat).abs().max().item():.2e}")

    # Involution + norm preservation on head_dim=128.
    key = torch.randn(1, 4, 64, 128)
    twice = fast_walsh_hadamard(fast_walsh_hadamard(key))
    if not torch.allclose(twice, key, atol=1e-4, rtol=1e-4):
        raise SystemExit(
            f"FWHT not involutory: maxΔ={(twice - key).abs().max().item():.2e}"
        )
    norms = key.float().norm(dim=-1)
    rot_norms = fast_walsh_hadamard(key).float().norm(dim=-1)
    if not torch.allclose(norms, rot_norms, atol=1e-4, rtol=1e-4):
        raise SystemExit("FWHT does not preserve L2")

    # qᵀk == (Hq)ᵀ(Hk)
    q = torch.randn(1, 4, 8, 128)
    k = torch.randn(1, 4, 64, 128)
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores_h = torch.matmul(
        fast_walsh_hadamard(q), fast_walsh_hadamard(k).transpose(-2, -1)
    )
    if not torch.allclose(scores, scores_h, atol=1e-4, rtol=1e-4):
        raise SystemExit(
            f"score not invariant: maxΔ={(scores - scores_h).abs().max().item():.2e}"
        )

    residual = 32
    tail = quantize_hadamard_per_token_roundtrip(k, 4, residual_length=residual)
    if not torch.equal(tail[..., -residual:, :], k[..., -residual:, :]):
        raise SystemExit("Hadamard residual tail was quantized")

    # Channel-outlier keys: Hadamard should beat naive per-token MSE.
    outliers = torch.randn(1, 2, 96, 128)
    outliers[..., 0] = outliers[..., 0] * 20.0
    naive = quantize_per_token_key_roundtrip(outliers, 4, residual_length=residual)
    had = quantize_hadamard_per_token_roundtrip(outliers, 4, residual_length=residual)
    kivi = quantize_kivi_key_roundtrip(outliers, 4, residual_length=residual)
    prefix = outliers.shape[-2] - residual
    mse = lambda a, b: (a[..., :prefix, :] - b[..., :prefix, :]).float().pow(2).mean().item()
    mse_naive = mse(outliers, naive)
    mse_had = mse(outliers, had)
    mse_kivi = mse(outliers, kivi)
    print(
        f"outlier MSE  naive={mse_naive:.5f}  hadamard={mse_had:.5f}  kivi={mse_kivi:.5f}"
    )
    if mse_had >= mse_naive:
        raise SystemExit("Hadamard per-token MSE should beat naive per-token on outliers")
    if math.isnan(mse_had) or math.isnan(mse_kivi):
        raise SystemExit("NaN in roundtrip")

    # QuaRot two-stage RHT: involution without quant; 4-head keys (H*D=512).
    k4h = torch.randn(1, 4, 96, 128)
    qrt = quantize_quarot_per_token_roundtrip(k4h, 8, residual_length=residual)
    if not torch.equal(qrt[..., -residual:, :], k4h[..., -residual:, :]):
        raise SystemExit("QuaRot residual tail was quantized")
    mse_qrt = mse(k4h, quantize_quarot_per_token_roundtrip(k4h, 4, residual_length=residual))
    print(f"quarot-4h outlier-free MSE 4bit={mse_qrt:.5f}")
    if math.isnan(mse_qrt):
        raise SystemExit("NaN in QuaRot roundtrip")

    # QJL: qᵀk̂ ≈ qᵀk (asymmetric 1-bit JL estimator).
    torch.manual_seed(1)
    qv = torch.randn(1, 4, 8, 128)
    kv = torch.randn(1, 4, 64, 128)
    k_qjl = quantize_qjl_key_roundtrip(kv, 4, residual_length=0)
    true_scores = torch.matmul(qv, kv.transpose(-2, -1))
    qjl_scores = torch.matmul(qv, k_qjl.transpose(-2, -1))
    corr = torch.corrcoef(torch.stack([true_scores.flatten(), qjl_scores.flatten()]))[0, 1]
    print(f"QJL m=512 score corr={corr.item():.3f}")
    if corr.item() < 0.7:
        raise SystemExit(f"QJL score correlation too low: {corr.item():.3f}")
    if not torch.equal(
        quantize_qjl_key_roundtrip(kv, 4, residual_length=32)[..., -32:, :],
        kv[..., -32:, :],
    ):
        raise SystemExit("QJL residual tail was quantized")

    print("OK")


if __name__ == "__main__":
    main()

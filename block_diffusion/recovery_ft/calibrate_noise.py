#!/usr/bin/env python3
"""Measure the real KIVI quantization error on real caches and check the sigma model.

Branches D and E replace the quantizer with additive Gaussian noise whose std is
``quant_noise.kivi_*_sigma`` (the uniform-rounding prediction ``delta/sqrt(12)``).
This script builds genuine prompt caches with the model, quantizes them with the
*same* functions the evaluation harness uses, and reports

  * measured std of ``dequant(x) - x`` per layer, for K and V;
  * the analytic sigma over the same elements;
  * their ratio (1.0 == the surrogate is exactly calibrated);
  * a normality check (kurtosis) on the error, since we replace it with a Gaussian.

The per-layer measured stds are written to JSON so training can optionally use
the measured values instead of the analytic ones (``--sigma-source calibrated``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent / "sparse_kv_exp"), str(_HERE.parent / "kv_quant")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data import load_math_train  # noqa: E402
from benchmark_utils import build_chat_prompt  # noqa: E402
from kv_cache_quant import quantize_key_roundtrip  # noqa: E402
from model_registry import load_model_and_tokenizer  # noqa: E402
from quant_noise import kivi_key_sigma, kivi_value_sigma  # noqa: E402
from quantization import apply_precision  # noqa: E402


def _stats(err: torch.Tensor, sigma: torch.Tensor, active: torch.Tensor) -> dict:
    e = err[active].float()
    s = sigma[active].float()
    if e.numel() == 0:
        return {}
    centered = e - e.mean()
    var = centered.pow(2).mean()
    kurt = (centered.pow(4).mean() / var.pow(2)).item() if var > 0 else float("nan")
    return {
        "measured_std": e.std().item(),
        "analytic_sigma": s.mean().item(),
        "ratio_measured_over_analytic": (e.std() / s.mean()).item(),
        "measured_mean": e.mean().item(),
        # Uniform rounding error has kurtosis 1.8; a Gaussian has 3.0.
        "kurtosis": kurt,
        "n_elements": int(e.numel()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--bits", type=int, nargs="+", default=[4])
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--kivi-group-size", type=int, default=32)
    ap.add_argument("--kivi-residual-length", type=int, default=32)
    ap.add_argument("--out", default="noise_calibration.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tokenizer, _ = load_model_and_tokenizer("fast_dllm_v2_7b", device)
    model.eval()
    model.model.bd_size = args.block_size
    model.config.bd_size = args.block_size

    rows = load_math_train(num_examples=args.num_prompts, seed=7, exclude_math500=False)

    report: dict = {
        "model": "fast_dllm_v2_7b",
        "k_quant_scheme": "kivi",
        "v_quant_scheme": "kivi_per_token",
        "kivi_group_size": args.kivi_group_size,
        "kivi_residual_length": args.kivi_residual_length,
        "num_prompts": len(rows),
        "bits": {},
    }

    caches = []
    with torch.no_grad():
        for r in rows:
            # Calibrate on prompt + reference solution: at inference the cache
            # holds the prompt *and* the blocks decoded so far, so the statistics
            # must cover generated-text KV as well.
            prompt = build_chat_prompt(tokenizer, r["problem"])
            text = prompt + r["solution"]
            ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            n = (ids.shape[1] // args.block_size) * args.block_size
            if n < 2 * args.block_size:
                continue
            out = model.forward(
                input_ids=ids[:, :n],
                use_cache=True,
                update_past_key_values=True,
                block_size=args.block_size,
            )
            pkv = out.past_key_values
            caches.append(
                [
                    (pkv.key_cache[i].detach().clone(), pkv.value_cache[i].detach().clone())
                    for i in range(len(pkv.key_cache))
                ]
            )
    print(f"[calib] collected {len(caches)} caches, "
          f"lens={[c[0][0].shape[2] for c in caches]}")

    for bits in args.bits:
        per_layer: dict[str, dict] = {}
        agg_k: list[dict] = []
        agg_v: list[dict] = []
        n_layers = len(caches[0])
        for layer in range(n_layers):
            k_acc, v_acc = [], []
            for cache in caches:
                k, v = cache[layer]
                k = k.float()
                v = v.float()

                k_q = quantize_key_roundtrip(
                    k, bits, scheme="kivi",
                    group_size=args.kivi_group_size,
                    residual_length=args.kivi_residual_length,
                )
                sig_k = kivi_key_sigma(
                    k, bits,
                    group_size=args.kivi_group_size,
                    residual_length=args.kivi_residual_length,
                )
                k_acc.append(_stats(k_q - k, sig_k, sig_k > 0))

                v_q = apply_precision(v, bits, "v_per_token")
                sig_v = kivi_value_sigma(v, bits, residual_length=0)
                v_acc.append(_stats(v_q - v, sig_v, sig_v > 0))

            def _mean(rows_: list[dict], key: str) -> float:
                vals = [r[key] for r in rows_ if r]
                return float(sum(vals) / len(vals)) if vals else float("nan")

            lk = {key: _mean(k_acc, key) for key in k_acc[0]} if k_acc and k_acc[0] else {}
            lv = {key: _mean(v_acc, key) for key in v_acc[0]} if v_acc and v_acc[0] else {}
            per_layer[str(layer)] = {"k": lk, "v": lv}
            agg_k.append(lk)
            agg_v.append(lv)

        def _overall(rows_: list[dict], key: str) -> float:
            vals = [r[key] for r in rows_ if r and key in r]
            return float(sum(vals) / len(vals)) if vals else float("nan")

        report["bits"][str(bits)] = {
            "per_layer": per_layer,
            "summary": {
                "k_ratio_measured_over_analytic": _overall(agg_k, "ratio_measured_over_analytic"),
                "v_ratio_measured_over_analytic": _overall(agg_v, "ratio_measured_over_analytic"),
                "k_kurtosis": _overall(agg_k, "kurtosis"),
                "v_kurtosis": _overall(agg_v, "kurtosis"),
                "k_measured_std": _overall(agg_k, "measured_std"),
                "v_measured_std": _overall(agg_v, "measured_std"),
            },
        }
        s = report["bits"][str(bits)]["summary"]
        print(f"\n=== {bits}-bit ===")
        print(f"  K: measured std {s['k_measured_std']:.5f}  "
              f"measured/analytic = {s['k_ratio_measured_over_analytic']:.4f}  "
              f"kurtosis {s['k_kurtosis']:.2f}")
        print(f"  V: measured std {s['v_measured_std']:.5f}  "
              f"measured/analytic = {s['v_ratio_measured_over_analytic']:.4f}  "
              f"kurtosis {s['v_kurtosis']:.2f}")
        print("  (uniform rounding error -> kurtosis 1.8; Gaussian -> 3.0;"
              " ratio 1.0 means the surrogate std is exact)")

    report["recommended_noise_scale"] = {
        str(b): {
            "k": report["bits"][str(b)]["summary"]["k_ratio_measured_over_analytic"],
            "v": report["bits"][str(b)]["summary"]["v_ratio_measured_over_analytic"],
        }
        for b in args.bits
    }

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

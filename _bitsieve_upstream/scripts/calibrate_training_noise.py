#!/usr/bin/env python3
"""Measure the packed cache's real rounding error and check the noise surrogate.

Branch D and E can replace the quantizer with additive Gaussian noise. That is
only honest if the noise has the variance the quantizer's error actually has,
so this builds genuine prefixes with the model, runs the *same*
``simulate_*_quantization`` functions the cache is bit-exact with, and reports

  * measured std of ``dequant(x) - x`` per layer, for K and V;
  * the analytic std ``delta/sqrt(12)`` over the same grouping;
  * their ratio (1.0 == the surrogate is exactly calibrated);
  * kurtosis, since a Gaussian is being substituted for the real error.

The per-bit ratios are written to JSON and consumed by --noise-calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.eval.benchmarks import competition_math_prompt  # noqa: E402
from bitsieve_fastdllm.eval.common import encode_prompt, load_fast_dllm  # noqa: E402
from bitsieve_fastdllm.reference import (  # noqa: E402
    simulate_key_quantization,
    simulate_value_quantization,
)
from bitsieve_fastdllm.training.data import load_math_train  # noqa: E402
from bitsieve_fastdllm.training.degrade import key_quant_sigma, value_quant_sigma  # noqa: E402


def _stats(err: torch.Tensor, sigma: torch.Tensor) -> dict:
    e = err.flatten().float()
    s = sigma.flatten().float()
    centered = e - e.mean()
    var = centered.pow(2).mean()
    return {
        "measured_std": e.std().item(),
        "analytic_sigma": s.mean().item(),
        "ratio": (e.std() / s.mean()).item(),
        "measured_mean": e.mean().item(),
        "kurtosis": (centered.pow(4).mean() / var.pow(2)).item() if var > 0 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 2])
    ap.add_argument("--num-prompts", type=int, default=40)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--key-token-group", type=int, default=32)
    ap.add_argument("--value-channel-group", type=int, default=32)
    ap.add_argument("--out", default="results/training_noise_calibration.json")
    args = ap.parse_args()

    model, tokenizer = load_fast_dllm(args.model, dtype=torch.bfloat16, device=args.device)
    model.eval()
    model.model.bd_size = args.block_size
    model.config.bd_size = args.block_size
    device = next(model.parameters()).device

    caches = []
    with torch.no_grad():
        for row in load_math_train(limit=args.num_prompts, seed=7, exclude_math500=False):
            # Prefix = prompt + reference solution: at inference the cache holds
            # the prompt *and* every block decoded so far, so the statistics have
            # to cover generated-text KV too.
            ids = encode_prompt(
                tokenizer,
                competition_math_prompt(row["problem"]) + row["solution"],
                max_input_tokens=4096, use_chat_template=True, device=device,
            )
            n = (ids.shape[1] // args.block_size) * args.block_size
            if n < 2 * args.block_size:
                continue
            pkv = model.forward(
                input_ids=ids[:, :n], use_cache=True,
                update_past_key_values=True, block_size=args.block_size,
            ).past_key_values
            caches.append([
                (pkv.key_cache[i].detach().clone(), pkv.value_cache[i].detach().clone())
                for i in range(len(pkv.key_cache))
            ])
    print(f"[calib] {len(caches)} prefixes, lengths {[c[0][0].shape[2] for c in caches][:8]}...")
    if not caches:
        print("no usable prefixes")
        return 1

    report: dict = {
        "model": args.model,
        "key_token_group": args.key_token_group,
        "value_channel_group": args.value_channel_group,
        "num_prefixes": len(caches),
        "bits": {},
        "noise_scale": {},
    }

    n_layers = len(caches[0])
    for bits in args.bits:
        per_layer = {}
        k_all, v_all = [], []
        for layer in range(n_layers):
            ks, vs = [], []
            for cache in caches:
                k, v = (t.float() for t in cache[layer])
                kq = simulate_key_quantization(
                    k, bits=bits, token_group=args.key_token_group, allow_ragged=True
                )
                ks.append(_stats(kq - k, key_quant_sigma(k, bits, token_group=args.key_token_group)))
                vq = simulate_value_quantization(
                    v, bits=bits, channel_group=args.value_channel_group
                )
                vs.append(_stats(vq - v, value_quant_sigma(v, bits, channel_group=args.value_channel_group)))

            def mean(rows, key):
                return float(sum(r[key] for r in rows) / len(rows))

            lk = {k: mean(ks, k) for k in ks[0]}
            lv = {k: mean(vs, k) for k in vs[0]}
            per_layer[str(layer)] = {"k": lk, "v": lv}
            k_all.append(lk)
            v_all.append(lv)

        def overall(rows, key):
            return float(sum(r[key] for r in rows) / len(rows))

        summary = {
            "k_ratio": overall(k_all, "ratio"),
            "v_ratio": overall(v_all, "ratio"),
            "k_measured_std": overall(k_all, "measured_std"),
            "v_measured_std": overall(v_all, "measured_std"),
            "k_kurtosis": overall(k_all, "kurtosis"),
            "v_kurtosis": overall(v_all, "kurtosis"),
        }
        report["bits"][str(bits)] = {"per_layer": per_layer, "summary": summary}
        report["noise_scale"][str(bits)] = {"k": summary["k_ratio"], "v": summary["v_ratio"]}
        print(f"\n=== {bits}-bit ===")
        print(f"  K: std {summary['k_measured_std']:.5f}  measured/analytic "
              f"{summary['k_ratio']:.4f}  kurtosis {summary['k_kurtosis']:.2f}")
        print(f"  V: std {summary['v_measured_std']:.5f}  measured/analytic "
              f"{summary['v_ratio']:.4f}  kurtosis {summary['v_kurtosis']:.2f}")
        print("  (uniform rounding error -> kurtosis 1.8; Gaussian -> 3.0)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

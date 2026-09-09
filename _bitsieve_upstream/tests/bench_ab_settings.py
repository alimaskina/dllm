#!/usr/bin/env python3
"""A/B the two honesty fixes against upstream settings at an identical budget.

Same model, same prompt, same 5%-of-prefix selector; the only differences are
`residual_tokens` (32 -> 0) and `dense_prefix_layers` (2 -> 0). Both fixes
should be speed-neutral or better: dropping the residual moves tokens from a
torch side-pass into the packed kernel, and dropping the dense prefix removes
two full-prefix attention passes per denoising step.

Coverage diagnostics stay off - this measures the deployed path.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import ExperimentConfig  # noqa: E402
from bitsieve_fastdllm.eval.benchmarks import load_benchmark  # noqa: E402
from bitsieve_fastdllm.eval.common import encode_prompt, load_fast_dllm  # noqa: E402
from bitsieve_fastdllm.runtime.generator import BitSieveGenerator  # noqa: E402


def run(model, tokenizer, cfg, input_ids, repeats: int) -> dict:
    gen = BitSieveGenerator(model, tokenizer, cfg)
    tps, tpob, decode = [], [], []
    out = None
    for _ in range(repeats):
        res = gen.generate(input_ids)
        m = res.metrics
        tps.append(float(m["tokens_per_second"]))
        decode.append(float(m["decode_ms"]))
        if m.get("mean_tpob_ms"):
            tpob.append(float(m["mean_tpob_ms"]))
        out = m
    return {
        "tokens_per_second": statistics.median(tps),
        "decode_ms": statistics.median(decode),
        "mean_tpob_ms": statistics.median(tpob) if tpob else None,
        "gen_tokens": out["generated_tokens_per_request"],
        "nfe": out["nfe"],
        "sparse_block_fraction": out["sparse_block_fraction"],
        "key_residual": out["packed_cache"]["key_residual"],
        "resident_cache_bytes": out["resident_cache_bytes"],
        "cache_compression_ratio": out["cache_compression_ratio"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fixed", default="configs/proposed_a_k4v4_p5.yaml")
    p.add_argument("--upstream", default="/tmp/upstream_settings_k4v4_p5.yaml")
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    args = p.parse_args()

    model, tokenizer = load_fast_dllm(args.model, dtype=torch.bfloat16, device="cuda")
    examples = load_benchmark(args.benchmark, tokenizer=tokenizer, limit=1, split="test")

    results = {}
    for label, path in (("upstream (r32, dense2)", args.upstream), ("fixed (r0, dense0)", args.fixed)):
        cfg = ExperimentConfig.load(path)
        ids = encode_prompt(
            tokenizer, examples[0].prompt,
            max_input_tokens=cfg.max_cache_tokens - cfg.generation.max_new_tokens,
            use_chat_template=True, device=next(model.parameters()).device,
        )
        print(f"running {label} ...", flush=True)
        results[label] = run(model, tokenizer, cfg, ids, args.repeats)

    print(f"\n{'metric':<26} {'upstream':>14} {'fixed':>14} {'delta':>10}")
    up = results["upstream (r32, dense2)"]
    fx = results["fixed (r0, dense0)"]
    for key in ("tokens_per_second", "decode_ms", "mean_tpob_ms", "gen_tokens", "nfe",
                "sparse_block_fraction", "key_residual", "resident_cache_bytes",
                "cache_compression_ratio"):
        a, b = up.get(key), fx.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
            print(f"{key:<26} {a:>14.4g} {b:>14.4g} {(b - a) / a * 100:>+9.1f}%")
        else:
            print(f"{key:<26} {str(a):>14} {str(b):>14} {'':>10}")


if __name__ == "__main__":
    main()

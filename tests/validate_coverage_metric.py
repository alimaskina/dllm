#!/usr/bin/env python3
"""Does the coverage metric actually discriminate, or does it self-score?

The reference top-k is built from exact keys over every masked query, so it is
independent of the candidate. If that independence holds, a selector starved to
a single query must score clearly worse than one using all masked queries on the
same input. If the reference had instead been derived from the candidate's own
scores, both would report near-perfect coverage - which is the failure mode this
check exists to rule out.

Three configs, one prompt, identical 5%-of-prefix budget:
  all    - every masked query, bf16 keys   (expected ceiling)
  middle - one central query, bf16 keys    (expected clearly worse)
  quant  - uniform-5 queries, K4/V4 keys   (expected close to the ceiling)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import ExperimentConfig  # noqa: E402
from bitsieve_fastdllm.eval.benchmarks import load_benchmark  # noqa: E402
from bitsieve_fastdllm.eval.common import encode_prompt, load_fast_dllm  # noqa: E402
from bitsieve_fastdllm.runtime.generator import BitSieveGenerator  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="qmsum")
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    args = p.parse_args()

    model, tokenizer = load_fast_dllm(args.model, dtype=torch.bfloat16, device="cuda")
    examples = load_benchmark(args.benchmark, tokenizer=tokenizer, limit=1, split="test")

    variants = [
        ("all (32 queries, bf16 K)", "/tmp/cov_all_p5.yaml"),
        ("middle (1 query, bf16 K)", "/tmp/cov_middle_p5.yaml"),
        ("uniform5 (K4/V4)", "/tmp/cov_quant_p5.yaml"),
    ]
    out: dict[str, dict] = {}
    for label, path in variants:
        cfg = ExperimentConfig.load(path)
        ids = encode_prompt(
            tokenizer, examples[0].prompt,
            max_input_tokens=cfg.max_cache_tokens - cfg.generation.max_new_tokens,
            use_chat_template=True, device=next(model.parameters()).device,
        )
        print(f"running {label} (prompt {int(ids.shape[1])} tok) ...", flush=True)
        res = BitSieveGenerator(model, tokenizer, cfg).generate(ids)
        cov = res.metrics.get("coverage")
        out[label] = cov or {}
        print(f"   -> {cov}", flush=True)

    print(f"\n{'variant':<28} {'mass_mean':>10} {'mass_min':>10} {'ovl_mean':>10} {'cells':>7}")
    for label, _ in variants:
        c = out[label]
        if not c:
            print(f"{label:<28} {'(none)':>10}")
            continue
        print(
            f"{label:<28} {c['mass_mean']:>10.4f} {c['mass_min']:>10.4f} "
            f"{c['overlap_mean']:>10.4f} {c['cells']:>7d}"
        )

    fails = 0

    def chk(ok: bool, msg: str) -> None:
        nonlocal fails
        if not ok:
            fails += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")

    print("\n--- assertions ---")
    a = out["all (32 queries, bf16 K)"]
    m = out["middle (1 query, bf16 K)"]
    q = out["uniform5 (K4/V4)"]
    if a and m and q:
        chk(
            m["mass_mean"] < a["mass_mean"] - 0.01,
            f"single query scores worse than all queries "
            f"({m['mass_mean']:.4f} < {a['mass_mean']:.4f}) - the reference is "
            "not the candidate's own ranking",
        )
        chk(
            m["overlap_mean"] < a["overlap_mean"] - 0.01,
            f"same for index overlap ({m['overlap_mean']:.4f} < {a['overlap_mean']:.4f})",
        )
        chk(
            all(c["mass_mean"] <= 1.0 + 1e-6 for c in (a, m, q)),
            "no variant exceeds the reference ceiling",
        )
        chk(
            a["mass_mean"] - q["mass_mean"] < a["mass_mean"] - m["mass_mean"],
            f"4-bit keys cost less coverage than losing queries "
            f"(quant loses {a['mass_mean'] - q['mass_mean']:.4f}, "
            f"middle loses {a['mass_mean'] - m['mass_mean']:.4f})",
        )
    else:
        chk(False, "all three variants produced coverage")

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

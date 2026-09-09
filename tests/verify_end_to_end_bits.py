#!/usr/bin/env python3
"""On the real 7B model: does declared bit width actually reach selection?

Same prompt, same selector mode/queries, same budget, same seed - only
k_bits/v_bits differ (16 / 4 / 2). If selection secretly ranked on a
full-precision copy regardless of the declared bits, the trace's selected
indices and the coverage numbers would be identical across all three. They
should instead move monotonically with bit width, matching the synthetic
kernel-level test in test_quant_is_load_bearing.py.
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

    runs = {}
    for bits in (16, 4, 2):
        cfg = ExperimentConfig.load(f"/tmp/bitsonly_k{bits}.yaml")
        torch.manual_seed(cfg.seed)
        ids = encode_prompt(
            tokenizer, examples[0].prompt,
            max_input_tokens=cfg.max_cache_tokens - cfg.generation.max_new_tokens,
            use_chat_template=True, device=next(model.parameters()).device,
        )
        print(f"running k_bits={bits} (prompt {int(ids.shape[1])} tok) ...", flush=True)
        res = BitSieveGenerator(model, tokenizer, cfg).generate(ids)
        first_block_indices = {
            rec.layer: rec.index_checksum
            for rec in res.trace.selections
            if rec.block == res.trace.selections[0].block
        } if res.trace.selections else {}
        runs[bits] = {
            "coverage": res.metrics.get("coverage"),
            "checksums": first_block_indices,
            "text": res.texts[0],
        }
        print(f"   coverage={runs[bits]['coverage']}", flush=True)

    fails = 0

    def chk(ok: bool, msg: str) -> None:
        nonlocal fails
        if not ok:
            fails += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")

    print("\n--- assertions (same run, only k_bits/v_bits differ) ---")
    c16, c4, c2 = runs[16]["checksums"], runs[4]["checksums"], runs[2]["checksums"]
    chk(bool(c16) and bool(c4) and bool(c2), "all three runs produced a first sparse block")
    if c16 and c4 and c2:
        diff_16_4 = sum(1 for l in c16 if c16.get(l) != c4.get(l))
        diff_16_2 = sum(1 for l in c16 if c16.get(l) != c2.get(l))
        diff_4_2 = sum(1 for l in c16 if c4.get(l) != c2.get(l))
        chk(
            diff_16_4 > 0,
            f"K4 selects different indices than K16 in >=1 layer ({diff_16_4}/{len(c16)})",
        )
        chk(
            diff_16_2 > 0,
            f"K2 selects different indices than K16 in >=1 layer ({diff_16_2}/{len(c16)})",
        )
        chk(
            diff_4_2 > 0,
            f"K2 selects different indices than K4 in >=1 layer ({diff_4_2}/{len(c16)})",
        )

    cov16, cov4, cov2 = runs[16]["coverage"], runs[4]["coverage"], runs[2]["coverage"]
    if cov16 and cov4 and cov2:
        chk(
            cov16["mass_mean"] >= cov4["mass_mean"] >= cov2["mass_mean"] - 1e-9,
            f"coverage degrades monotonically with fewer bits "
            f"(K16={cov16['mass_mean']:.4f} >= K4={cov4['mass_mean']:.4f} "
            f">= K2={cov2['mass_mean']:.4f})",
        )
    else:
        chk(False, "coverage present for all three runs")

    chk(
        len({runs[b]["text"] for b in (16, 4, 2)}) > 1,
        "generated text is not identical across bit widths",
    )

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

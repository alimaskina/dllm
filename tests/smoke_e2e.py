#!/usr/bin/env python3
"""End-to-end smoke on the real checkpoint: did the fixes take effect?

Checks, on one short-prompt math example with the 2048-token budget:
  1. residual_tokens=0        -> nothing is held in fp16 in the packed cache
  2. dense_prefix_layers=0    -> layers 0/1 are no longer dense-bypassed
  3. sparse actually engages   -> blocks_sparse > 0, and the fraction is reported
  4. coverage is honest        -> scored against fp16 all-masked-query reference
  5. memory accounting         -> compact buffers are counted, not hidden
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import ExperimentConfig  # noqa: E402
from bitsieve_fastdllm.eval.benchmarks import load_benchmark, max_new_tokens_for  # noqa: E402
from bitsieve_fastdllm.eval.common import encode_prompt, load_fast_dllm  # noqa: E402
from bitsieve_fastdllm.runtime.generator import BitSieveGenerator  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/proposed_a_k4v4_k512.yaml")
    p.add_argument("--benchmark", default="gsm8k")
    p.add_argument("--device", default="cuda")
    p.add_argument("--coverage", action="store_true", help="enable honest coverage scoring")
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    args = p.parse_args()

    raw = ExperimentConfig.load(ROOT / args.config).to_dict()
    budget = max_new_tokens_for(args.benchmark)
    raw["generation"]["max_new_tokens"] = budget
    if args.coverage:
        raw["coverage_diagnostics"] = True
    cfg = ExperimentConfig.from_dict(raw)

    print(f"config           : {cfg.name}")
    print(f"benchmark        : {args.benchmark}")
    print(f"max_new_tokens   : {cfg.generation.max_new_tokens}  (expect 2048 for math)")
    print(f"residual_tokens  : {cfg.quant.residual_tokens}  (expect 0)")
    print(f"dense_prefix_lyr : {cfg.selector.dense_prefix_layers}  (expect 0)")
    print(f"topk             : {cfg.selector.topk}  pct={cfg.selector.topk_percent}")
    print(f"coverage_diag    : {cfg.coverage_diagnostics}", flush=True)

    model, tokenizer = load_fast_dllm(args.model, dtype=torch.bfloat16, device=args.device)
    examples = load_benchmark(args.benchmark, tokenizer=tokenizer, limit=1, split="test")
    ex = examples[0]
    input_ids = encode_prompt(
        tokenizer, ex.prompt,
        max_input_tokens=cfg.max_cache_tokens - cfg.generation.max_new_tokens,
        use_chat_template=True, device=next(model.parameters()).device,
    )
    print(f"prompt_tokens    : {int(input_ids.shape[1])}", flush=True)

    gen = BitSieveGenerator(model, tokenizer, cfg)
    result = gen.generate(input_ids)
    m = result.metrics

    keys = [
        "cache_tokens", "blocks_sparse", "blocks_dense_bypass",
        "sparse_block_fraction", "sparse_layer_step_fraction",
        "generated_tokens_per_request", "nfe", "decode_ms", "tokens_per_second",
        "compact_cache_bytes", "resident_cache_bytes",
        "cache_compression_ratio", "packed_only_compression_ratio",
    ]
    print("\n--- runtime ---")
    for k in keys:
        print(f"  {k:32s} {m.get(k)}")
    print(f"  {'packed_cache.key_residual':32s} {m['packed_cache']['key_residual']}")
    print(f"  {'packed_cache.value_residual':32s} {m['packed_cache']['value_residual']}")
    print(f"  {'coverage':32s} {json.dumps(m.get('coverage'))}")

    tr = result.trace
    layers_selected = sorted({r.layer for r in tr.selections}) if tr else []
    print(f"\nlayers that ran selection: {layers_selected[:6]}... n={len(layers_selected)}")

    print("\n--- assertions ---")
    fails = 0

    def chk(name, ok, detail=""):
        nonlocal fails
        if not ok:
            fails += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")

    chk("math budget is 2048", cfg.generation.max_new_tokens == 2048)
    chk("no fp16 tail in packed cache",
        m["packed_cache"]["key_residual"] == 0 and m["packed_cache"]["value_residual"] == 0)
    chk("sparse path engaged", (m.get("blocks_sparse") or 0) > 0,
        f"sparse={m.get('blocks_sparse')} bypass={m.get('blocks_dense_bypass')}")
    chk("layers 0 and 1 take part in selection",
        0 in layers_selected and 1 in layers_selected)
    chk("compact buffers are counted", (m.get("compact_cache_bytes") or 0) > 0)
    chk("compression ratio is the honest (resident) one",
        m.get("cache_compression_ratio") is not None
        and m["cache_compression_ratio"] <= m["packed_only_compression_ratio"])
    if args.coverage:
        cov = m.get("coverage")
        chk("coverage was scored", cov is not None and cov["cells"] > 0)
        if cov:
            chk("coverage mass <= 1 (reference is the ceiling, not self-scored)",
                cov["mass_mean"] <= 1.0 + 1e-6, f"mass_mean={cov['mass_mean']:.4f}")

    print(f"\ngenerated (first 300 chars):\n{result.texts[0][:300]}")
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

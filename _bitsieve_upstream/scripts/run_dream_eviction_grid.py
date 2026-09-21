#!/usr/bin/env python3
"""Eviction study on DreamReasoner-8B.

Same seven arms as scripts/run_eviction_grid.py, but Dream needs its own runner
for two reasons: its remote code requires transformers 5.x while Fast-dLLM-v2
pins 4.53.1, so the two cannot share an environment; and the 8B checkpoint is
worth loading once for all arms rather than once per subprocess.

  PYTHONPATH=src python scripts/run_dream_eviction_grid.py \
      --device cuda:0 --limit 60 --out results/dream_eviction
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import ExperimentConfig  # noqa: E402
from bitsieve_fastdllm.eval.benchmarks import load_benchmark  # noqa: E402
from bitsieve_fastdllm.eval.common import encode_prompt  # noqa: E402
from bitsieve_fastdllm.eval.metrics import score_prediction  # noqa: E402
from bitsieve_fastdllm.runtime.dream_generator import DreamBitSieveGenerator  # noqa: E402

MODEL_ID = "Dream-org/DreamReasoner-8B"
BASE_CONFIG = "configs/proposed_a_k4v4_k512.yaml"
ARMS = [
    ("dense_bf16", 16, 16, "none", None),
    ("bf16__evict-none", 16, 16, "none", None),
    ("bf16__evict-recent", 16, 16, "recent", None),
    ("bf16__evict-ema", 16, 16, "ema_recent", None),
    ("k4v4__evict-none", 4, 4, "none", None),
    ("k4v4__evict-recent", 4, 4, "recent", None),
    ("k4v4__evict-ema", 4, 4, "ema_recent", None),
]


def build_config(args, k_bits, v_bits, policy, arm) -> ExperimentConfig:
    raw = ExperimentConfig.load(ROOT / BASE_CONFIG).to_dict()
    raw["quant"].update(k_bits=k_bits, v_bits=v_bits)
    # A fixed budget: a percent budget on a short math prompt collapses to a
    # handful of tokens and hides the question from the model, which reads as a
    # quality collapse that has nothing to do with eviction.
    raw["selector"]["topk"] = args.topk
    raw["selector"]["topk_percent"] = None
    raw["generation"].update(
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        threshold=args.threshold,
        top_p=1.0,
        temperature=0.0,
    )
    raw["eviction"] = {
        "policy": policy,
        "decay": args.decay,
        "recent_window": args.window,
        "capacity_percent": args.capacity_percent,
        "capacity_floor": args.capacity_floor,
        "interval_blocks": args.interval,
    }
    raw["coverage_diagnostics"] = args.coverage and policy != "dense"
    raw["name"] = arm
    if arm == "dense_bf16":
        raw["semantic"] = "dense"
        raw["eviction"] = {"policy": "none"}
        raw["coverage_diagnostics"] = False
    return ExperimentConfig.from_dict(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--benchmark", default="math500")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--capacity-floor", type=int, default=256)
    ap.add_argument("--capacity-percent", type=float, default=5.0)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--interval", type=int, default=4)
    ap.add_argument("--coverage", action="store_true", default=True)
    ap.add_argument("--no-coverage", dest="coverage", action="store_false")
    ap.add_argument("--arms", help="comma-separated subset")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "executable": sys.executable,
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "model": MODEL_ID,
                "device": args.device,
                "settings": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(args.device)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        .to(args.device)
        .eval()
    )
    examples = load_benchmark(args.benchmark, tokenizer=tok, limit=args.limit, split="test")
    print(f"{MODEL_ID} on {args.device}: {len(examples)} examples")

    wanted = {a.strip() for a in args.arms.split(",")} if args.arms else None
    for arm, kb, vb, policy, _ in ARMS:
        if wanted and arm not in wanted:
            continue
        dest = out / f"{arm}.jsonl"
        done = set()
        if dest.exists():
            done = {
                json.loads(l)["id"] for l in dest.read_text(encoding="utf-8").splitlines() if l.strip()
            }
            if len(done) >= len(examples):
                print(f"skip (complete) {arm}")
                continue
        cfg = build_config(args, kb, vb, policy, arm)
        gen = DreamBitSieveGenerator(model, tok, cfg)
        t0 = time.time()
        try:
            with dest.open("a", encoding="utf-8") as fh:
                for i, ex in enumerate(examples):
                    if ex.example_id in done:
                        continue
                    ids = encode_prompt(
                        tok, ex.prompt, max_input_tokens=cfg.max_cache_tokens - args.max_new_tokens,
                        use_chat_template=True, device=args.device,
                    )
                    r = gen.generate(ids)
                    fh.write(json.dumps({
                        "id": ex.example_id,
                        "score": score_prediction(args.benchmark, r.texts[0], ex.references),
                        "prediction": r.texts[0],
                        "references": ex.references,
                        "runtime": r.metrics,
                        "config": cfg.to_dict(),
                    }, ensure_ascii=False) + "\n")
                    fh.flush()
                    if (i + 1) % 10 == 0:
                        print(f"  {arm}: {i + 1}/{len(examples)} "
                              f"({(time.time() - t0) / (i + 1):.1f}s/ex)", flush=True)
        finally:
            gen.patch.unpatch()
        print(f"== {arm} done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

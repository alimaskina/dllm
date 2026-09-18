#!/usr/bin/env python3
"""Precision-vs-budget coverage sweep: is it better to keep MORE entries at LOWER precision?

An arm holding k entries at b bits costs b*k bits of cache, so 4-bit at 4x the
entries costs exactly what fp16 at 1x does. The sweep asks which side of that
trade retains more of the model's attention mass.

    fp16              4bit      3bit      2bit          <- 1x entries
                      4bit x2   3bit x2   2bit x2       <- 2x
                      4bit x4   3bit x4   2bit x4       <- 4x

Baselines (1x):  GSM8K k=32   |   MuSiQue and HotpotQA rho=2.5% of the prefix

Every arm is scored on the SAME queries and keys, from one generation pass per
task: the run generates with the fp16 baseline arm, and each arm's selection is
then evaluated against the same fp16 attention reference at every layer and
block. Running each arm's own generation instead would let the trajectories
diverge after the first block, and the coverages would stop being measurements
of the same thing.

3-bit has no packed kernel here (3 bits straddle byte boundaries), so arms are
scored through simulate_key_quantization, which is bit-exact with the packed
path at 2 and 4 bits and defines the same grid at 3. It is a quality
measurement, not a speed or memory one - nothing in this script stores anything
compactly. The memory column is arithmetic: bits x entries.

    python scripts/coverage_sweep.py --tasks gsm8k,musique,hotpotqa -n 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import ExperimentConfig  # noqa: E402
from bitsieve_fastdllm.eval.benchmarks import (  # noqa: E402
    MATH_BENCHMARKS,
    load_benchmark,
    max_new_tokens_for,
    uses_chat_template,
)
from bitsieve_fastdllm.eval.common import encode_prompt, load_fast_dllm  # noqa: E402
from bitsieve_fastdllm.eval.metrics import score_prediction  # noqa: E402
from bitsieve_fastdllm.runtime.generator import BitSieveGenerator  # noqa: E402

BITS = (4, 3, 2)
MULTIPLIERS = (1, 2, 4)
BASE_CONFIG = "configs/suite/sparse_fp16_all_p5.yaml"

# Base budgets, as given: a fixed entry count for math (its prompts are ~100
# tokens, so a percentage degenerates), a percentage for LongBench.
BASE_TOPK = 32
BASE_PCT = 2.5


def build_arms(benchmark: str) -> list[dict]:
    """The 10 arms, as a budget the config understands."""
    fixed = benchmark in MATH_BENCHMARKS

    def budget(mult: int) -> dict:
        return {"topk": BASE_TOPK * mult} if fixed else {"topk_percent": BASE_PCT * mult}

    arms = [{"name": "fp16", "bits": 16, **budget(1)}]
    for mult in MULTIPLIERS:
        for bits in BITS:
            suffix = "" if mult == 1 else f"_x{mult}"
            arms.append({"name": f"{bits}bit{suffix}", "bits": bits, **budget(mult)})
    return arms


def relative_memory(arm: dict, benchmark: str) -> float:
    """Cache bits an arm costs, relative to the fp16 baseline arm."""
    fixed = benchmark in MATH_BENCHMARKS
    base = BASE_TOPK if fixed else BASE_PCT
    entries = arm.get("topk") if fixed else arm.get("topk_percent")
    return (arm["bits"] * float(entries)) / (16.0 * base)


def build_config(benchmark: str, arms: list[dict]) -> ExperimentConfig:
    raw = ExperimentConfig.load(ROOT / BASE_CONFIG).to_dict()
    # Generation runs on the fp16 baseline arm; every other arm is scored against
    # the same reference rather than generating its own trajectory.
    base = arms[0]
    raw["selector"]["topk"] = base.get("topk", raw["selector"]["topk"])
    raw["selector"]["topk_percent"] = base.get("topk_percent")
    raw["generation"]["max_new_tokens"] = max_new_tokens_for(benchmark)
    raw["coverage_diagnostics"] = True
    raw["coverage_arms"] = arms
    raw["name"] = f"coverage_sweep_{benchmark}"
    return ExperimentConfig.from_dict(raw)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", default="gsm8k,musique,hotpotqa")
    p.add_argument("-n", "--num-examples", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--revision", default="0661abf5f9f0ee338970d091052a26c8efa51974")
    p.add_argument("--output-root", default="results/coverage_sweep")
    args = p.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_fast_dllm(
        args.model, dtype=torch.bfloat16, device=args.device, revision=args.revision
    )
    device = next(model.parameters()).device

    for benchmark in tasks:
        arms = build_arms(benchmark)
        cfg = build_config(benchmark, arms)
        examples = load_benchmark(
            benchmark, tokenizer=tokenizer, limit=args.num_examples, split="test"
        )
        out_path = out_root / f"{benchmark}.jsonl"
        done = set()
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(str(json.loads(line)["id"]))
        print(
            f"\n=== {benchmark}: {len(examples)} examples, {len(arms)} arms"
            + (f", resuming past {len(done)}" if done else "")
            + " ===",
            flush=True,
        )

        generator = BitSieveGenerator(model, tokenizer, cfg)
        with out_path.open("a", encoding="utf-8") as handle:
            for i, example in enumerate(examples, 1):
                if str(example.example_id) in done:
                    continue
                max_input = cfg.max_cache_tokens - cfg.generation.max_new_tokens
                input_ids = encode_prompt(
                    tokenizer,
                    example.prompt,
                    max_input_tokens=max_input,
                    use_chat_template=uses_chat_template(benchmark),
                    device=device,
                )
                t0 = time.time()
                result = generator.generate(input_ids)
                trace = result.trace
                summary = trace.coverage_arms_summary() if trace is not None else None
                if not summary:
                    raise SystemExit(
                        "no coverage arms were recorded - the selector never ran "
                        "sparsely on this example (check the budget against the "
                        "prompt length)"
                    )
                row = {
                    "id": example.example_id,
                    "benchmark": benchmark,
                    "prompt_tokens": int(input_ids.shape[1]),
                    "score": score_prediction(
                        benchmark,
                        result.texts[0],
                        example.references,
                        all_classes=example.metadata.get("all_classes"),
                    ),
                    "wall_s": time.time() - t0,
                    "arms": summary,
                    "config": cfg.to_dict(),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                best = max(summary.items(), key=lambda kv: kv[1]["mass_abs_mean"])
                print(
                    f"  [{i}/{len(examples)}] {example.example_id} "
                    f"{time.time() - t0:.1f}s  best arm: {best[0]} "
                    f"mass_abs={best[1]['mass_abs_mean']:.4f}",
                    flush=True,
                )

    print(f"\nDone. Report with:  python scripts/report_coverage_sweep.py {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

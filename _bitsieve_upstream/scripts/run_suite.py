#!/usr/bin/env python3
"""One-button quality + coverage + speed suite across four method variants.

Runs GSM8K and a spread of LongBench tasks through:

  dense                 official dense BF16, no selector           (ceiling)
  sparse_fp16_all       BF16 cache, ALL masked queries rank the top-k (MAGE-style)
  sparse_fp16_middle    BF16 cache, ONE center query ranks the top-k  (HERALD-proxy)
  sparse_k4v4_all       KIVI K4/V4 cache, ALL masked queries rank the top-k

All three selector configs use a percent-of-prefix budget (5% by default), not
a fixed k - a fixed k silently runs dense on short prompts (see
docs/methodology.md, "A fixed budget does not always engage"), which would
make this comparison meaningless on GSM8K's short prompts. `sparse_block_fraction`
is checked and printed per row specifically to catch that regression if a
config is ever changed back to a fixed k.

Quality/speed and coverage are measured in SEPARATE passes for every selector
variant: coverage_diagnostics keeps a shadow fp16 key cache and costs real
time, so a speed number taken while it is on would not describe the deployed
path (see docs/methodology.md). The quality pass is what tokens/s and TPOB are
read from; the coverage pass is diagnostic only and does not affect scoring.

Usage
-----
    bash scripts/run_suite.sh                       # the button: full 15+15 run
    python scripts/run_suite.py --smoke              # ~2 min correctness check
    python scripts/run_suite.py --gsm8k-n 15 \\
        --longbench-tasks 2wikimqa,qmsum,repobench-p --longbench-n 5 \\
        --variants dense,sparse_fp16_all,sparse_fp16_middle,sparse_k4v4_all \\
        --device cuda:0 --output-root results/suite_run

Re-running with the same --output-root resumes: any (variant, pass, benchmark,
example_id) already present in its JSONL is skipped, so a killed run or an
added variant does not redo finished work. Delete the output-root (or use a
new one) for a clean run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import ExperimentConfig  # noqa: E402
from bitsieve_fastdllm.eval.benchmarks import (  # noqa: E402
    LONG_BENCH_CONFIGS,
    load_benchmark,
    max_new_tokens_for,
)
from bitsieve_fastdllm.eval.common import encode_prompt, load_fast_dllm  # noqa: E402
from bitsieve_fastdllm.eval.metrics import score_prediction  # noqa: E402
from bitsieve_fastdllm.runtime.generator import BitSieveGenerator  # noqa: E402
from bitsieve_fastdllm.runtime.official_generator import OfficialDenseGenerator  # noqa: E402

SUITE_CONFIG_DIR = ROOT / "configs" / "suite"

# name -> (config file stem, has a selector at all)
VARIANTS: dict[str, tuple[str, bool]] = {
    "dense": ("dense_bf16", False),
    "sparse_fp16_all": ("sparse_fp16_all_p5", True),
    "sparse_fp16_middle": ("sparse_fp16_middle_p5", True),
    "sparse_k4v4_all": ("sparse_k4v4_all_p5", True),
}

DEFAULT_LONGBENCH_TASKS = ["2wikimqa", "qmsum", "repobench-p"]


def build_generator(cfg: ExperimentConfig, model, tokenizer):
    if cfg.engine == "official":
        return OfficialDenseGenerator(model, tokenizer, cfg)
    return BitSieveGenerator(model, tokenizer, cfg)


def load_config(stem: str, *, topk_pct: float, coverage: bool) -> ExperimentConfig:
    raw = ExperimentConfig.load(SUITE_CONFIG_DIR / f"{stem}.yaml").to_dict()
    if raw["selector"].get("topk_percent") is not None:
        raw["selector"]["topk_percent"] = topk_pct
    raw["coverage_diagnostics"] = coverage and raw["semantic"] != "dense"
    raw["name"] = f"{raw['name']}{'_covpass' if coverage else ''}"
    return ExperimentConfig.from_dict(raw)


def _done_keys(path: Path) -> set[tuple[str, str, str]]:
    """Rows to skip on resume. A failed row is deliberately NOT counted as
    done - a transient hiccup (shared-GPU noise, one-off kernel/driver error)
    should be retried on the next run rather than permanently marked skipped."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("runtime") is None:
            continue
        done.add((row["variant"], row["pass"], row["benchmark"], str(row["id"])))
    return done


def run_pass(
    *,
    variant: str,
    pass_name: str,
    cfg: ExperimentConfig,
    model,
    tokenizer,
    benchmark: str,
    examples: list,
    output_path: Path,
    device: str,
) -> None:
    done = _done_keys(output_path)
    generator = build_generator(cfg, model, tokenizer)
    max_new = max_new_tokens_for(benchmark)
    if cfg.generation.max_new_tokens != max_new:
        raw = cfg.to_dict()
        raw["generation"]["max_new_tokens"] = max_new
        cfg = ExperimentConfig.from_dict(raw)
        generator = build_generator(cfg, model, tokenizer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = [e for e in examples if (variant, pass_name, benchmark, str(e.example_id)) not in done]
    if not pending:
        print(f"  [{variant}/{pass_name}] {benchmark}: already complete ({len(examples)} rows)", flush=True)
        return
    print(
        f"  [{variant}/{pass_name}] {benchmark}: running {len(pending)}/{len(examples)} "
        f"(max_new_tokens={max_new})",
        flush=True,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        for i, example in enumerate(pending, 1):
            max_input = cfg.max_cache_tokens - cfg.generation.max_new_tokens
            input_ids = encode_prompt(
                tokenizer, example.prompt, max_input_tokens=max_input,
                use_chat_template=True, device=device,
            )
            t0 = time.time()
            # A single example failing (a transient kernel/driver hiccup on a
            # shared GPU, an OOM on one unusually long prompt) must not lose
            # the whole suite's results. The failure is recorded - not
            # silently skipped - so it is visible in the JSONL and excluded
            # from mean_score rather than counted as 0 or as a pass.
            try:
                result = generator.generate(input_ids)
            except Exception as exc:  # noqa: BLE001
                wall_s = time.time() - t0
                print(
                    f"    [{i}/{len(pending)}] id={example.example_id} FAILED after "
                    f"{wall_s:.1f}s: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                row = {
                    "variant": variant, "pass": pass_name, "benchmark": benchmark,
                    "id": example.example_id, "prompt_tokens": int(input_ids.shape[1]),
                    "score": None, "wall_s": wall_s, "runtime": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            wall_s = time.time() - t0
            prediction = result.texts[0]
            score = score_prediction(benchmark, prediction, example.references)
            row = {
                "variant": variant,
                "pass": pass_name,
                "benchmark": benchmark,
                "id": example.example_id,
                "prompt_tokens": int(input_ids.shape[1]),
                "score": score,
                "wall_s": wall_s,
                "runtime": result.metrics,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"    [{i}/{len(pending)}] id={example.example_id} score={score:.3f} "
                f"{wall_s:.1f}s tok/s={result.metrics.get('tokens_per_second')}",
                flush=True,
            )


def summarize(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("no rows to summarize")
        return

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["variant"], row["pass"], row["benchmark"]), []).append(row)

    def med(values: list[float]) -> float | None:
        vals = [v for v in values if v is not None]
        return statistics.median(vals) if vals else None

    print(f"\n{'variant':<20} {'pass':<9} {'benchmark':<14} {'n':>3} {'fail':>4} {'score':>7} "
          f"{'sparse_frac':>11} {'cov_mass':>9} {'cov_ovl':>8} {'tok/s':>7} {'tpob_ms':>8}")
    summary_rows = []
    for (variant, pass_name, benchmark), rs in sorted(groups.items()):
        ok_rows = [r for r in rs if r.get("runtime") is not None]
        n_fail = len(rs) - len(ok_rows)
        scores = [r["score"] for r in ok_rows if r.get("score") is not None]
        sparse_frac = med([r["runtime"].get("sparse_block_fraction") for r in ok_rows])
        cov = [r["runtime"].get("coverage") for r in ok_rows if r["runtime"].get("coverage")]
        cov_mass = med([c["mass_mean"] for c in cov]) if cov else None
        cov_ovl = med([c["overlap_mean"] for c in cov]) if cov else None
        tps = med([r["runtime"].get("tokens_per_second") for r in ok_rows])
        tpob = med([r["runtime"].get("mean_tpob_ms") for r in ok_rows])
        line = {
            "variant": variant, "pass": pass_name, "benchmark": benchmark, "n": len(rs),
            "n_failed": n_fail,
            "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
            "median_sparse_block_fraction": sparse_frac,
            "median_coverage_mass": cov_mass,
            "median_coverage_overlap": cov_ovl,
            "median_tokens_per_second": tps,
            "median_tpob_ms": tpob,
        }
        summary_rows.append(line)
        score_str = "" if line["mean_score"] is None else f"{line['mean_score']:.3f}"
        print(
            f"{variant:<20} {pass_name:<9} {benchmark:<14} {len(rs):>3} {n_fail:>4} "
            f"{score_str:>7} "
            f"{('' if sparse_frac is None else f'{sparse_frac:.2f}'):>11} "
            f"{('' if cov_mass is None else f'{cov_mass:.3f}'):>9} "
            f"{('' if cov_ovl is None else f'{cov_ovl:.3f}'):>8} "
            f"{('' if tps is None else f'{tps:.1f}'):>7} "
            f"{('' if tpob is None else f'{tpob:.0f}'):>8}"
        )

    (output_root / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    warn = [
        r for r in summary_rows
        if r["pass"] == "quality" and r["variant"] != "dense"
        and r["median_sparse_block_fraction"] is not None
        and r["median_sparse_block_fraction"] < 0.99
    ]
    if warn:
        print("\n/!\\ sparse path did not fully engage for:")
        for r in warn:
            print(f"    {r['variant']} on {r['benchmark']}: "
                  f"sparse_block_fraction={r['median_sparse_block_fraction']:.2f}")

    failed = [r for r in summary_rows if r["n_failed"] > 0]
    if failed:
        print("\n/!\\ some examples failed (re-run the same command to retry them):")
        for r in failed:
            print(f"    {r['variant']}/{r['pass']} on {r['benchmark']}: {r['n_failed']}/{r['n']} failed")

    print(f"\nSaved -> {output_root / 'summary.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gsm8k-n", type=int, default=15)
    p.add_argument("--longbench-tasks", default=",".join(DEFAULT_LONGBENCH_TASKS),
                   help=f"comma list from: {', '.join(sorted(set(LONG_BENCH_CONFIGS.values())))}")
    p.add_argument("--longbench-n", type=int, default=5, help="examples per LongBench task")
    p.add_argument("--variants", default=",".join(VARIANTS),
                   help=f"comma list from: {', '.join(VARIANTS)}")
    p.add_argument("--topk-pct", type=float, default=5.0)
    p.add_argument("--skip-coverage", action="store_true",
                   help="quality+speed pass only; skip the separate coverage pass")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--output-root", default="results/suite_run")
    p.add_argument("--smoke", action="store_true", help="--gsm8k-n 1 --longbench-n 1, one task")
    args = p.parse_args()

    if args.smoke:
        args.gsm8k_n = 1
        args.longbench_n = 1
        if args.longbench_tasks == ",".join(DEFAULT_LONGBENCH_TASKS):
            args.longbench_tasks = "2wikimqa"

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}, choose from {list(VARIANTS)}")
    longbench_tasks = [t.strip() for t in args.longbench_tasks.split(",") if t.strip()]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"variants        : {variants}")
    print(f"gsm8k           : n={args.gsm8k_n}")
    print(f"longbench       : tasks={longbench_tasks} n_per_task={args.longbench_n}")
    print(f"topk_pct        : {args.topk_pct}")
    print(f"coverage pass   : {'skipped' if args.skip_coverage else 'enabled'}")
    print(f"output_root     : {output_root}", flush=True)

    model, tokenizer = load_fast_dllm(args.model, dtype=torch.bfloat16, device=args.device)

    benchmarks: dict[str, list] = {"gsm8k": load_benchmark("gsm8k", tokenizer=tokenizer, limit=args.gsm8k_n, split="test")}
    for task in longbench_tasks:
        benchmarks[task] = load_benchmark(task, tokenizer=tokenizer, limit=args.longbench_n, split="test")

    t_start = time.time()
    for variant in variants:
        stem, has_selector = VARIANTS[variant]
        passes = [("quality", False)]
        if has_selector and not args.skip_coverage:
            passes.append(("coverage", True))
        for pass_name, coverage in passes:
            cfg = load_config(stem, topk_pct=args.topk_pct, coverage=coverage)
            for benchmark, examples in benchmarks.items():
                out_path = output_root / f"{variant}.jsonl"
                run_pass(
                    variant=variant, pass_name=pass_name, cfg=cfg,
                    model=model, tokenizer=tokenizer, benchmark=benchmark,
                    examples=examples, output_path=out_path, device=args.device,
                )

    print(f"\ntotal wall time: {time.time() - t_start:.0f}s")
    summarize(output_root)


if __name__ == "__main__":
    main()

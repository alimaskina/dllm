#!/usr/bin/env python3
"""Recovery evaluation grid — one branch (optionally a LoRA adapter) over the spec grid.

  GSM8K, MATH500   : dense bf16 ceiling + b=4 x k in {32, 64}        (in-domain)
  LongBench 2wikimqa / hotpotqa / musique / narrativeqa
                   : dense bf16 ceiling + b=4 x rho in {2.5, 5}%     (out-of-domain;
                     not a single long prompt appears in training)

Each cell reports score, coverage (share of the reference attention mass the
selected old-cache tokens carry) and resident cache bytes, so the quality
number is always read next to what it cost.

Decoding is the *same* ``batch_sample_sparse_kv`` path the original study uses;
this script only swaps the weights (base, or base + adapter) underneath it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent / "sparse_kv_exp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import ExperimentConfig, PrecisionConfig, SelectorConfig  # noqa: E402
from eval_utils import attach_sampler, set_seed  # noqa: E402
from logging_utils import append_jsonl  # noqa: E402
from model_registry import load_model_and_tokenizer  # noqa: E402
from model_utils import configure_block_size, default_small_block_size  # noqa: E402

BENCH_TASKS = ("gsm8k", "math500")
LONGBENCH_TASKS = ("2wikimqa", "hotpotqa", "musique", "narrativeqa")
BENCH_TOPKS = (32, 64)
LONGBENCH_PCTS = (2.5, 5.0)
EXEC_BITS = "4"
# Precision the first-pass selector reads. "fp16" reproduces the original sweep
# grid; "exec" makes the selector rank on the same quantized cache execution
# uses, which is the only setting where the 4-bit cache is an actual memory
# saving (otherwise both views stay resident — see README "Resident bytes").
SELECTOR_BITS = "fp16"

_COMMON = dict(
    block_size=32,
    small_block_size=8,
    threshold=1.0,
    max_new_tokens=2048,
    log_selected_indices=False,
    save_full_cost_steps=False,
)


def dense_config() -> ExperimentConfig:
    """bf16 ceiling: no sparsity, no quantization, upstream decoding."""
    return ExperimentConfig(
        name="dense_bf16", baseline="original", sparse_old_cache=False, **_COMMON
    )


def sparse_config(*, topk: int | None = None, topk_pct: float | None = None) -> ExperimentConfig:
    sel: dict = dict(mode="all_mean", per_head=True)
    if topk_pct is not None:
        sel["topk_pct"] = topk_pct
        sel["topk"] = 256
        tag = "p" + f"{topk_pct:g}".replace(".", "p")
    else:
        sel["topk"] = int(topk)
        tag = f"k{topk}"
    if SELECTOR_BITS != "fp16":
        tag += f"_sel{SELECTOR_BITS}"
    return ExperimentConfig(
        name=f"b{EXEC_BITS}_{tag}",
        baseline="sparse_quant_kv",
        sparse_old_cache=True,
        selector=SelectorConfig(**sel),
        selector_precision=PrecisionConfig(
            k_bits=SELECTOR_BITS, v_bits=SELECTOR_BITS, q_bits="fp16"
        ),
        exec_precision=PrecisionConfig(k_bits=EXEC_BITS, v_bits=EXEC_BITS, q_bits="fp16"),
        **_COMMON,
    )


def configs_for(task: str) -> list[ExperimentConfig]:
    if task in BENCH_TASKS:
        return [dense_config()] + [sparse_config(topk=k) for k in BENCH_TOPKS]
    return [dense_config()] + [sparse_config(topk_pct=p) for p in LONGBENCH_PCTS]


def _done_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["config"]["name"], r["task"], r["example_id"]))
    return out


def summarize(records: list[dict]) -> dict:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by[(r["task"], r["config"]["name"])].append(r)

    rows = []
    for (task, cfg), rs in sorted(by.items()):
        scores = [r["score"] for r in rs]
        covs = [r["avg_topk_coverage"] for r in rs if r.get("avg_topk_coverage") is not None]
        def _cost(key):
            return [r["cost"][key] for r in rs if r.get("cost") and r["cost"].get(key)]

        res = _cost("kv_resident_bytes_per_cache_token")
        res_ratio = _cost("kv_resident_ratio_vs_dense_fp16")
        res_exec = _cost("kv_resident_bytes_exec_view")
        two_views = any(
            r.get("cost", {}).get("kv_resident_keeps_two_views") for r in rs
        )
        rows.append(
            {
                "task": task,
                "config": cfg,
                "n": len(rs),
                "score": sum(scores) / len(scores) if scores else None,
                "coverage": sum(covs) / len(covs) if covs else None,
                "resident_bytes_per_token": sum(res) / len(res) if res else None,
                "resident_ratio_vs_bf16": sum(res_ratio) / len(res_ratio) if res_ratio else None,
                "keeps_two_cache_views": two_views,
            }
        )
    return {"rows": rows}


def write_report(analysis: dict, path: Path, branch: str) -> None:
    lines = [
        f"# Recovery grid — {branch}",
        "",
        "`coverage` = share of the reference attention mass held by the selected "
        "old-cache tokens. `resident` = old-cache bytes that must stay live per "
        "cache token, quantization scales/zero-points included; `2 views` marks "
        "configurations whose selector precision differs from execution, so both "
        "cache copies have to be kept and the ratio to bf16 is not a saving.",
        "",
        "| task | config | n | score | coverage | resident B/token | resident / bf16 | 2 views |",
        "|---|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in analysis["rows"]:
        cov = f"{r['coverage']:.3f}" if r["coverage"] is not None else "—"
        bt = f"{r['resident_bytes_per_token']:.0f}" if r["resident_bytes_per_token"] else "—"
        rr = f"{r['resident_ratio_vs_bf16']:.3f}" if r["resident_ratio_vs_bf16"] else "—"
        sc = f"{r['score']:.4f}" if r["score"] is not None else "—"
        tv = "yes" if r.get("keeps_two_cache_views") else ""
        lines.append(
            f"| {r['task']} | {r['config']} | {r['n']} | {sc} | {cov} | {bt} | {rr} | {tv} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True, help="label for this run, e.g. A_none or D_sft_noise")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for branch A")
    ap.add_argument("--tasks", nargs="+", default=list(BENCH_TASKS) + list(LONGBENCH_TASKS))
    ap.add_argument("--num-examples", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument(
        "--selector-precision", choices=("fp16", "exec"), default="fp16",
        help="fp16 reproduces the original grid (two cache views stay resident); "
             "exec ranks on the quantized cache, the only real memory saving",
    )
    args = ap.parse_args()

    global SELECTOR_BITS
    SELECTOR_BITS = EXEC_BITS if args.selector_precision == "exec" else "fp16"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "results.jsonl"

    if args.analyze_only:
        recs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
        an = summarize(recs)
        (out_dir / "analysis.json").write_text(json.dumps(an, indent=2), encoding="utf-8")
        write_report(an, out_dir / "report.md", args.branch)
        print(f"report -> {out_dir/'report.md'}")
        return

    set_seed(args.seed)
    device = torch.device(args.device)
    model, tokenizer, upstream = load_model_and_tokenizer("fast_dllm_v2_7b", device)
    model.eval()

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()     # fold LoRA in; decoding sees plain weights
        model.eval()
        print(f"[eval] merged adapter {args.adapter}")
    else:
        print("[eval] no adapter (branch A / training-free baseline)")

    # run_one lives in the original harness; import after the model exists so the
    # heavy deps (lm_eval) are only paid once.
    from run_benchmark_sweep import run_one as bench_run_one
    from run_longbench import run_one as lb_run_one
    from benchmark_utils import load_benchmark_samples
    from longbench_utils import load_longbench_samples

    done = _done_keys(jsonl)
    meta = {
        "branch": args.branch, "adapter": args.adapter, "tasks": args.tasks,
        "num_examples": args.num_examples, "seed": args.seed,
        "bench_topks": BENCH_TOPKS, "longbench_pcts": LONGBENCH_PCTS,
        "exec_bits": EXEC_BITS, "selector_bits": SELECTOR_BITS,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for task in args.tasks:
        is_bench = task in BENCH_TASKS
        samples = (
            load_benchmark_samples(task, num_examples=args.num_examples, seed=args.seed)
            if is_bench
            else load_longbench_samples(task, num_examples=args.num_examples, seed=args.seed)
        )
        for cfg in configs_for(task):
            configure_block_size(model, cfg.block_size)
            attach_sampler(model, cfg, upstream)
            pending = [s for s in samples if (cfg.name, task, s["idx"]) not in done]
            print(f"\n=== {task} / {cfg.name}: {len(pending)} of {len(samples)} to run")
            for s in pending:
                t0 = time.time()
                rec = (
                    bench_run_one(model, tokenizer, s, cfg, task=task)
                    if is_bench
                    else lb_run_one(model, tokenizer, s, cfg)
                )
                rec["branch"] = args.branch
                append_jsonl(jsonl, rec)
                print(f"  ex{s['idx']:<4d} score={rec['score']:.3f} "
                      f"cov={rec['avg_topk_coverage']} {time.time()-t0:.1f}s")

    recs = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    an = summarize(recs)
    (out_dir / "analysis.json").write_text(json.dumps(an, indent=2), encoding="utf-8")
    write_report(an, out_dir / "report.md", args.branch)
    print(f"\nreport -> {out_dir/'report.md'}")


if __name__ == "__main__":
    main()

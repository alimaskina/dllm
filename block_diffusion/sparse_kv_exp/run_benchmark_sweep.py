#!/usr/bin/env python3
"""Benchmark sweep: GSM8K / MATH500 / GPQA — same grid as LongBench (per-head)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from benchmark_utils import (  # noqa: E402
    DEFAULT_BENCHMARK_TASKS,
    build_chat_prompt,
    extract_prediction,
    grade_sample,
    load_benchmark_samples,
    max_new_tokens_for_task,
)
from config import ExperimentConfig  # noqa: E402
from eval_utils import attach_sampler, set_seed  # noqa: E402
from logging_utils import append_jsonl  # noqa: E402
from handoff_sweep_configs import HANDOFF_FAMILIES, HANDOFF_NUM_EXAMPLES, HANDOFF_TOPK_PCTS, select_handoff_configs  # noqa: E402
from longbench_sweep_configs import TOPK_PCTS, TOPKS, select_sweep_configs  # noqa: E402
from model_registry import get_model_spec, load_model_and_tokenizer  # noqa: E402
from model_utils import configure_block_size, default_small_block_size  # noqa: E402
from run_longbench import _done_keys  # noqa: E402
from run_longbench_sweep import (  # noqa: E402
    _resolve_configs,
    summarize_sweep,
    write_sweep_report,
)


def _select_configs(
    sweep: str,
    topks: tuple[int, ...],
    topk_pcts: tuple[float, ...],
    families: list[str],
) -> tuple[list[ExperimentConfig], tuple[int, ...], tuple[float, ...]]:
    return _resolve_configs(
        sweep=sweep,
        topks=topks,
        topk_pcts=topk_pcts,
        families=families,
    )


@torch.no_grad()
def run_one(
    model,
    tokenizer,
    sample: dict,
    exp_cfg: ExperimentConfig,
    *,
    task: str,
) -> dict:
    max_gen = max_new_tokens_for_task(task)
    prompt = build_chat_prompt(tokenizer, sample["question"])
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = int(input_ids.shape[1])

    run_log: list = []
    seq_len = torch.tensor([prompt_len], device=model.device)
    small = exp_cfg.small_block_size or default_small_block_size(exp_cfg.block_size)
    common = dict(
        input_ids=input_ids,
        tokenizer=tokenizer,
        block_size=exp_cfg.block_size,
        small_block_size=small,
        max_new_tokens=max_gen,
        min_len=prompt_len,
        seq_len=seq_len,
        threshold=exp_cfg.threshold,
    )
    if exp_cfg.baseline == "original":
        out = model.mdm_sample(**common)
    else:
        out = model.mdm_sample(**common, exp_config=exp_cfg, experiment_log=run_log)

    output_ids = out[0]
    gen_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
    state = run_log[-1] if run_log else {}
    pred = extract_prediction(sample["grader"], gen_text)
    correct = grade_sample(sample["grader"], sample["gold"], pred)

    covs = [
        layer["attention_mass_captured"]
        for blk in state.get("blocks", [])
        for layer in blk.get("layers", {}).values()
        if "attention_mass_captured" in layer
    ]

    return {
        "task": task,
        "example_id": sample["idx"],
        "question": sample["question"],
        "gold": sample["gold"],
        "prediction": pred,
        "correct": correct,
        "score": float(correct),
        "generated_answer": gen_text,
        "prompt_tokens": prompt_len,
        "gen_tokens": int(output_ids.shape[0] - prompt_len),
        "avg_topk_coverage": sum(covs) / len(covs) if covs else None,
        "config": exp_cfg.to_dict(),
        "cost": state.get("cost_summary", {}),
        "num_blocks": len(state.get("blocks", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        choices=("default", "handoff"),
        default="default",
    )
    parser.add_argument(
        "--model",
        default="fast_dllm_v2_7b",
        help="Model preset: fast_dllm_v2_7b | llada2_mini_16b",
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_BENCHMARK_TASKS)
    parser.add_argument("--topks", nargs="+", type=int, default=list(TOPKS))
    parser.add_argument(
        "--topk-pcts",
        nargs="+",
        type=float,
        default=[],
        help="Cache %% tiers (e.g. 2.5 5 10 20)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["baseline", "middle", "extreme_k2v2", "extreme_k4v2", "extreme_k4v4"],
    )
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--output-dir", type=str, default="results/benchmark_sweep_perhead")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    handoff = args.sweep == "handoff"
    num_examples = args.num_examples
    if num_examples is None:
        num_examples = HANDOFF_NUM_EXAMPLES if handoff else 50
    topks = tuple(args.topks)
    topk_pcts = tuple(args.topk_pcts)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    meta_path = out_dir / "sweep_meta.json"

    if args.analyze_only:
        records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        analysis = summarize_sweep(records)
        (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            handoff = meta.get("sweep") == "handoff"
            topks = tuple(meta.get("topks") or ())
            topk_pcts = tuple(meta.get("topk_pcts") or ())
        elif handoff and not topk_pcts:
            topk_pcts = HANDOFF_TOPK_PCTS
        write_sweep_report(
            analysis,
            out_dir / "report.md",
            topks,
            topk_pcts,
            title="Benchmark Sweep Report",
            handoff=handoff,
        )
        print(f"Report → {out_dir / 'report.md'}")
        return

    configs, report_topks, report_pcts = _select_configs(
        args.sweep, topks, topk_pcts, args.families
    )
    meta_path.write_text(
        json.dumps(
            {
                "sweep": args.sweep,
                "model": args.model,
                "topks": list(report_topks),
                "topk_pcts": list(report_pcts),
                "families": args.families if not handoff else list(HANDOFF_FAMILIES),
                "num_examples": num_examples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    done = _done_keys(jsonl_path)
    set_seed(args.seed)

    device = torch.device(args.device)
    spec = get_model_spec(args.model)
    print(f"Model preset: {args.model} ({spec.hf_id})")
    print(
        f"Configs: {len(configs)}, examples/task: {num_examples}, "
        f"topks: {report_topks or '—'}, topk_pcts: {report_pcts or '—'}"
    )
    model, tokenizer, upstream = load_model_and_tokenizer(args.model, device)
    configure_block_size(model, configs[0].block_size)

    for task in args.tasks:
        samples = load_benchmark_samples(task, num_examples=num_examples, seed=args.seed)
        print(f"\n=== Task {task}: {len(samples)} examples, max_gen={max_new_tokens_for_task(task)} ===")
        for exp_cfg in configs:
            attach_sampler(model, exp_cfg, upstream)
            pending = [s for s in samples if (exp_cfg.name, task, s["idx"]) not in done]
            if not pending:
                print(f"  {exp_cfg.name}: cached")
                continue
            print(f"  RUN {exp_cfg.name} ({len(pending)} ex)")
            for sample in pending:
                t0 = time.time()
                record = run_one(model, tokenizer, sample, exp_cfg, task=task)
                append_jsonl(jsonl_path, record)
                done.add((exp_cfg.name, task, sample["idx"]))
                dt = time.time() - t0
                print(
                    f"    ex{sample['idx']} ok={record['correct']} "
                    f"cov={record['avg_topk_coverage']} prompt={record['prompt_tokens']} "
                    f"gen={record['gen_tokens']} {dt:.1f}s"
                )

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    analysis = summarize_sweep(records)
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    write_sweep_report(
        analysis,
        out_dir / "report.md",
        report_topks,
        report_pcts,
        title="Benchmark Sweep Report",
        handoff=handoff,
    )
    print(f"\nDone → {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

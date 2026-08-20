#!/usr/bin/env python3
"""Adaptive sweep: extremes first, skip expensive configs when already near-ideal."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from analyze_sweep import analyze_sweep_jsonl, write_sweep_report  # noqa: E402
from config import ExperimentConfig  # noqa: E402
from eval_utils import (  # noqa: E402
    attach_sampler,
    build_chat_prompt,
    extract_answer,
    load_gsm8k_samples,
    set_seed,
)
from logging_utils import append_jsonl  # noqa: E402
from model_utils import configure_block_size, default_small_block_size  # noqa: E402
from sweep_configs import (  # noqa: E402
    build_adaptive_sweep,
    config_metrics,
    should_skip_entry,
)

_UPSTREAM = _ROOT.parent / "_fast_dllm_upstream" / "v2"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
import generation_functions as upstream_gen  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _done_keys(jsonl_path: Path) -> set[tuple[str, int]]:
    return {(r["config"]["name"], r["example_id"]) for r in _load_jsonl(jsonl_path)}


@torch.no_grad()
def run_one_example(model, tokenizer, sample: dict, exp_cfg: ExperimentConfig) -> dict:
    prompt = build_chat_prompt(tokenizer, sample["question"])
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    run_log: list = []
    seq_len = torch.tensor([prompt_len], device=model.device)
    small = exp_cfg.small_block_size or default_small_block_size(exp_cfg.block_size)
    common = dict(
        input_ids=input_ids,
        tokenizer=tokenizer,
        block_size=exp_cfg.block_size,
        small_block_size=small,
        max_new_tokens=exp_cfg.max_new_tokens,
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
    pred_num = extract_answer(gen_text)
    state = run_log[-1] if run_log else {}

    return {
        "task": "gsm8k",
        "example_id": sample["idx"],
        "question": sample["question"],
        "gold_num": sample["gold_num"],
        "pred_num": pred_num,
        "correct": pred_num is not None and sample["gold_num"] is not None and pred_num == sample["gold_num"],
        "generated_answer": gen_text,
        "gen_tokens": int(output_ids.shape[0] - prompt_len),
        "config": exp_cfg.to_dict(),
        "blocks": state.get("blocks", []),
        "cost": state.get("cost_summary", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--output-dir", type=str, default="results/sweep_gsm8k")
    parser.add_argument("--no-adaptive", action="store_true", help="Run full grid without skipping")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    skip_log_path = out_dir / "skipped.jsonl"

    if args.analyze_only:
        analysis = analyze_sweep_jsonl(jsonl_path)
        (out_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_sweep_report(analysis, out_dir / "report.md")
        print(f"Report → {out_dir / 'report.md'}")
        return

    entries = build_adaptive_sweep()
    done = _done_keys(jsonl_path)
    set_seed(args.seed)

    print(f"Loading model on {args.device} ...")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        "Efficient-Large-Model/Fast_dLLM_v2_7B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "Efficient-Large-Model/Fast_dLLM_v2_7B", trust_remote_code=True
    )
    configure_block_size(model, entries[0].cfg.block_size)
    samples = load_gsm8k_samples(args.num_examples, args.seed)

    tier_counts = defaultdict(int)
    for e in entries:
        tier_counts[e.tier] += 1
    print(f"Grid: {len(entries)} configs ({dict(tier_counts)}), adaptive={not args.no_adaptive}")

    completed_by_config: dict[str, list[dict]] = defaultdict(list)
    for r in _load_jsonl(jsonl_path):
        completed_by_config[r["config"]["name"]].append(r)

    baseline_accuracy = 1.0
    baseline_recs = completed_by_config.get("baseline_dense_fp16", [])
    if baseline_recs:
        baseline_accuracy = config_metrics(baseline_recs)["accuracy"]

    skipped_configs: set[str] = set()
    skip_messages: list[str] = []
    t0 = time.time()
    ran = 0

    for ei, entry in enumerate(entries):
        exp_cfg = entry.cfg

        if exp_cfg.name in skipped_configs:
            continue

        # Need baseline before adaptive skip
        if not args.no_adaptive and entry.tier != "baseline":
            if not baseline_recs and exp_cfg.name != "baseline_dense_fp16":
                pass  # will run baseline when reached
            elif baseline_recs and should_skip_entry(
                entry,
                baseline_accuracy=baseline_accuracy,
                completed_by_config=completed_by_config,
                skip_log=skip_messages,
            ):
                skipped_configs.add(exp_cfg.name)
                append_jsonl(skip_log_path, {"config": exp_cfg.name, "reason": skip_messages[-1]})
                print(f"SKIP [{ei+1}/{len(entries)}] {exp_cfg.name}  ({entry.tier})")
                continue

        attach_sampler(model, exp_cfg, upstream_gen.Fast_dLLM_QwenForCausalLM.batch_sample)
        pending = [s for s in samples if (exp_cfg.name, s["idx"]) not in done]
        if not pending:
            print(f"DONE [{ei+1}/{len(entries)}] {exp_cfg.name} (cached)")
            continue

        print(f"RUN  [{ei+1}/{len(entries)}] {exp_cfg.name}  tier={entry.tier}  ({len(pending)} ex)")
        for sample in pending:
            t1 = time.time()
            record = run_one_example(model, tokenizer, sample, exp_cfg)
            append_jsonl(jsonl_path, record)
            done.add((exp_cfg.name, sample["idx"]))
            completed_by_config[exp_cfg.name].append(record)
            ran += 1
            dt = time.time() - t1
            cov = config_metrics([record]).get("avg_coverage")
            cov_s = f"{cov:.3f}" if cov is not None else "—"
            print(
                f"  ex{sample['idx']} correct={record['correct']} coverage={cov_s} "
                f"gen_tok={record['gen_tokens']} {dt:.1f}s"
            )

        if exp_cfg.name == "baseline_dense_fp16":
            baseline_recs = completed_by_config[exp_cfg.name]
            baseline_accuracy = config_metrics(baseline_recs)["accuracy"]
            print(f"  → baseline accuracy={baseline_accuracy:.2%}")

        m = config_metrics(completed_by_config[exp_cfg.name])
        if m.get("avg_coverage") is not None:
            print(f"  → acc={m['accuracy']:.2%} avg_coverage={m['avg_coverage']:.3f}")

    elapsed = time.time() - t0
    print(f"\nFinished: {ran} new runs, {len(skipped_configs)} skipped, {elapsed/60:.1f} min")
    if skip_messages:
        print("Skip reasons:")
        for msg in skip_messages:
            print(f"  {msg}")

    analysis = analyze_sweep_jsonl(jsonl_path)
    analysis["skipped_configs"] = sorted(skipped_configs)
    analysis["skip_messages"] = skip_messages
    (out_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_sweep_report(analysis, out_dir / "report.md")
    print(f"Report → {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

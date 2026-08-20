#!/usr/bin/env python3
"""LongBench eval: baseline vs extreme k128/k2v2 vs middle-selector fp16."""

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

from config import ExperimentConfig  # noqa: E402
from eval_utils import attach_sampler, set_seed  # noqa: E402
from logging_utils import append_jsonl  # noqa: E402
from longbench_presets import longbench_presets  # noqa: E402
from longbench_utils import (  # noqa: E402
    DEFAULT_TASKS,
    load_longbench_samples,
    max_new_tokens_for_task,
    prepare_inputs,
    score_prediction,
)
from model_utils import configure_block_size, default_small_block_size  # noqa: E402

_UPSTREAM = _ROOT.parent / "_fast_dllm_upstream" / "v2"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
import generation_functions as upstream_gen  # noqa: E402


def _done_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.add((r["config"]["name"], r["task"], r["example_id"]))
    return out


@torch.no_grad()
def run_one(
    model,
    tokenizer,
    sample: dict,
    exp_cfg: ExperimentConfig,
    *,
    model_max_tokens: int = 32768,
) -> dict:
    task = sample["task"]
    max_gen = max_new_tokens_for_task(task)
    input_ids, prompt_text, prompt_len = prepare_inputs(
        tokenizer, task, sample, model_max_tokens=model_max_tokens
    )
    input_ids = input_ids.to(model.device)

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

    score = score_prediction(task, gen_text, sample["answers"], sample.get("all_classes"))
    covs = [
        layer["attention_mass_captured"]
        for blk in state.get("blocks", [])
        for layer in blk.get("layers", {}).values()
        if "attention_mass_captured" in layer
    ]

    return {
        "task": task,
        "example_id": sample["idx"],
        "longbench_id": sample.get("_id"),
        "prompt_tokens": prompt_len,
        "context_length_field": sample.get("length"),
        "answers": sample["answers"],
        "score": score,
        "generated_answer": gen_text,
        "gen_tokens": int(output_ids.shape[0] - prompt_len),
        "avg_topk_coverage": sum(covs) / len(covs) if covs else None,
        "config": exp_cfg.to_dict(),
        "cost": state.get("cost_summary", {}),
        "num_blocks": len(state.get("blocks", [])),
    }


def summarize_records(records: list[dict]) -> dict:
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_key[(r["config"]["name"], r["task"])].append(r)
    rows = []
    for (cfg, task), rs in sorted(by_key.items()):
        scores = [r["score"] for r in rs]
        covs = [r["avg_topk_coverage"] for r in rs if r.get("avg_topk_coverage") is not None]
        rows.append(
            {
                "config": cfg,
                "task": task,
                "n": len(rs),
                "score_pct": round(100 * sum(scores) / len(scores), 2),
                "avg_coverage": round(sum(covs) / len(covs), 4) if covs else None,
                "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in rs) / len(rs)),
                "avg_gen_tokens": round(sum(r["gen_tokens"] for r in rs) / len(rs)),
            }
        )
    return {"summaries": rows}


def write_report(analysis: dict, path: Path) -> None:
    lines = ["# LongBench Sparse-KV Report", ""]
    lines.append("| Config | Task | N | Score% | Coverage | Prompt tok | Gen tok |")
    lines.append("|--------|------|---|--------|----------|------------|---------|")
    for row in analysis["summaries"]:
        cov = row["avg_coverage"]
        cov_s = f"{cov:.3f}" if cov is not None else "—"
        lines.append(
            f"| {row['config']} | {row['task']} | {row['n']} | {row['score_pct']} | "
            f"{cov_s} | {row['avg_prompt_tokens']} | {row['avg_gen_tokens']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--configs", nargs="+", default=["baseline", "extreme_k128_k2v2", "middle_fp16"])
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--output-dir", type=str, default="results/longbench")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"

    if args.analyze_only:
        records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        analysis = summarize_records(records)
        (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        write_report(analysis, out_dir / "report.md")
        print(f"Report → {out_dir / 'report.md'}")
        return

    presets = longbench_presets()
    configs = [presets[c] for c in args.configs if c in presets]
    done = _done_keys(jsonl_path)
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Loading model on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        "Efficient-Large-Model/Fast_dLLM_v2_7B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "Efficient-Large-Model/Fast_dLLM_v2_7B", trust_remote_code=True
    )
    configure_block_size(model, configs[0].block_size)
    model_max = getattr(model.config, "max_position_embeddings", 32768)

    for task in args.tasks:
        samples = load_longbench_samples(task, num_examples=args.num_examples, seed=args.seed)
        print(f"\n=== Task {task}: {len(samples)} examples ===")
        for exp_cfg in configs:
            attach_sampler(model, exp_cfg, upstream_gen.Fast_dLLM_QwenForCausalLM.batch_sample)
            pending = [s for s in samples if (exp_cfg.name, task, s["idx"]) not in done]
            if not pending:
                print(f"  {exp_cfg.name}: cached")
                continue
            print(f"  RUN {exp_cfg.name} ({len(pending)} ex)")
            for sample in pending:
                t0 = time.time()
                record = run_one(model, tokenizer, sample, exp_cfg, model_max_tokens=model_max)
                append_jsonl(jsonl_path, record)
                done.add((exp_cfg.name, task, sample["idx"]))
                dt = time.time() - t0
                print(
                    f"    ex{sample['idx']} score={record['score']:.3f} "
                    f"cov={record['avg_topk_coverage']} prompt={record['prompt_tokens']} "
                    f"gen={record['gen_tokens']} {dt:.1f}s"
                )

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    analysis = summarize_records(records)
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    write_report(analysis, out_dir / "report.md")
    print(f"\nDone → {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

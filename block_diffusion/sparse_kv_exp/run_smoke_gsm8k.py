#!/usr/bin/env python3
"""Smoke test: sparse old-cache + low-bit KV/Q attention on GSM8K (2–5 examples)."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from lm_eval import tasks
from transformers import AutoModelForCausalLM, AutoTokenizer

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from config import ExperimentConfig, load_config, preset_config  # noqa: E402
from generation import batch_sample_sparse_kv  # noqa: E402
from logging_utils import append_jsonl, summarize_run, write_markdown_report  # noqa: E402
from model_utils import configure_block_size, default_small_block_size  # noqa: E402

# Upstream original baseline
_UPSTREAM = _ROOT.parent / "_fast_dllm_upstream" / "v2"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
import generation_functions as upstream_gen  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_answer(text: str) -> str | None:
    m = re.search(r"####\s*(-?[\d.,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return nums[-1] if nums else None


def format_gsm8k_question(question: str) -> str:
    ctx = f"Question: {question}\nAnswer:"
    return ctx.replace(
        "Answer:",
        "Please reason step by step, and put your final answer within \\boxed{{}}.",
    )


def build_chat_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": format_gsm8k_question(question)}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load_gsm8k_samples(n: int, seed: int) -> list[dict]:
    task_dict = tasks.get_task_dict(["gsm8k"])
    task = task_dict["gsm8k"]
    task._config.num_fewshot = 0
    task.set_fewshot_seed(seed=seed)
    task.build_all_requests(limit=n, rank=0, world_size=1)
    samples = []
    for i, inst in enumerate(task._instances):
        doc = inst.doc
        samples.append(
            {
                "idx": i,
                "question": doc["question"],
                "gold_answer": task.doc_to_target(doc),
                "gold_num": extract_answer(task.doc_to_target(doc)),
            }
        )
    return samples


def attach_sampler(model, exp_cfg: ExperimentConfig):
    if exp_cfg.baseline == "original":
        model.mdm_sample = types.MethodType(upstream_gen.Fast_dLLM_QwenForCausalLM.batch_sample, model)
    else:
        model.mdm_sample = types.MethodType(batch_sample_sparse_kv, model)


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    exp_cfg: ExperimentConfig,
    experiment_log: list | None = None,
) -> torch.Tensor:
    seq_len = torch.tensor([prompt_ids.shape[1]], device=prompt_ids.device)
    min_len = prompt_ids.shape[1]
    small = exp_cfg.small_block_size or default_small_block_size(exp_cfg.block_size)
    common = dict(
        input_ids=prompt_ids,
        tokenizer=tokenizer,
        block_size=exp_cfg.block_size,
        small_block_size=small,
        max_new_tokens=exp_cfg.max_new_tokens,
        min_len=min_len,
        seq_len=seq_len,
        threshold=exp_cfg.threshold,
    )
    if exp_cfg.baseline == "original":
        out = model.mdm_sample(**common)
    else:
        out = model.mdm_sample(**common, exp_config=exp_cfg, experiment_log=experiment_log)
    return out[0]


def run_config(
    model,
    tokenizer,
    samples: list[dict],
    exp_cfg: ExperimentConfig,
    *,
    jsonl_path: Path,
) -> dict:
    attach_sampler(model, exp_cfg)
    examples = []
    t0 = time.time()

    for sample in samples:
        prompt = build_chat_prompt(tokenizer, sample["question"])
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        prompt_len = input_ids.shape[1]

        run_log: list = []
        output_ids = generate_one(model, tokenizer, input_ids, exp_cfg, experiment_log=run_log)
        gen_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
        pred_num = extract_answer(gen_text)
        correct = pred_num is not None and sample["gold_num"] is not None and pred_num == sample["gold_num"]

        cost_summary = run_log[-1].get("cost_summary", {}) if run_log else {}
        ex_record = {
            "task": "gsm8k",
            "example_id": sample["idx"],
            "question": sample["question"],
            "gold_num": sample["gold_num"],
            "pred_num": pred_num,
            "correct": correct,
            "generated_answer": gen_text,
            "config": exp_cfg.to_dict(),
            "blocks": run_log[-1]["blocks"] if run_log else [],
            "cost": cost_summary,
        }

        append_jsonl(jsonl_path, ex_record)
        examples.append(ex_record)

    elapsed = time.time() - t0
    # Aggregate cost across examples (mean ratios)
    costs = [e["cost"] for e in examples if e.get("cost")]
    merged_cost = costs[0] if len(costs) == 1 else {"per_example": costs}
    summary = summarize_run(
        config_name=exp_cfg.name,
        config=exp_cfg.to_dict(),
        examples=examples,
        cost_aggregate=merged_cost if isinstance(merged_cost, dict) else {},
    )
    summary["wall_clock_sec"] = elapsed
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse KV smoke test on GSM8K")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--preset", type=str, default=None, choices=["A", "B", "C", "D"])
    parser.add_argument("--all-presets", action="store_true", help="Run A,B,C,D sequentially")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:5")
    args = parser.parse_args()

    if args.all_presets:
        configs = [preset_config(p) for p in ("A", "B", "C", "D")]
    elif args.preset:
        configs = [preset_config(args.preset)]
    elif args.config:
        configs = [load_config(args.config)]
    else:
        configs = [preset_config(p) for p in ("A", "B", "C", "D")]

    base_cfg = configs[0]
    if args.num_examples is not None:
        for c in configs:
            c.num_examples = args.num_examples
    out_dir = Path(args.output_dir or base_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(base_cfg.seed)
    print(f"Loading model {base_cfg.model_path} ...")
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        base_cfg.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_cfg.model_path, trust_remote_code=True)
    configure_block_size(model, base_cfg.block_size)

    samples = load_gsm8k_samples(base_cfg.num_examples, base_cfg.seed)
    jsonl_path = out_dir / "results.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    run_summaries = []
    for exp_cfg in configs:
        print(f"\n=== Running config {exp_cfg.name} (baseline={exp_cfg.baseline}) ===")
        summary = run_config(model, tokenizer, samples, exp_cfg, jsonl_path=jsonl_path)
        run_summaries.append(summary)
        print(
            f"  accuracy={summary['accuracy']:.2%} "
            f"avg_topk_coverage={summary.get('avg_topk_coverage')} "
            f"wall={summary['wall_clock_sec']:.1f}s"
        )

    md_path = out_dir / "report.md"
    aggregate = {
        "num_configs": len(run_summaries),
        "mean_accuracy": sum(r["accuracy"] for r in run_summaries) / len(run_summaries),
    }
    write_markdown_report(md_path, title="Sparse KV Smoke Test", runs=run_summaries, aggregate=aggregate)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(run_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {jsonl_path}, {md_path}, {summary_path}")


if __name__ == "__main__":
    main()

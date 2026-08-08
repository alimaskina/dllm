#!/usr/bin/env python3
"""Sweep block size (bd_size) on GSM8K with shared model load."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_volatility import aggregate, format_markdown
from model_utils import configure_block_size, default_small_block_size
from run_volatility import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_THRESHOLD,
    MODEL_PATH,
    extract_answer,
    load_gsm8k_samples,
    set_seed,
)
from generation_traced import mdm_sample_traced, trace_to_jsonable


def run_bd_size(
    *,
    model,
    tokenizer,
    samples: list[dict],
    bd_size: int,
    small_block_size: int | None,
    max_new_tokens: int,
    threshold: float,
    device: str,
    out_dir: Path,
    seed: int,
    model_path: str,
    equal_to_block: bool = False,
) -> dict:
    sbs = small_block_size if small_block_size is not None else default_small_block_size(
        bd_size, equal_to_block=equal_to_block
    )
    configure_block_size(model, bd_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_path": model_path,
        "task": "gsm8k",
        "num_fewshot": 0,
        "apply_chat_template": True,
        "max_new_tokens": max_new_tokens,
        "bd_size": bd_size,
        "small_block_size": sbs,
        "small_block_equals_block": equal_to_block,
        "threshold": threshold,
        "seed": seed,
        "n_samples": len(samples),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    traces_path = out_dir / "traces.jsonl"
    t0 = time.time()
    records = []

    with traces_path.open("w") as fout:
        for sample in samples:
            from run_volatility import build_chat_prompt

            prompt = build_chat_prompt(tokenizer, sample["question"])
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

            _, trace = mdm_sample_traced(
                model,
                input_ids,
                tokenizer=tokenizer,
                block_size=bd_size,
                max_new_tokens=max_new_tokens,
                small_block_size=sbs,
                threshold=threshold,
            )

            gen_text = tokenizer.decode(trace.output_ids, skip_special_tokens=True)
            pred_num = extract_answer(gen_text)
            record = {
                "idx": sample["idx"],
                "question": sample["question"],
                "gold_num": sample["gold_num"],
                "pred_num": pred_num,
                "correct": pred_num == sample["gold_num"],
                "prompt_len": trace.prompt_len,
                "n_gen_tokens": len(trace.output_ids),
                "generation": gen_text,
                "trace": trace_to_jsonable(trace, tokenizer),
            }
            records.append(record)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                f"[bd={bd_size} {sample['idx']+1}/{len(samples)}] "
                f"blocks={len(trace.blocks)} correct={record['correct']}"
            )

    elapsed = time.time() - t0
    meta["elapsed_sec"] = elapsed
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    summary = aggregate(records)
    summary["bd_size"] = bd_size
    summary["elapsed_sec"] = elapsed
    (out_dir / "volatility_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "volatility_report.md").write_text(format_markdown(summary, meta))
    print(f"bd={bd_size}: acc={summary['accuracy']:.1%} elapsed={elapsed:.0f}s → {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--bd-sizes", default="16,32,64", help="Comma-separated block sizes")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out-root", default="checkpoints/sweep_bd")
    parser.add_argument(
        "--small-block-equals-block",
        action="store_true",
        help="Set small_block_size = bd_size (no sub-blocks)",
    )
    args = parser.parse_args()

    bd_sizes = [int(x.strip()) for x in args.bd_sizes.split(",") if x.strip()]
    set_seed(args.seed)

    print(f"Loading model {args.model_path} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    samples = load_gsm8k_samples(args.n, args.seed)
    print(
        f"Sweep bd_sizes={bd_sizes} n={len(samples)} "
        f"small_block={'=bd_size' if args.small_block_equals_block else 'bd_size//4'}"
    )

    out_root = Path(args.out_root)
    sweep_summaries = []
    sweep_t0 = time.time()

    for bd_size in bd_sizes:
        tag = "sbeq" if args.small_block_equals_block else "sb4"
        out_dir = out_root / f"n{args.n}_bd{bd_size}_{tag}"
        summary = run_bd_size(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            bd_size=bd_size,
            small_block_size=bd_size if args.small_block_equals_block else None,
            max_new_tokens=args.max_new_tokens,
            threshold=args.threshold,
            device=args.device,
            out_dir=out_dir,
            seed=args.seed,
            model_path=args.model_path,
            equal_to_block=args.small_block_equals_block,
        )
        sweep_summaries.append(summary)
        torch.cuda.empty_cache()

    sweep_meta = {
        "n_samples": args.n,
        "bd_sizes": bd_sizes,
        "small_block_equals_block": args.small_block_equals_block,
        "seed": args.seed,
        "total_elapsed_sec": time.time() - sweep_t0,
        "runs": sweep_summaries,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sweep_summary.json").write_text(json.dumps(sweep_meta, indent=2))
    print(f"Sweep done → {out_root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()

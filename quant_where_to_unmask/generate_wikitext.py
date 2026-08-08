#!/usr/bin/env python3
"""Generate LLaDA continuations from WikiText prompts with step traces."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from lm_eval.api.instance import Instance
from transformers import AutoTokenizer

from eval_llada import LLaDAEvalHarness, set_seed


def is_usable_line(text: str, min_chars: int) -> bool:
    s = text.strip()
    if len(s) < min_chars:
        return False
    if s.startswith("="):
        return False
    if not any(ch.isalpha() for ch in s):
        return False
    return True


def build_prompts(
    *,
    dataset: str,
    split: str,
    n: int,
    prompt_tokens: int,
    min_chars: int,
    seed: int,
    model_path: str,
) -> list[dict]:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ds = load_dataset("wikitext", dataset, split=split)
    candidates = [row["text"] for row in ds if is_usable_line(row["text"], min_chars)]

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if len(candidates) < n:
        raise ValueError(f"Only {len(candidates)} usable WikiText lines, need {n}")

    prompts: list[dict] = []
    seen: set[str] = set()
    for text in candidates:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < prompt_tokens + 32:
            continue
        prefix_ids = ids[:prompt_tokens]
        prompt = tok.decode(prefix_ids)
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(
            {
                "doc_id": len(prompts),
                "source_text": text,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
            }
        )
        if len(prompts) >= n:
            break

    if len(prompts) < n:
        raise ValueError(f"Built only {len(prompts)} unique prompts, need {n}")
    return prompts


def build_instances(prompts: list[dict], stop: list[str]) -> list[Instance]:
    instances: list[Instance] = []
    for row in prompts:
        instances.append(
            Instance(
                "generate_until",
                {
                    "source_text": row["source_text"],
                    "prompt_tokens": row["prompt_tokens"],
                },
                (row["prompt"], {"until": stop}),
                row["doc_id"],
            )
        )
    return instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=48)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant", default="fp16", choices=["fp16", "bf16", "int4", "int8"])
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--meta-out", default="")
    parser.add_argument("--stop", default="\\n\\n")
    args = parser.parse_args()

    set_seed(args.seed)
    stop = [s.encode("utf-8").decode("unicode_escape") for s in args.stop.split(",") if s]

    ckpt = Path(args.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    meta_out = Path(args.meta_out) if args.meta_out else ckpt / "prompts.json"

    print(f"Loading WikiText prompts: dataset={args.dataset} split={args.split} n={args.n}")
    prompts = build_prompts(
        dataset=args.dataset,
        split=args.split,
        n=args.n,
        prompt_tokens=args.prompt_tokens,
        min_chars=args.min_chars,
        seed=args.seed,
        model_path=args.model,
    )
    meta_out.write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved prompts → {meta_out}")

    instances = build_instances(prompts, stop)
    print(f"Generating {len(instances)} samples with traces → {ckpt}")

    t0 = time.time()
    model = LLaDAEvalHarness(
        model_path=args.model,
        quant=args.quant,
        gen_length=args.gen_length,
        steps=args.steps,
        block_length=args.block_length,
        checkpoint_dir=str(ckpt),
        gen_batch_size=args.gen_batch_size,
        is_check_greedy=False,
    )
    outputs = model.generate_until(instances)
    elapsed = time.time() - t0

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "n": len(prompts),
        "prompt_tokens": args.prompt_tokens,
        "quant": args.quant,
        "gen_length": args.gen_length,
        "steps": args.steps,
        "checkpoint_dir": str(ckpt),
        "elapsed_sec": round(elapsed, 1),
        "avg_chars": round(sum(len(x) for x in outputs) / len(outputs), 1),
    }
    (ckpt / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done in {elapsed:.0f}s, avg generation len={summary['avg_chars']} chars")
    print(f"Summary → {ckpt / 'summary.json'}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

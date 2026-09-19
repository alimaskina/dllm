#!/usr/bin/env python3
"""MATH500 top-K KV prune experiment: step-0 importance → pruned attention."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
import types
from pathlib import Path

import numpy as np
import torch
from lm_eval import tasks
from transformers import AutoModelForCausalLM, AutoTokenizer

import generation_kv_topk_prune
from kv_cache_quant import DEFAULT_KIVI_GROUP_SIZE
from model_utils import configure_block_size, default_small_block_size

MODEL_PATH = "Efficient-Large-Model/Fast_dLLM_v2_7B"

TOPK_VARIANTS = [
    {"key": "top128_fp", "topk": 128, "bits": 0, "label": "top-128 fp (no quant)"},
    {"key": "top256_kivi8", "topk": 256, "bits": 8, "label": "top-256 KIVI8"},
    {"key": "top512_kivi4", "topk": 512, "bits": 4, "label": "top-512 KIVI4"},
    {"key": "top1024_kivi2", "topk": 1024, "bits": 2, "label": "top-1024 KIVI2"},
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def normalize_math(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    return s.lower()


def format_question(question: str) -> str:
    return question.replace(
        "Answer:",
        "Please reason step by step, and put your final answer within \\boxed{}.",
    )


def build_chat_prompt(tokenizer, question: str) -> str:
    user_text = format_question(question) if "Answer:" in question else (
        f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load_math500_samples(n: int, seed: int) -> list[dict]:
    td = tasks.get_task_dict(["hendrycks_math500"])
    task = td["hendrycks_math500"]
    task._config.num_fewshot = 0
    task.set_fewshot_seed(seed=seed)
    task.build_all_requests(limit=n, rank=0, world_size=1)
    samples = []
    for i, inst in enumerate(task._instances):
        doc = inst.doc
        samples.append(
            {
                "idx": i,
                "question": doc["problem"],
                "gold": task.doc_to_target(doc),
            }
        )
    return samples


def grade_math(gold: str, pred: str | None) -> bool:
    if pred is None:
        return False
    return normalize_math(gold) == normalize_math(pred)


def extract_pred(text: str) -> str | None:
    boxed = extract_boxed(text)
    return boxed if boxed is not None else None


def generate_one(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    threshold: float,
    topk: int,
    topk_bits: int,
    kivi_group_size: int,
    prune_log: list | None,
) -> torch.Tensor:
    seq_len = torch.tensor([prompt_ids.shape[1]], device=prompt_ids.device)
    min_len = prompt_ids.shape[1]
    out = generation_kv_topk_prune.batch_sample(
        model,
        prompt_ids,
        tokenizer=tokenizer,
        block_size=block_size,
        small_block_size=small_block_size,
        max_new_tokens=max_new_tokens,
        min_len=min_len,
        seq_len=seq_len,
        threshold=threshold,
        topk=topk,
        topk_bits=topk_bits,
        kivi_group_size=kivi_group_size,
        prune_log=prune_log,
    )
    return out[0]


def run_variant(
    model,
    tokenizer,
    samples: list[dict],
    variant: dict,
    *,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    threshold: float,
    kivi_group_size: int,
) -> dict:
    results = []
    correct = 0
    prune_rows: list[dict] = []
    t0 = time.time()

    for sample in samples:
        prompt = build_chat_prompt(tokenizer, sample["question"])
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        prompt_len = input_ids.shape[1]
        sample_prune_log: list[dict] = []

        output_ids = generate_one(
            model,
            tokenizer,
            input_ids,
            block_size=block_size,
            small_block_size=small_block_size,
            max_new_tokens=max_new_tokens,
            threshold=threshold,
            topk=variant["topk"],
            topk_bits=variant["bits"],
            kivi_group_size=kivi_group_size,
            prune_log=sample_prune_log,
        )
        for row in sample_prune_log:
            row["sample_idx"] = sample["idx"]
            prune_rows.append(row)

        gen_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
        pred = extract_pred(gen_text)
        is_correct = grade_math(sample["gold"], pred)
        correct += int(is_correct)
        results.append(
            {
                "idx": sample["idx"],
                "gold": sample["gold"],
                "pred": pred,
                "correct": is_correct,
                "generation": gen_text,
            }
        )
        print(
            f"[{variant['key']}] [{sample['idx'] + 1}/{len(samples)}] "
            f"correct={is_correct} pred={pred!r}"
        )

    elapsed = time.time() - t0
    kept = [r["topk_kept"] for r in prune_rows]
    cached = [r["num_cached"] for r in prune_rows]
    stats = {
        "median_topk_kept": statistics.median(kept) if kept else 0,
        "median_num_cached": statistics.median(cached) if cached else 0,
        "n_prune_rows": len(prune_rows),
    }
    return {
        "variant": variant,
        "n_samples": len(samples),
        "correct": correct,
        "accuracy": correct / len(samples) if samples else 0.0,
        "elapsed_sec": elapsed,
        "prune_stats": stats,
        "results": results,
        "prune_log": prune_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--bd-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant keys or 'all'",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    sbs = args.small_block_size or default_small_block_size(args.bd_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.variants == "all":
        variants = TOPK_VARIANTS
    else:
        keys = {k.strip() for k in args.variants.split(",")}
        variants = [v for v in TOPK_VARIANTS if v["key"] in keys]

    print(f"Loading model on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    configure_block_size(model, args.bd_size)
    model.mdm_sample = types.MethodType(generation_kv_topk_prune.batch_sample, model)

    samples = load_math500_samples(args.n, args.seed)
    print(f"Loaded {len(samples)} MATH500 samples")

    meta = {
        "model_path": MODEL_PATH,
        "task": "hendrycks_math500",
        "n_samples": len(samples),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "bd_size": args.bd_size,
        "small_block_size": sbs,
        "threshold": args.threshold,
        "experiment": "topk_kv_prune",
        "variants": variants,
        "policy": (
            "step-0 capture attention → top-K cached tokens by layer-mean importance; "
            "remaining block steps attend only to those tokens (eval_mask patch); "
            "optional KIVI quant on kept tokens"
        ),
    }
    summary: dict = {"meta": meta}

    common = dict(
        block_size=args.bd_size,
        small_block_size=sbs,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        kivi_group_size=DEFAULT_KIVI_GROUP_SIZE,
    )

    for variant in variants:
        print(f"\n=== {variant['label']} ===")
        set_seed(args.seed)
        summary[variant["key"]] = run_variant(
            model, tokenizer, samples, variant, **common
        )
        acc = summary[variant["key"]]["accuracy"]
        st = summary[variant["key"]]["prune_stats"]
        print(
            f"{variant['key']}: accuracy={acc:.1%} "
            f"median_kept={st['median_topk_kept']:.0f} "
            f"median_cached={st['median_num_cached']:.0f}"
        )

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\nSaved → {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

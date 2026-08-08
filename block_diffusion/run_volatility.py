#!/usr/bin/env python3
"""Run Fast-dLLM v2 on GSM8K with block-level token volatility tracing."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from lm_eval import tasks
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_traced import mdm_sample_traced, trace_to_jsonable
from model_utils import configure_block_size, default_small_block_size

MODEL_PATH = "Efficient-Large-Model/Fast_dLLM_v2_7B"

# Official Fast-dLLM v2 GSM8K settings (v2/eval_script.sh)
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_BD_SIZE = 32
DEFAULT_SMALL_BLOCK_SIZE = 8
DEFAULT_THRESHOLD = 1.0
DEFAULT_NUM_FEWSHOT = 0


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
    """Same string transform as Fast-dLLM eval.py generate_until."""
    ctx = f"Question: {question}\nAnswer:"
    return ctx.replace(
        "Answer:",
        "Please reason step by step, and put your final answer within \\boxed{{}}.",
    )


def build_chat_prompt(tokenizer, question: str) -> str:
    user_text = format_gsm8k_question(question)
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def load_gsm8k_samples(n: int, seed: int) -> list[dict]:
    task_dict = tasks.get_task_dict(["gsm8k"])
    task = task_dict["gsm8k"]
    task._config.num_fewshot = DEFAULT_NUM_FEWSHOT
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32, help="Number of GSM8K test samples")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--bd-size", type=int, default=DEFAULT_BD_SIZE)
    parser.add_argument(
        "--small-block-size",
        type=int,
        default=None,
        help="Override small_block_size explicitly",
    )
    parser.add_argument(
        "--small-block-equals-block",
        action="store_true",
        help="Set small_block_size = bd_size (no sub-blocks)",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out-dir", default="checkpoints/gsm8k_volatility_n32")
    parser.add_argument("--use-block-cache", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    sbs = args.small_block_size or default_small_block_size(
        args.bd_size, equal_to_block=args.small_block_equals_block
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Config: n={args.n} bd_size={args.bd_size} "
        f"small_block_size={sbs} threshold={args.threshold}"
    )
    print(f"Loading model {args.model_path} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    configure_block_size(model, args.bd_size)

    samples = load_gsm8k_samples(args.n, args.seed)
    print(f"Loaded {len(samples)} GSM8K samples (num_fewshot={DEFAULT_NUM_FEWSHOT})")

    meta = {
        "model_path": args.model_path,
        "task": "gsm8k",
        "num_fewshot": DEFAULT_NUM_FEWSHOT,
        "apply_chat_template": True,
        "max_new_tokens": args.max_new_tokens,
        "bd_size": args.bd_size,
        "small_block_size": sbs,
        "small_block_equals_block": args.small_block_equals_block,
        "threshold": args.threshold,
        "seed": args.seed,
        "n_samples": len(samples),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    traces_path = out_dir / "traces.jsonl"
    t0 = time.time()

    with traces_path.open("w") as fout:
        for sample in samples:
            prompt = build_chat_prompt(tokenizer, sample["question"])
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)

            _, trace = mdm_sample_traced(
                model,
                input_ids,
                tokenizer=tokenizer,
                block_size=args.bd_size,
                max_new_tokens=args.max_new_tokens,
                small_block_size=sbs,
                threshold=args.threshold,
                use_block_cache=args.use_block_cache,
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
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                f"[{sample['idx']+1}/{len(samples)}] "
                f"blocks={len(trace.blocks)} "
                f"correct={record['correct']} "
                f"pred={pred_num} gold={sample['gold_num']}"
            )

    elapsed = time.time() - t0
    meta["elapsed_sec"] = elapsed
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done in {elapsed:.1f}s → {traces_path}")


if __name__ == "__main__":
    main()

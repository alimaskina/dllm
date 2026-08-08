#!/usr/bin/env python3
"""Run Fast-dLLM v2 generations with cross-block attention capture for KV proxy test."""

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

from generation_attn_traced import attn_trace_to_jsonable, mdm_sample_attn_traced
from generation_traced import trace_to_jsonable
from model_utils import configure_block_size, default_small_block_size

MODEL_PATH = "Efficient-Large-Model/Fast_dLLM_v2_7B"
DEFAULT_MAX_NEW_TOKENS = 512
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


def parse_layers(spec: str | None) -> set[int] | None:
    if spec is None or spec.lower() in ("all", "*"):
        return None
    return {int(x.strip()) for x in spec.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--bd-size", type=int, default=DEFAULT_BD_SIZE)
    parser.add_argument("--small-block-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--layers", default="all", help="Comma-separated layer ids or 'all'")
    parser.add_argument("--out-dir", default="checkpoints/kv_proxy_smoke")
    args = parser.parse_args()

    set_seed(args.seed)
    sbs = args.small_block_size or default_small_block_size(args.bd_size)
    attn_layers = parse_layers(args.layers)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Config: n={args.n} bd_size={args.bd_size} sbs={sbs} "
        f"max_new_tokens={args.max_new_tokens} layers={args.layers}"
    )
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
    meta = {
        "model_path": args.model_path,
        "task": "gsm8k",
        "max_new_tokens": args.max_new_tokens,
        "bd_size": args.bd_size,
        "small_block_size": sbs,
        "threshold": args.threshold,
        "attn_layers": args.layers,
        "seed": args.seed,
        "n_samples": len(samples),
        "experiment": "kv_proxy_one_shot",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    traces_path = out_dir / "attn_traces.jsonl"
    t0 = time.time()

    with traces_path.open("w") as fout:
        for sample in samples:
            prompt = build_chat_prompt(tokenizer, sample["question"])
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)

            _, attn_trace = mdm_sample_attn_traced(
                model,
                input_ids,
                tokenizer=tokenizer,
                block_size=args.bd_size,
                max_new_tokens=args.max_new_tokens,
                small_block_size=sbs,
                threshold=args.threshold,
                attn_layers=attn_layers,
            )

            gen_trace = attn_trace.generation_trace
            gen_text = tokenizer.decode(gen_trace.output_ids, skip_special_tokens=True)
            pred_num = extract_answer(gen_text)

            record = {
                "idx": sample["idx"],
                "question": sample["question"],
                "gold_num": sample["gold_num"],
                "pred_num": pred_num,
                "correct": pred_num == sample["gold_num"],
                "prompt_len": attn_trace.prompt_len,
                "n_gen_tokens": len(gen_trace.output_ids),
                "n_gen_blocks": len(gen_trace.blocks),
                "generation": gen_text,
                "attn": attn_trace_to_jsonable(attn_trace),
                "trace": trace_to_jsonable(gen_trace, tokenizer),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(
                f"[{sample['idx']+1}/{len(samples)}] "
                f"blocks={len(gen_trace.blocks)} "
                f"attn_records={len(attn_trace.records)} "
                f"correct={record['correct']}"
            )

    meta["elapsed_sec"] = time.time() - t0
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done → {traces_path}")


if __name__ == "__main__":
    main()

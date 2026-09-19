#!/usr/bin/env python3
"""GSM8K accuracy eval for Fast-dLLM v2 with optional int4/int8 KV cache quantization."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import types
from pathlib import Path

import numpy as np
import torch
from lm_eval import tasks
from transformers import AutoModelForCausalLM, AutoTokenizer

import generation_kv_quant
from model_utils import configure_block_size, default_small_block_size

MODEL_PATH = "Efficient-Large-Model/Fast_dLLM_v2_7B"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_BD_SIZE = 32
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
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


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


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    threshold: float,
    quantize_kv_cache: bool = False,
    kv_quant_bits: int = 0,
    keys_pre_rope: bool = False,
    kv_quant_scheme: str = "block",
) -> torch.Tensor:
    seq_len = torch.tensor([prompt_ids.shape[1]], device=prompt_ids.device)
    min_len = prompt_ids.shape[1]
    out = model.mdm_sample(
        prompt_ids,
        tokenizer=tokenizer,
        block_size=block_size,
        small_block_size=small_block_size,
        max_new_tokens=max_new_tokens,
        min_len=min_len,
        seq_len=seq_len,
        threshold=threshold,
        quantize_kv_cache=quantize_kv_cache,
        kv_quant_bits=kv_quant_bits,
        keys_pre_rope=keys_pre_rope,
        kv_quant_scheme=kv_quant_scheme,
    )
    return out[0]


def run_eval(
    model,
    tokenizer,
    samples: list[dict],
    *,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    threshold: float,
    quantize_kv_cache: bool = False,
    kv_quant_bits: int = 0,
    keys_pre_rope: bool = False,
    kv_quant_scheme: str = "block",
) -> dict:
    results = []
    correct = 0
    t0 = time.time()

    for sample in samples:
        prompt = build_chat_prompt(tokenizer, sample["question"])
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        prompt_len = input_ids.shape[1]

        output_ids = generate_one(
            model,
            tokenizer,
            input_ids,
            block_size=block_size,
            small_block_size=small_block_size,
            max_new_tokens=max_new_tokens,
            threshold=threshold,
            quantize_kv_cache=quantize_kv_cache,
            kv_quant_bits=kv_quant_bits,
            keys_pre_rope=keys_pre_rope,
            kv_quant_scheme=kv_quant_scheme,
        )
        gen_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
        pred_num = extract_answer(gen_text)
        is_correct = pred_num == sample["gold_num"]
        correct += int(is_correct)

        record = {
            "idx": sample["idx"],
            "gold_num": sample["gold_num"],
            "pred_num": pred_num,
            "correct": is_correct,
            "generation": gen_text,
        }
        results.append(record)
        print(
            f"[{sample['idx'] + 1}/{len(samples)}] "
            f"correct={is_correct} pred={pred_num} gold={sample['gold_num']}"
        )

    elapsed = time.time() - t0
    acc = correct / len(samples) if samples else 0.0
    return {
        "n_samples": len(samples),
        "correct": correct,
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--bd-size", type=int, default=DEFAULT_BD_SIZE)
    parser.add_argument("--small-block-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--keys-pre-rope", action="store_true", help="Quantize keys in pre-RoPE space")
    parser.add_argument(
        "--mode",
        choices=["baseline", "kv_int8", "kv_int4", "both", "both4", "both_all", "both4_micro"],
        default="both4_micro",
        help="baseline | kv_int8 | kv_int4 | both | both4 | both_all | both4_micro (baseline+int4 micro16)",
    )
    parser.add_argument("--kv-scheme", choices=["block", "micro16"], default="block")
    parser.add_argument("--kv-bits", type=int, default=None, help="Override quant bits for kv_int* modes")
    parser.add_argument("--out-dir", default="checkpoints/gsm8k_kv_quant_n50")
    args = parser.parse_args()

    set_seed(args.seed)
    sbs = args.small_block_size or default_small_block_size(args.bd_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    model.mdm_sample = types.MethodType(generation_kv_quant.batch_sample, model)

    samples = load_gsm8k_samples(args.n, args.seed)
    print(f"Loaded {len(samples)} GSM8K samples")

    meta = {
        "model_path": args.model_path,
        "task": "gsm8k",
        "num_fewshot": DEFAULT_NUM_FEWSHOT,
        "apply_chat_template": True,
        "max_new_tokens": args.max_new_tokens,
        "bd_size": args.bd_size,
        "small_block_size": sbs,
        "threshold": args.threshold,
        "seed": args.seed,
        "n_samples": len(samples),
        "keys_pre_rope": args.keys_pre_rope,
        "kv_scheme": args.kv_scheme,
        "kv_quant": f"int4_{args.kv_scheme}" if args.kv_scheme == "micro16" else "int4/int8_per_block_scalar",
    }
    if args.mode == "both_all":
        meta["kv_quant"] = "int4/int8_per_block_scalar"

    summary = {"meta": meta}
    kpr = args.keys_pre_rope
    scheme = args.kv_scheme
    rope_tag = " pre-RoPE keys" if kpr else ""
    scheme_tag = f" scheme={scheme}" if scheme != "block" else ""

    if args.mode in ("baseline", "both", "both4", "both_all", "both4_micro"):
        print("\n=== Baseline (full-precision KV cache) ===")
        set_seed(args.seed)
        summary["baseline"] = run_eval(
            model,
            tokenizer,
            samples,
            block_size=args.bd_size,
            small_block_size=sbs,
            max_new_tokens=args.max_new_tokens,
            threshold=args.threshold,
            quantize_kv_cache=False,
            kv_quant_bits=0,
        )
        print(f"Baseline accuracy: {summary['baseline']['accuracy']:.1%}")

    if args.mode in ("kv_int8", "both", "both_all"):
        bits = args.kv_bits or 8
        print(f"\n=== KV int{bits}{rope_tag} ===")
        set_seed(args.seed)
        summary[f"kv_int{bits}"] = run_eval(
            model,
            tokenizer,
            samples,
            block_size=args.bd_size,
            small_block_size=sbs,
            max_new_tokens=args.max_new_tokens,
            threshold=args.threshold,
            kv_quant_bits=bits,
            keys_pre_rope=kpr,
            kv_quant_scheme=scheme,
        )
        print(f"KV int{bits}{scheme_tag} accuracy: {summary[f'kv_int{bits}']['accuracy']:.1%}")

    if args.mode in ("kv_int4", "both4", "both_all"):
        bits = args.kv_bits or 4
        print(f"\n=== KV int{bits}{rope_tag} scheme=block ===")
        set_seed(args.seed)
        summary[f"kv_int{bits}"] = run_eval(
            model,
            tokenizer,
            samples,
            block_size=args.bd_size,
            small_block_size=sbs,
            max_new_tokens=args.max_new_tokens,
            threshold=args.threshold,
            kv_quant_bits=bits,
            keys_pre_rope=kpr,
            kv_quant_scheme="block",
        )
        print(f"KV int{bits} accuracy: {summary[f'kv_int{bits}']['accuracy']:.1%}")

    if args.mode == "both4_micro":
        bits = 4
        print(f"\n=== KV int4 scheme=micro16 ===")
        set_seed(args.seed)
        summary["kv_int4_micro16"] = run_eval(
            model,
            tokenizer,
            samples,
            block_size=args.bd_size,
            small_block_size=sbs,
            max_new_tokens=args.max_new_tokens,
            threshold=args.threshold,
            kv_quant_bits=bits,
            keys_pre_rope=kpr,
            kv_quant_scheme="micro16",
        )
        print(f"KV int4 micro16 accuracy: {summary['kv_int4_micro16']['accuracy']:.1%}")

    if args.mode == "both":
        b = summary["baseline"]["accuracy"]
        q = summary["kv_int8"]["accuracy"]
        summary["delta_int8"] = q - b
        print(f"\nDelta (kv_int8 - baseline): {summary['delta_int8']:+.1%}")
    elif args.mode == "both4_micro":
        b = summary["baseline"]["accuracy"]
        q = summary["kv_int4_micro16"]["accuracy"]
        summary["delta_int4_micro16"] = q - b
        print(f"\nDelta (kv_int4 micro16 - baseline): {summary['delta_int4_micro16']:+.1%}")
    elif args.mode in ("both4", "both_all"):
        b = summary["baseline"]["accuracy"]
        if "kv_int4" in summary:
            q4 = summary["kv_int4"]["accuracy"]
            summary["delta_int4"] = q4 - b
            print(f"\nDelta (kv_int4 - baseline): {summary['delta_int4']:+.1%}")
        if args.mode == "both_all" and "kv_int8" in summary:
            q8 = summary["kv_int8"]["accuracy"]
            summary["delta_int8"] = q8 - b
            print(f"Delta (kv_int8 - baseline): {summary['delta_int8']:+.1%}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

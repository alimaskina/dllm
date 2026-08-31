#!/usr/bin/env python3
"""DreamReasoner-8B sparse-KV benchmark (gsm8k / math500).

Runs three sparse configs on top of Dream's block-diffusion:
- fp16_all_kK   — selector attention on fp16 K, keep-set = top-K by mean(all queries × heads).
- fp16_middle_kK — selector attention on fp16 K, keep-set = top-K by middle query row.
- k4sel_v4_kK  — K & V quantized to 4-bit KIVI (group_size=32) in BOTH selector and exec.

Records JSONL in a uniform schema for regrade.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src/ so we can import the sparse-KV modules.
_BENCH_DIR = Path(__file__).resolve().parent
_ROOT = _BENCH_DIR.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from dream_sparse import block_diffusion_generate_sparse  # noqa: E402
from logging_utils import append_jsonl  # noqa: E402

MODEL_ID = "Dream-org/DreamReasoner-8B"
MAX_NEW_TOKENS = {"gsm8k": 2048, "math500": 1024}


def max_new_tokens_for_task(task: str) -> int:
    return MAX_NEW_TOKENS[task]


def extract_boxed(text):
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


def extract_gsm8k_answer(text):
    m = re.search(r"####\s*(-?[\d.,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    boxed = extract_boxed(text)
    if boxed:
        nums = re.findall(r"-?\d+(?:\.\d+)?", boxed)
        return nums[-1] if nums else boxed.strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return nums[-1] if nums else None


def normalize_math(s):
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    return s.lower()


def format_question(question):
    if "Answer:" in question:
        return question.replace(
            "Answer:",
            "Please reason step by step, and put your final answer within \\boxed{}.",
        )
    return f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def extract_prediction(grader, text):
    if grader == "math":
        boxed = extract_boxed(text)
        return boxed if boxed is not None else text.strip()[:120]
    return extract_gsm8k_answer(text)


def grade_sample(grader, gold, prediction):
    if prediction is None:
        return False
    if grader == "math":
        pg = normalize_math(extract_boxed(prediction) or prediction)
        gg = normalize_math(gold)
        return pg == gg
    return str(prediction) == str(gold)


def load_benchmark_samples(task, *, num_examples):
    if task == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        return [
            {
                "idx": i,
                "task": task,
                "question": ds[int(i)]["question"],
                "gold": extract_gsm8k_answer(ds[int(i)]["answer"]),
                "grader": "gsm8k",
            }
            for i in range(min(num_examples, len(ds)))
        ]
    if task == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            {
                "idx": i,
                "task": task,
                "question": ds[int(i)]["problem"],
                "gold": ds[int(i)]["answer"],
                "grader": "math",
            }
            for i in range(min(num_examples, len(ds)))
        ]
    raise ValueError(task)


def build_dream_prompt(tokenizer, question):
    messages = [{"role": "user", "content": format_question(question)}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
    )


def _done_keys(path, config_name):
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["config"]["name"] == config_name:
            out.add((r["config"]["name"], r["task"], r["example_id"]))
    return out


@torch.no_grad()
def run_one(model, tokenizer, sample, *, block_length, sparse_topk, config_name, selector_mode="middle",
             selector_k_bits=16, selector_v_bits=16, exec_k_bits=16, exec_v_bits=16):
    task = sample["task"]
    max_gen = max_new_tokens_for_task(task)
    prompt = build_dream_prompt(tokenizer, sample["question"])
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = int(input_ids.shape[1])

    seqs, nfe = block_diffusion_generate_sparse(
        model,
        input_ids=input_ids,
        mask_id=model.config.mask_token_id,
        gen_length=max_gen,
        block_length=block_length,
        sparse_topk=sparse_topk,
        selector_mode=selector_mode,
        selector_k_bits=selector_k_bits,
        selector_v_bits=selector_v_bits,
        exec_k_bits=exec_k_bits,
        exec_v_bits=exec_v_bits,
    )
    output_ids = seqs[0]
    gen_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
    pred = extract_prediction(sample["grader"], gen_text)
    correct = grade_sample(sample["grader"], sample["gold"], pred)
    return {
        "task": task,
        "example_id": sample["idx"],
        "question": sample["question"],
        "gold": sample["gold"],
        "prediction": pred,
        "correct": bool(correct),
        "score": float(correct),
        "generated_answer": gen_text,
        "prompt_tokens": prompt_len,
        "gen_tokens": int(output_ids.shape[0] - prompt_len),
        "avg_topk_coverage_fp16_ref": None,
        "config": {
            "name": config_name,
            "model": MODEL_ID,
            "block_length": block_length,
            "sparse_topk": sparse_topk,
            "selector_mode": selector_mode,
            "max_new_tokens": max_gen,
            "dtype": "bfloat16",
            "baseline": "sparse_fp16",
        },
        "num_blocks": None,
        "nfe": nfe,
    }


def summarize(records):
    by_key = defaultdict(list)
    for r in records:
        by_key[(r["config"]["name"], r["task"])].append(r)
    rows = []
    for (cfg, task), rs in sorted(by_key.items()):
        n = len(rs)
        score = 100 * sum(r["score"] for r in rs) / n
        nfes = [r.get("nfe") for r in rs if r.get("nfe")]
        rows.append({
            "config": cfg,
            "task": task,
            "n": n,
            "score_pct": round(score, 2),
            "avg_nfe": (sum(nfes) / len(nfes)) if nfes else None,
        })
    return {"summaries": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("gsm8k", "math500"), required=True)
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--k", type=int, action="append", default=[], help="Fixed top-K budget(s), default 32 and 64")
    p.add_argument("--selector", choices=("middle", "all_mean"), default="middle")
    p.add_argument("--selector-k-bits", type=int, default=16)
    p.add_argument("--selector-v-bits", type=int, default=16)
    p.add_argument("--exec-k-bits", type=int, default=16)
    p.add_argument("--exec-v-bits", type=int, default=16, help="Set K/V bits to 4 for KIVI quant (k4sel_v4).")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    ks = args.k or [32, 64]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "results.jsonl"

    device = torch.device(args.device)
    print(f"Loading {MODEL_ID} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).eval().to(device)

    samples = load_benchmark_samples(args.task, num_examples=args.num_examples)
    print(f"{args.task} n={len(samples)} ks={ks} on {args.device}", flush=True)

    since_analysis = 0
    for k in ks:
        # k4sel_v4: all four precisions == 4 → symmetric quant selector + exec
        is_k4sel = args.selector_k_bits == 4 and args.exec_k_bits == 4 and args.exec_v_bits == 4
        v_suffix = "_k4sel_v4" if is_k4sel else ("_v4" if args.exec_v_bits == 4 else "")
        config_name = f"dream_sparse_{args.selector}_k{k}{v_suffix}"
        done = _done_keys(jsonl_path, config_name)
        pending = [s for s in samples if (config_name, args.task, s["idx"]) not in done]
        print(f"\n=== {config_name}: RUN {len(pending)} ===", flush=True)
        for sample in pending:
            ex_id = sample["idx"]
            print(f"  ex{ex_id} START", flush=True)
            t0 = time.time()
            try:
                rec = run_one(model, tokenizer, sample,
                              block_length=args.block_length, sparse_topk=k,
                              config_name=config_name, selector_mode=args.selector,
                              selector_k_bits=args.selector_k_bits, selector_v_bits=args.selector_v_bits,
                              exec_k_bits=args.exec_k_bits, exec_v_bits=args.exec_v_bits)
            except Exception:
                import traceback
                traceback.print_exc()
                raise
            append_jsonl(jsonl_path, rec)
            done.add((config_name, args.task, ex_id))
            since_analysis += 1
            dt = time.time() - t0
            print(f"  ex{ex_id} score={rec['score']:.3f} nfe={rec.get('nfe')} {dt:.0f}s", flush=True)
            if since_analysis >= 10:
                records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
                (out / "analysis.json").write_text(json.dumps(summarize(records), indent=2), encoding="utf-8")
                since_analysis = 0

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    (out / "analysis.json").write_text(json.dumps(summarize(records), indent=2), encoding="utf-8")
    print(f"\nDone → {out}", flush=True)


if __name__ == "__main__":
    main()

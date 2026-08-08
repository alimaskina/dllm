#!/usr/bin/env python3
"""
Dump GSM8K generations using the official lm-eval task builder.

Uses task.build_all_requests() — same path as eval_llada.py CLI:
  - num_fewshot from gsm8k.yaml (5)
  - doc_to_text / doc_to_target from task config
  - generation_kwargs (stop tokens) via construct_requests()

For fully official artifact trail, prefer run_gsm8k.sh with --log_samples
and then: python show_samples.py results_gsm8k_fp16_smoke.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from lm_eval import tasks

from eval_llada import LLaDAEvalHarness, set_seed


def extract_answer(text: str):
    m = re.search(r"####\s*(-?[\d.,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return nums[-1] if nums else None


def build_official_instances(n: int, fewshot_seed: int = 1234):
    """Build Instance objects exactly as lm-eval evaluator does."""
    task_dict = tasks.get_task_dict(["gsm8k"])
    task = task_dict["gsm8k"]
    task.set_fewshot_seed(seed=fewshot_seed)
    task.build_all_requests(limit=n, rank=0, world_size=1)
    return task, list(task._instances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--fewshot-seed", type=int, default=1234)
    parser.add_argument("--out", default="generations_gsm8k_smoke.md")
    parser.add_argument("--quants", default="fp16,int4")
    args = parser.parse_args()

    set_seed(1234)
    model_path = "GSAI-ML/LLaDA-8B-Base"
    gen_kwargs = dict(
        model_path=model_path,
        gen_length=1024,
        steps=1024,
        block_length=1024,
        is_check_greedy=False,
    )

    print("Building official lm-eval instances...")
    task, instances = build_official_instances(args.n, args.fewshot_seed)
    print(f"  task={task.config.task}  num_fewshot={task.config.num_fewshot}  n={len(instances)}")

    # peek at first prompt to verify 5-shot structure
    first_ctx = instances[0].arguments[0]
    n_fewshot_blocks = first_ctx.count("Question:")
    print(f"  first prompt has {n_fewshot_blocks} 'Question:' blocks (expect 6 = 5 fewshot + 1 target)")

    gens_by_quant = {}
    for quant in args.quants.split(","):
        print(f"\n=== Loading {quant} ===")
        t0 = time.time()
        model = LLaDAEvalHarness(quant=quant.strip(), **gen_kwargs)
        gens_by_quant[quant] = model.generate_until(instances)
        print(f"Done {quant} in {time.time()-t0:.0f}s")
        del model
        torch.cuda.empty_cache()

    lines = [
        "# GSM8K generations (official lm-eval task builder)\n",
        f"**Setup:** gsm8k.yaml, num_fewshot={task.config.num_fewshot}, "
        f"gen/steps/block=1024, fewshot_seed={args.fewshot_seed}, n={args.n}\n",
    ]

    samples = []
    for i, inst in enumerate(instances):
        doc = inst.doc
        gold = task.doc_to_target(doc)
        gold_num = extract_answer(gold)
        short_q = doc["question"]
        prompt = inst.arguments[0]

        row = {
            "idx": i,
            "question": short_q,
            "gold": gold_num,
            "prompt_hash": hash(prompt),
            "n_question_blocks": prompt.count("Question:"),
        }
        lines.append(f"\n## Example {i+1}\n")
        lines.append(f"**Question:** {short_q}\n")
        lines.append(f"**Gold:** {gold_num}\n")
        lines.append(f"**Prompt blocks:** {prompt.count('Question:')} Question: / 5 fewshot + 1 target\n")

        for quant, gens in gens_by_quant.items():
            gen = gens[i]
            pred = extract_answer(gen)
            ok = pred == gold_num
            row[f"{quant}_pred"] = pred
            row[f"{quant}_ok"] = ok
            lines.append(f"**{quant} pred:** {pred} {'✓' if ok else '✗'}\n")
            trunc = gen[:2000] + ("...(truncated)" if len(gen) > 2000 else "")
            lines.append(f"\n### {quant} generation\n```\n{trunc}\n```\n")

        if len(gens_by_quant) == 2:
            q = list(gens_by_quant.keys())
            if gens_by_quant[q[0]][i] != gens_by_quant[q[1]][i]:
                lines.append("\n> ⚠️ FP16 and INT4 outputs differ\n")

        row["prompt"] = prompt
        for quant, gens in gens_by_quant.items():
            row[f"{quant}_gen"] = gens[i]
        samples.append(row)

    out = Path(args.out)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"\nSaved → {out}")

    summary = {"n": args.n, "num_fewshot": task.config.num_fewshot}
    for quant in gens_by_quant:
        key = f"{quant}_acc"
        summary[key] = sum(s[f"{quant}_ok"] for s in samples) / len(samples)

    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps({"summary": summary, "samples": samples}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved → {json_path}")
    for quant in gens_by_quant:
        print(f"  {quant} acc: {summary[f'{quant}_acc']:.1%}")


if __name__ == "__main__":
    main()

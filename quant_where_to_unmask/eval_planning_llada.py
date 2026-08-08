#!/usr/bin/env python3
"""Dream eval_planning.py protocol for LLaDA (Sudoku 4x4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import trange
from transformers import AutoTokenizer

from eval_llada import load_llada_model
from generate import generate
from tasks.sudoku4.utils import is_valid_sudoku_dream


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dream_postprocess(gen: str) -> str:
    return gen.split("<|endoftext|>")[0].split("\n\n")[0].replace(" ", "")


def build_inputs(data: list[dict], n_few_shots: int = 8) -> tuple[list[dict], list[str]]:
    template = (
        "Fill the positions where the values are 0 in a 4x4 grid with digits 1-4 so that "
        "each column, each row, and each of the four 2x2 subgrids that compose the grid "
        "contains all of the digits from 1 to 4.\n\n"
    )
    template += "\n\n".join(
        f"Input:\n{i['input']}\nOutput:\n{i['output']}" for i in data[:n_few_shots]
    )
    template += "\n\nInput:\n{input}\nOutput:\n "
    test = data[n_few_shots:]
    inputs = [template.format(input=i["input"]) for i in test]
    return test, inputs


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    texts: list[str],
    *,
    steps: int,
    max_new_tokens: int,
    block_length: int,
    remasking: str,
    temperature: float,
    device: str,
    mask_id: int = 126336,
) -> list[str]:
    responses = []
    for text in texts:
        ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
        attn = torch.ones_like(ids)
        out = generate(
            model,
            ids,
            attention_mask=attn,
            steps=steps,
            gen_length=max_new_tokens,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            mask_id=mask_id,
        )
        new_tokens = out[0, ids.shape[1] :]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


def eval_sudoku(
    model,
    tokenizer,
    data_path: Path,
    *,
    steps: int = 24,
    max_new_tokens: int = 24,
    block_length: int = 8,
    remasking: str = "entropy",
    temperature: float = 0.0,
    device: str = "cuda",
    limit: int | None = None,
) -> dict:
    data = read_jsonl(data_path)
    test, inputs = build_inputs(data)
    if limit is not None:
        test, inputs = test[:limit], inputs[:limit]

    generations = generate_batch(
        model,
        tokenizer,
        inputs,
        steps=steps,
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        remasking=remasking,
        temperature=temperature,
        device=device,
    )
    generations = [dream_postprocess(g) for g in generations]
    acc = sum(is_valid_sudoku_dream(i["input"], j) for i, j in zip(test, generations)) / len(test)
    exact = sum(
        i["output"].replace("\n", "") == j.replace("\n", "") for i, j in zip(test, generations)
    ) / len(test)
    return {
        "valid": acc,
        "exact": exact,
        "n": len(test),
        "example_input": inputs[0],
        "example_gen": generations[0],
        "example_gold": test[0]["output"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--quant", default="fp16", choices=["fp16", "bf16", "int4"])
    p.add_argument("--data", default="tasks/sudoku4/data/sudoku_4x4_10.jsonl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--remasking", default="low_confidence")
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--block-length", type=int, default=24)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    model = load_llada_model(args.model, args.quant, {"device_map": {"": args.device}})
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data_path = Path(args.data)

    if args.sweep:
        configs = [
            ("entropy", 24, 24, 8),
            ("entropy", 24, 24, 24),
            ("low_confidence", 24, 24, 24),
            ("topk_margin", 24, 24, 24),
            ("entropy", 24, 48, 8),
            ("entropy", 32, 32, 8),
        ]
        print(f"Dream-protocol sweep on {args.data} limit={args.limit or 'all'}")
        for remask, steps, mx, block in configs:
            r = eval_sudoku(
                model,
                tokenizer,
                data_path,
                steps=steps,
                max_new_tokens=mx,
                block_length=block,
                remasking=remask,
                device=args.device,
                limit=args.limit,
            )
            print(
                f"valid={r['valid']:.1%} exact={r['exact']:.1%} "
                f"remask={remask} steps={steps} max_new={mx} block={block}"
            )
        return

    r = eval_sudoku(
        model,
        tokenizer,
        data_path,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        block_length=args.block_length,
        remasking=args.remasking,
        device=args.device,
        limit=args.limit,
    )
    print(f"data={args.data} n={r['n']}")
    print(f"valid_sudoku (Dream metric): {r['valid']:.1%}")
    print(f"exact_match: {r['exact']:.1%}")
    print("example input tail:", r["example_input"][-120:])
    print("gold :", r["example_gold"])
    print("gen  :", r["example_gen"])


if __name__ == "__main__":
    main()

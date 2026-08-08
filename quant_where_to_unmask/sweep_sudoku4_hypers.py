#!/usr/bin/env python3
"""Quick hyper sweep for 4x4 Sudoku (Dream data) on a fixed subset."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from eval_llada import load_llada_model
from generate import generate
from tasks.sudoku4.utils import INSTRUCTION_4X4, N_FEWSHOT, clean_generation, is_valid_sudoku_dream


@dataclass(frozen=True)
class Config:
    remasking: str
    gen_length: int
    steps: int
    block_length: int
    temperature: float = 0.0

    def label(self) -> str:
        return (
            f"remask={self.remasking} gen={self.gen_length} "
            f"steps={self.steps} block={self.block_length} temp={self.temperature}"
        )


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_prompt(tokenizer, fewshot: list[dict], test_input: str) -> torch.Tensor:
    shots = "\n\n".join(
        f"Input:\n{r['input']}\nOutput:\n{r['output']}" for r in fewshot
    )
    text = (
        f"{INSTRUCTION_4X4}\n\n{shots}\n\nInput:\n{test_input}\nOutput:\n "
    )
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def score_config(
    model,
    tokenizer,
    device,
    fewshot: list[dict],
    test_rows: list[dict],
    cfg: Config,
    mask_id: int = 126336,
) -> tuple[float, float, int]:
    exact = valid = 0
    for row in test_rows:
        prompt = build_prompt(tokenizer, fewshot, row["input"]).unsqueeze(0).to(device)
        attn = torch.ones_like(prompt)
        out = generate(
            model,
            prompt,
            attention_mask=attn,
            steps=cfg.steps,
            gen_length=cfg.gen_length,
            block_length=cfg.block_length,
            temperature=cfg.temperature,
            remasking=cfg.remasking,
            mask_id=mask_id,
        )
        gen = clean_generation(tokenizer.decode(out[0, prompt.shape[1] :], skip_special_tokens=True))
        gold = row["output"]
        if gen.replace("\n", "") == gold.replace("\n", ""):
            exact += 1
        if is_valid_sudoku_dream(row["input"], gen):
            valid += 1
    n = len(test_rows)
    return exact / n, valid / n, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--quant", default="fp16", choices=["fp16", "bf16", "int4"])
    parser.add_argument("--data", default="tasks/sudoku4/data/sudoku_4x4_8.jsonl")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--full", action="store_true", help="Run best config on all 100 test puzzles")
    args = parser.parse_args()

    rows = load_rows(Path(args.data))
    fewshot = rows[:N_FEWSHOT]
    n_test = 100 if args.full else args.limit
    test_rows = rows[N_FEWSHOT : N_FEWSHOT + n_test]

    device = f"cuda:{args.gpu}"
    model = load_llada_model(args.model, args.quant, {"device_map": {"": device}})
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.full:
        configs = [Config("entropy", 24, 24, 8)]
    else:
        configs = [
            Config("low_confidence", 24, 24, 24),
        Config("entropy", 24, 24, 24),
        Config("topk_margin", 24, 24, 24),
        Config("random", 24, 24, 24),
        Config("low_confidence", 19, 19, 19),
        Config("entropy", 19, 19, 19),
        Config("low_confidence", 24, 48, 24),
        Config("entropy", 24, 48, 24),
        Config("low_confidence", 24, 96, 24),
        Config("entropy", 24, 96, 24),
        Config("low_confidence", 32, 32, 32),
        Config("entropy", 32, 32, 32),
        Config("low_confidence", 24, 24, 8),
        Config("entropy", 24, 24, 8),
        Config("low_confidence", 24, 24, 4),
        Config("entropy", 24, 24, 4),
        Config("entropy", 24, 24, 24, 0.1),
        # refine around best
        Config("entropy", 24, 48, 8),
        Config("entropy", 24, 96, 8),
        Config("entropy", 24, 24, 6),
        Config("entropy", 24, 24, 12),
        Config("entropy", 24, 24, 16),
        Config("topk_margin", 24, 24, 8),
        Config("entropy", 32, 32, 8),
        Config("entropy", 48, 48, 8),
        Config("entropy", 24, 24, 8, 0.05),
        Config("entropy", 24, 24, 8, 0.2),
        ]

    print(f"model={args.model} quant={args.quant} n={len(test_rows)} device={device}")
    print(f"{'exact':>7} {'valid':>7}  config")
    print("-" * 72)
    best = None
    for cfg in configs:
        exact, valid, _ = score_config(model, tokenizer, device, fewshot, test_rows, cfg)
        print(f"{exact:6.1%} {valid:6.1%}  {cfg.label()}")
        if best is None or valid > best[0] or (valid == best[0] and exact > best[1]):
            best = (valid, exact, cfg)
    print("-" * 72)
    print(
        f"BEST valid={best[0]:.1%} exact={best[1]:.1%} :: {best[2].label()}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 1b: multilingual language competence via translation-matching MCQ (OPUS-100).

Distractors are random translations from the same language split.
Models answer with a single letter A/B/C/D.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Reuse LLaDA generation from sibling project.
QWT = Path(__file__).resolve().parents[1] / "quant_where_to_unmask"
sys.path.insert(0, str(QWT))
from generate import generate  # noqa: E402

MODEL_PRESETS = {
    "llada": "GSAI-ML/LLaDA-8B-Base",
    "dream": "Dream-org/Dream-v0-Base-7B",
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen15": "Qwen/Qwen2.5-1.5B",
}

LANG_PAIRS = {
    "en": ("en-ru", "en"),
    "de": ("de-en", "de"),
    "ru": ("en-ru", "ru"),
    "tr": ("en-tr", "tr"),
    "fi": ("en-fi", "fi"),
    "zh": ("en-zh", "zh"),
    "ko": ("en-ko", "ko"),
}

LETTER = ["A", "B", "C", "D"]


@dataclass
class MCQExample:
    lang: str
    question: str
    choices: list[str]
    answer: str


def build_mcq(lang: str, n: int, seed: int = 42) -> list[MCQExample]:
    pair, col = LANG_PAIRS[lang]
    ds = load_dataset("Helsinki-NLP/opus-100", pair, split=f"validation[:{max(n * 4, 200)}]")
    rng = random.Random(seed)
    rows = [row["translation"] for row in ds if row["translation"].get(col)]
    rng.shuffle(rows)
    examples: list[MCQExample] = []
    for i in range(min(n, len(rows))):
        gold = rows[i][col]
        pool = [r[col] for j, r in enumerate(rows) if j != i and r.get(col) and r[col] != gold]
        if len(pool) < 3:
            continue
        distractors = rng.sample(pool, 3)
        choices = distractors + [gold]
        rng.shuffle(choices)
        answer = LETTER[choices.index(gold)]
        if lang == "en":
            q = f"Which English sentence is a natural, fluent sentence?\n"
        else:
            q = f"Which sentence is a natural, fluent sentence in {lang.upper()}?\n"
        for j, c in enumerate(choices):
            q += f"{LETTER[j]}. {c}\n"
        q += "Answer with a single letter (A, B, C, or D)."
        examples.append(MCQExample(lang=lang, question=q, choices=choices, answer=answer))
    return examples


def parse_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


@torch.no_grad()
def eval_qwen_ar(model, tokenizer, examples: list[MCQExample], device: str) -> list[dict]:
    results = []
    for ex in tqdm(examples, desc="qwen-ar"):
        prompt = ex.question + "\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_letter(gen) or "?"
        results.append({"pred": pred, "gold": ex.answer, "ok": pred == ex.answer, "raw": gen})
    return results


@torch.no_grad()
def eval_llada_dlm(model, tokenizer, examples: list[MCQExample], device: str, steps: int = 64) -> list[dict]:
    mask_id = 126336
    results = []
    for ex in tqdm(examples, desc="llada-dlm"):
        prompt = ex.question + "\nAnswer:"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_t = torch.tensor([ids], dtype=torch.long, device=device)
        gen_len = 8
        out, _, _ = generate(
            model,
            prompt_t,
            steps=steps,
            gen_length=gen_len,
            block_length=gen_len,
            remasking="low_confidence",
            mask_id=mask_id,
            tokenizer=tokenizer,
        )
        new_ids = out[0, prompt_t.shape[1] :].tolist()
        gen = tokenizer.decode(new_ids, skip_special_tokens=True)
        pred = parse_letter(gen) or "?"
        results.append({"pred": pred, "gold": ex.answer, "ok": pred == ex.answer, "raw": gen})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen15", choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--langs", default="en,ru,de,tr,fi,zh,ko")
    parser.add_argument("--n-per-lang", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("multi_language/results/competence_mcq.json"))
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model_id = MODEL_PRESETS[args.model]
    print(f"Loading {args.model} ({model_id}) on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    is_dlm = args.model == "llada"
    if is_dlm:
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=args.device
        )
        model.eval()
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=args.device
        )
        model.eval()

    all_rows = []
    summary = {}
    for lang in langs:
        examples = build_mcq(lang, args.n_per_lang)
        if not examples:
            print(f"Skip {lang}: not enough examples")
            continue
        if is_dlm:
            res = eval_llada_dlm(model, tokenizer, examples, args.device.split(":")[0] if ":" in args.device else args.device)
        else:
            res = eval_qwen_ar(model, tokenizer, examples, args.device)
        acc = sum(r["ok"] for r in res) / len(res)
        summary[lang] = {"acc": acc, "n": len(res)}
        print(f"{lang}: acc={acc:.1%} ({len(res)} examples)")
        all_rows.append({"lang": lang, "model": args.model, "acc": acc, "n": len(res), "details": res})

    args.out.write_text(json.dumps({"summary": summary, "results": all_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

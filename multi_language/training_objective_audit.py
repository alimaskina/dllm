#!/usr/bin/env python3
"""
Step 2: analytical training-objective audit for independent token masking.

For each word with k subtokens, under mask probability t per token:
  P(all siblings masked) = t^k
  P(at least one sibling visible) = 1 - t^k

We aggregate over the empirical word-length distribution per language.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from data_sources import load_opus_texts
from word_utils import iter_corpus_words

TOKENIZER_PRESETS = {
    "llada": "GSAI-ML/LLaDA-8B-Base",
    "qwen": "Qwen/Qwen2.5-7B",
}


@dataclass
class ObjectiveRow:
    lang: str
    tokenizer: str
    t: float
    n_words: int
    p_all_siblings_masked: float
    p_any_sibling_visible: float
    p_whole_word_unresolved: float  # same as all siblings masked for contiguous words
    expected_k: float


def word_k_distribution(texts: list[str], lang: str, tokenizer) -> Counter[int]:
    ctr: Counter[int] = Counter()
    for word, _ in iter_corpus_words(iter(texts), lang):
        k = len(tokenizer.encode(word, add_special_tokens=False))
        k = max(k, 1)
        ctr[k] += 1
    return ctr


def aggregate_objective(k_dist: Counter[int], t: float) -> tuple[float, float, float]:
    n = sum(k_dist.values())
    p_all = sum((t ** k) * c for k, c in k_dist.items()) / n
    p_any = 1.0 - p_all
    exp_k = sum(k * c for k, c in k_dist.items()) / n
    return p_all, p_any, exp_k


def main() -> None:
    parser = argparse.ArgumentParser(description="Training objective audit (Step 2)")
    parser.add_argument("--tokenizer", default="llada", choices=list(TOKENIZER_PRESETS.keys()))
    parser.add_argument("--langs", default="en,de,ru,tr,fi,zh,ko")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument(
        "--t-grid",
        default="0.1,0.2,0.3,0.5,0.7,0.9",
        help="Comma-separated mask ratios",
    )
    parser.add_argument("--out", type=Path, default=Path("multi_language/results/objective_audit.json"))
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    t_values = [float(x) for x in args.t_grid.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model_id = TOKENIZER_PRESETS[args.tokenizer]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    rows: list[dict] = []
    k_dists: dict[str, Counter[int]] = {}

    for lang in langs:
        texts = load_opus_texts(lang, max_samples=args.max_samples)
        k_dists[lang] = word_k_distribution(texts, lang, tokenizer)

    print(f"\nObjective audit — tokenizer={args.tokenizer}")
    for lang in langs:
        kd = k_dists[lang]
        n = sum(kd.values())
        print(f"\n{lang}: n_words={n}, k distribution: " + ", ".join(f"k={k}:{c/n:.2%}" for k, c in sorted(kd.items())[:6]))
        for t in t_values:
            p_all, p_any, exp_k = aggregate_objective(kd, t)
            row = ObjectiveRow(
                lang=lang,
                tokenizer=args.tokenizer,
                t=t,
                n_words=n,
                p_all_siblings_masked=p_all,
                p_any_sibling_visible=p_any,
                p_whole_word_unresolved=p_all,
                expected_k=exp_k,
            )
            rows.append(asdict(row))
            if t in (0.3, 0.7):
                print(f"  t={t:.1f}: P(all masked)={p_all:.4f}  P(any visible)={p_any:.4f}")

    # Cross-language gap at low t (key for hypothesis).
    print(f"\n{'=' * 60}")
    print("Low-noise whole-word-unresolved gap vs EN (t=0.3):")
    en_kd = k_dists["en"]
    en_p, _, _ = aggregate_objective(en_kd, 0.3)
    for lang in langs:
        if lang == "en":
            continue
        p, _, _ = aggregate_objective(k_dists[lang], 0.3)
        ratio = p / en_p if en_p > 0 else float("nan")
        print(f"  {lang}: P(all masked|t=0.3)={p:.5f}  Mismatch_vs_EN={ratio:.3f}x")

    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

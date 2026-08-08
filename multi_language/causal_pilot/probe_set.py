#!/usr/bin/env python3
"""Build / load fixed held-out probe set for causal pilot NLL eval."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

import sys

ML = Path(__file__).resolve().parents[1]
if str(ML) not in sys.path:
    sys.path.insert(0, str(ML))

from data_sources import load_opus_texts
from oracle_trajectory_random import tokenize_sentence_with_meta
from tokenization_audit import TOKENIZER_PRESETS


def build_probe_set(
    texts: list[str],
    tokenizer,
    *,
    lang: str,
    n_probe: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    pool: list[dict] = []
    for text in texts:
        ids, wps, metas = tokenize_sentence_with_meta(text, lang, tokenizer)
        multi = sum(1 for m in metas if m["k"] >= 2)
        has_k3 = any(m["k"] >= 3 for m in metas)
        if min_len <= len(ids) <= max_len and multi >= 2 and has_k3:
            pool.append(
                {
                    "text": text,
                    "token_ids": ids,
                    "word_positions": wps,
                    "word_metas": metas,
                }
            )
    rng.shuffle(pool)
    if len(pool) < n_probe:
        raise RuntimeError(f"Only {len(pool)} eligible probe sentences (need {n_probe})")
    return pool[:n_probe]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("--n-probe", type=int, default=120)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-pool", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="llada")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "probe_set_en.json",
    )
    args = parser.parse_args()

    model_id = TOKENIZER_PRESETS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    texts = load_opus_texts(args.lang, max_samples=args.max_pool)
    probe = build_probe_set(
        texts,
        tokenizer,
        lang=args.lang,
        n_probe=args.n_probe,
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed,
    )
    payload = {
        "model": args.model,
        "lang": args.lang,
        "seed": args.seed,
        "n_probe": len(probe),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "sentences": probe,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(probe)} sentences)")


if __name__ == "__main__":
    main()

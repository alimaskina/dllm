"""Training text stream for causal pilot."""

from __future__ import annotations

import random
from typing import Iterator

from datasets import load_dataset

from oracle_trajectory_random import tokenize_sentence


def load_train_texts(source: str = "opus_en", max_samples: int = 8000) -> list[str]:
    if source == "opus_en":
        n = max_samples if max_samples > 0 else None
        spec = f"train[:{n}]" if n else "train"
        ds = load_dataset("Helsinki-NLP/opus-100", "en-ru", split=spec)
        return [row["translation"]["en"] for row in ds if row["translation"].get("en")]
    if source == "wikitext2":
        n = max_samples if max_samples > 0 else None
        spec = f"train[:{n}]" if n else "train"
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=spec)
        return [row["text"].strip() for row in ds if row["text"].strip()]
    raise ValueError(f"Unknown source: {source}")


def iter_training_examples(
    texts: list[str],
    tokenizer,
    *,
    lang: str = "en",
    min_len: int = 16,
    max_len: int = 128,
    rng: random.Random | None = None,
) -> Iterator[tuple[list[int], list[list[int]]]]:
    """Yield (token_ids, word_positions) with length in [min_len, max_len]."""
    rng = rng or random.Random(0)
    indices = list(range(len(texts)))
    while True:
        rng.shuffle(indices)
        for i in indices:
            ids, wps = tokenize_sentence(texts[i], lang, tokenizer)
            if min_len <= len(ids) <= max_len:
                yield ids, wps

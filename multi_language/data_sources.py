"""Load multilingual text for PoC experiments."""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

# OPUS-100 pair config → target language column.
OPUS_LANG_CONFIGS: dict[str, tuple[str, str]] = {
    "en": ("en-ru", "en"),
    "de": ("de-en", "de"),
    "ru": ("en-ru", "ru"),
    "tr": ("en-tr", "tr"),
    "fi": ("en-fi", "fi"),
    "zh": ("en-zh", "zh"),
    "ko": ("en-ko", "ko"),
}


def load_opus_texts(lang: str, split: str = "validation", max_samples: int = 5000) -> list[str]:
    """Load target-language sentences from OPUS-100."""
    if lang not in OPUS_LANG_CONFIGS:
        raise ValueError(f"Unsupported language: {lang}")
    pair, col = OPUS_LANG_CONFIGS[lang]
    n = max_samples if max_samples > 0 else None
    slice_spec = f"{split}[:{n}]" if n else split
    ds = load_dataset("Helsinki-NLP/opus-100", pair, split=slice_spec)
    return [row["translation"][col] for row in ds if row["translation"].get(col)]


def iter_lang_texts(lang: str, max_samples: int = 5000) -> Iterator[str]:
    for text in load_opus_texts(lang, max_samples=max_samples):
        yield text

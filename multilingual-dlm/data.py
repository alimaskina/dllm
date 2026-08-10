"""Dataset loaders and word -> token mapping."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterator

from datasets import load_dataset
from transformers import PreTrainedTokenizer

SHARED_LANGS = ("en", "es", "de", "ru", "tr", "fi")
FLORES_LANGS = SHARED_LANGS

# ISO 639-1 -> FLORES-200 script code.
FLORES200_CONFIG = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "ru": "rus_Cyrl",
    "tr": "tur_Latn",
    "fi": "fin_Latn",
}
FLORES_CONFIG = FLORES200_CONFIG

BELEBELE_CONFIG = dict(FLORES200_CONFIG)
MASSIVE_CONFIG = {
    "en": "en",
    "es": "es",
    "de": "de",
    "ru": "ru",
    "tr": "tr",
    "fi": "fi",
}

FLORES_DATASET = "tomasmajercik/flores-parquet"
FLORES_PLUS_DATASET = "openlanguagedata/flores_plus"
FLORES_PLUS_FALLBACK = "yash9439/flores200"
BELEBELE_DATASET = "facebook/belebele"
MASSIVE_DATASET = "mteb/amazon_massive_scenario"

WHITESPACE_LANGS = frozenset({"en", "es", "de", "ru", "tr", "fi", "ko"})
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]+")


def _is_punct_or_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def segment_words(text: str, lang: str) -> list[str]:
    """Whitespace words for European langs; same rules as multi_language/word_utils.py."""
    if lang == "zh":
        return [c for m in _CJK_RE.finditer(text) for c in m.group()]
    if lang == "ko":
        words: list[str] = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            m = _HANGUL_RE.match(text, i)
            if m:
                words.extend(list(m.group()))
                i = m.end()
                continue
            j = i
            while j < len(text) and not text[j].isspace() and not _HANGUL_RE.match(text, j):
                j += 1
            tok = text[i:j].strip()
            if tok and not all(_is_punct_or_symbol(c) for c in tok):
                words.append(tok)
            i = j
        return words
    return [w for w in re.findall(r"\S+", text) if w]


def _words_with_char_spans(text: str, lang: str) -> list[tuple[str, int, int]]:
    """Return (word, char_start, char_end) aligned to the source text."""
    if lang in WHITESPACE_LANGS or lang not in {"zh", "ko"}:
        out: list[tuple[str, int, int]] = []
        for m in re.finditer(r"\S+", text):
            out.append((m.group(), m.start(), m.end()))
        return out

    words = segment_words(text, lang)
    spans: list[tuple[str, int, int]] = []
    search_from = 0
    for w in words:
        idx = text.find(w, search_from)
        if idx < 0:
            idx = text.find(w)
        if idx < 0:
            spans.append((w, search_from, search_from + len(w)))
            search_from += len(w)
        else:
            spans.append((w, idx, idx + len(w)))
            search_from = idx + len(w)
    return spans


def get_word_units(
    text: str,
    tokenizer: PreTrainedTokenizer,
    lang: str = "en",
) -> list[dict]:
    """
    Map each word to tokenizer positions.

    Returns list of dicts with keys:
      word, char_span (start, end), token_positions, token_ids, k
    """
    units: list[dict] = []
    offset = 0
    for word, char_start, char_end in _words_with_char_spans(text, lang):
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if not token_ids:
            continue
        positions = list(range(offset, offset + len(token_ids)))
        offset += len(token_ids)
        units.append(
            {
                "word": word,
                "char_span": (char_start, char_end),
                "token_positions": positions,
                "token_ids": token_ids,
                "k": len(token_ids),
            }
        )
    return units


def build_input_ids(word_units: list[dict]) -> list[int]:
    """Flatten per-word token ids into one sequence."""
    ids: list[int] = []
    for unit in word_units:
        ids.extend(unit["token_ids"])
    return ids


def debug_word_units(
    text: str,
    tokenizer: PreTrainedTokenizer,
    lang: str = "en",
    n: int = 5,
) -> None:
    """Print a few word -> subtoken alignments for manual inspection."""
    units = get_word_units(text, tokenizer, lang=lang)
    print(f"[debug_word_units] lang={lang}, n_tokens={sum(u['k'] for u in units)}")
    for unit in units[:n]:
        subtokens = tokenizer.convert_ids_to_tokens(unit["token_ids"])
        snippet = text[unit["char_span"][0] : unit["char_span"][1]]
        print(
            f"  word={unit['word']!r} chars={unit['char_span']} "
            f"pos={unit['token_positions']} k={unit['k']} "
            f"subtokens={subtokens} snippet={snippet!r}"
        )


def _check_langs(langs: tuple[str, ...], mapping: dict[str, str]) -> None:
    missing = set(langs) - set(mapping)
    if missing:
        raise ValueError(f"Unsupported langs: {sorted(missing)}")


def _belebele_example_id(link: str, question_number: int) -> str:
    return f"{link}#{question_number}"


def _flores_plus_split(split: str) -> str:
    if split in {"validation", "dev"}:
        return "dev"
    if split == "devtest":
        return "devtest"
    raise ValueError(f"Unsupported FLORES+ split {split!r}; expected dev, validation, or devtest")


def _take_examples(rows: list[dict], max_examples: int | None) -> list[dict]:
    if max_examples is None:
        return rows
    return rows[:max_examples]


def load_flores(
    langs: tuple[str, ...] = FLORES_LANGS,
    split: str = "validation",
    max_examples: int | None = None,
) -> list[dict]:
    """
    Main shared parallel corpus (FLORES-200 via parquet).

    Returns list of:
      {"example_id": int, "translations": {lang: sentence, ...}}
    """
    _check_langs(langs, FLORES_CONFIG)

    by_lang: dict[str, dict[int, str]] = {}
    for lang in langs:
        config = FLORES_CONFIG[lang]
        ds = load_dataset(FLORES_DATASET, config, split=split)
        by_lang[lang] = {int(row["id"]): row["sentence"] for row in ds}

    common_ids = sorted(set.intersection(*(set(d.keys()) for d in by_lang.values())))
    rows = [
        {
            "example_id": example_id,
            "translations": {lang: by_lang[lang][example_id] for lang in langs},
        }
        for example_id in common_ids
    ]
    return _take_examples(rows, max_examples)


def load_flores_plus(
    langs: tuple[str, ...] = FLORES_LANGS,
    split: str = "dev",
    max_examples: int | None = None,
) -> list[dict]:
    """
    FLORES+ parallel sentences with stable example_id.

    Tries the official gated Hub dataset first; falls back to consolidated
    FLORES-200 parquet (yash9439/flores200) when access is unavailable.
    """
    _check_langs(langs, FLORES200_CONFIG)
    split_name = _flores_plus_split(split)

    try:
        by_lang: dict[str, dict[int, str]] = {}
        for lang in langs:
            config = FLORES200_CONFIG[lang]
            ds = load_dataset(FLORES_PLUS_DATASET, config, split=split_name)
            by_lang[lang] = {int(row["id"]): row["text"] for row in ds}

        common_ids = sorted(set.intersection(*(set(d.keys()) for d in by_lang.values())))
        rows = [
            {
                "example_id": example_id,
                "translations": {lang: by_lang[lang][example_id] for lang in langs},
            }
            for example_id in common_ids
        ]
        return _take_examples(rows, max_examples)
    except Exception:
        ds = load_dataset(FLORES_PLUS_FALLBACK, split=split_name)
        rows: list[dict] = []
        for idx, row in enumerate(ds):
            example_id = idx + 1
            translations = {lang: row[FLORES200_CONFIG[lang]] for lang in langs}
            rows.append({"example_id": example_id, "translations": translations})
        return _take_examples(rows, max_examples)


def load_belebele(
    langs: tuple[str, ...] = FLORES_LANGS,
    split: str = "test",
    max_examples: int | None = None,
) -> list[dict]:
    """
    Belebele reading-comprehension MCQ examples for extra evals.

    Parallel examples are keyed by (link, question_number) -> example_id string.

    Returns list of:
      {
        "example_id": str,
        "items": {
          lang: {
            "passage": str,
            "question": str,
            "choices": list[str],
            "correct_answer_num": int,
          },
          ...
        },
      }
    """
    _check_langs(langs, BELEBELE_CONFIG)

    by_lang: dict[str, dict[str, dict]] = {}
    for lang in langs:
        config = BELEBELE_CONFIG[lang]
        ds = load_dataset(BELEBELE_DATASET, config, split=split)
        by_lang[lang] = {}
        for row in ds:
            example_id = _belebele_example_id(row["link"], int(row["question_number"]))
            by_lang[lang][example_id] = {
                "passage": row["flores_passage"],
                "question": row["question"],
                "choices": [
                    row["mc_answer1"],
                    row["mc_answer2"],
                    row["mc_answer3"],
                    row["mc_answer4"],
                ],
                "correct_answer_num": int(row["correct_answer_num"]),
            }

    common_ids = sorted(set.intersection(*(set(d.keys()) for d in by_lang.values())))
    rows = [
        {
            "example_id": example_id,
            "items": {lang: by_lang[lang][example_id] for lang in langs},
        }
        for example_id in common_ids
    ]
    return _take_examples(rows, max_examples)


def load_massive(
    langs: tuple[str, ...] = FLORES_LANGS,
    split: str = "test",
    max_examples: int | None = None,
) -> list[dict]:
    """
    MASSIVE intent utterances for extra evals.

    Parallel utterances share the dataset-native `id` as example_id.

    Returns list of:
      {
        "example_id": str,
        "label": str,
        "label_text": str,
        "utterances": {lang: text, ...},
      }
    """
    _check_langs(langs, MASSIVE_CONFIG)

    by_lang: dict[str, dict[str, dict]] = {}
    for lang in langs:
        config = MASSIVE_CONFIG[lang]
        ds = load_dataset(MASSIVE_DATASET, config, split=split)
        by_lang[lang] = {
            str(row["id"]): {
                "text": row["text"],
                "label": row["label"],
                "label_text": row["label_text"],
            }
            for row in ds
        }

    common_ids = sorted(set.intersection(*(set(d.keys()) for d in by_lang.values())), key=int)
    ref_lang = langs[0]
    rows = [
        {
            "example_id": example_id,
            "label": by_lang[ref_lang][example_id]["label"],
            "label_text": by_lang[ref_lang][example_id]["label_text"],
            "utterances": {lang: by_lang[lang][example_id]["text"] for lang in langs},
        }
        for example_id in common_ids
    ]
    return _take_examples(rows, max_examples)


def iter_flores_texts(
    lang: str,
    split: str = "validation",
    max_examples: int | None = None,
) -> Iterator[tuple[int, str]]:
    """Yield (example_id, sentence) for one language from the main FLORES loader."""
    for row in load_flores((lang,), split=split, max_examples=max_examples):
        yield row["example_id"], row["translations"][lang]

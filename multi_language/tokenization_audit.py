#!/usr/bin/env python3
"""Step 1: multilingual tokenization / fragmentation audit."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from transformers import AutoTokenizer

from data_sources import OPUS_LANG_CONFIGS, load_opus_texts
from word_utils import iter_corpus_words, segment_words, word_char_len

TOKENIZER_PRESETS: dict[str, str] = {
    "llada": "GSAI-ML/LLaDA-8B-Base",
    "qwen": "Qwen/Qwen2.5-7B",
    "dream": "Dream-org/Dream-v0-Base-7B",
    "fast_dllm": "Efficient-Large-Model/Fast_dLLM_v2_7B",
}


@dataclass
class LangStats:
    lang: str
    tokenizer: str
    n_words: int
    p_k1: float
    p_k2: float
    p_k3: float
    p_k4plus: float
    mean_tokens_per_word: float
    median_tokens_per_word: float
    mean_chars_per_word: float
    by_char_len: dict[str, dict[str, float]]


def count_subtokens(word: str, tokenizer) -> int:
    # No special tokens; measure raw piece count for the word string.
    ids = tokenizer.encode(word, add_special_tokens=False)
    return max(len(ids), 1)


def bucket_char_len(n: int) -> str:
    if n <= 3:
        return "1-3"
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    return "11+"


def audit_language(
    lang: str,
    tokenizer,
    tokenizer_name: str,
    texts: list[str],
) -> LangStats:
    k_counts: Counter[int] = Counter()
    char_lens: list[int] = []
    tokens_per_word: list[int] = []
    by_char: dict[str, Counter[int]] = defaultdict(Counter)

    for word, clen in iter_corpus_words(iter(texts), lang):
        k = count_subtokens(word, tokenizer)
        k_counts[k] += 1
        char_lens.append(clen)
        tokens_per_word.append(k)
        by_char[bucket_char_len(clen)][k] += 1

    n = sum(k_counts.values())
    if n == 0:
        raise RuntimeError(f"No words collected for lang={lang}")

    def p_at_least(threshold: int) -> float:
        return sum(c for k, c in k_counts.items() if k >= threshold) / n

    def p_exact(k: int) -> float:
        return k_counts.get(k, 0) / n

    char_summary = {}
    for bucket, ctr in sorted(by_char.items()):
        bn = sum(ctr.values())
        char_summary[bucket] = {
            "n": bn,
            "p_k1": ctr.get(1, 0) / bn,
            "p_k2": ctr.get(2, 0) / bn,
            "p_k3": ctr.get(3, 0) / bn,
            "p_k4plus": sum(c for kk, c in ctr.items() if kk >= 4) / bn,
            "mean_k": sum(kk * c for kk, c in ctr.items()) / bn,
        }

    return LangStats(
        lang=lang,
        tokenizer=tokenizer_name,
        n_words=n,
        p_k1=p_exact(1),
        p_k2=p_exact(2),
        p_k3=p_exact(3),
        p_k4plus=p_at_least(4),
        mean_tokens_per_word=statistics.mean(tokens_per_word),
        median_tokens_per_word=statistics.median(tokens_per_word),
        mean_chars_per_word=statistics.mean(char_lens),
        by_char_len=char_summary,
    )


def print_table(stats: list[LangStats]) -> None:
    tok = stats[0].tokenizer if stats else "?"
    print(f"\n{'=' * 72}")
    print(f"Tokenization audit — tokenizer={tok}")
    print(f"{'lang':>4}  {'n':>8}  {'k=1':>6}  {'k=2':>6}  {'k=3':>6}  {'k>=4':>6}  {'mean_k':>7}  {'med_k':>6}")
    for s in stats:
        print(
            f"{s.lang:>4}  {s.n_words:8d}  {100*s.p_k1:5.1f}%  {100*s.p_k2:5.1f}%  "
            f"{100*s.p_k3:5.1f}%  {100*s.p_k4plus:5.1f}%  {s.mean_tokens_per_word:7.3f}  {s.median_tokens_per_word:6.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multilingual tokenization audit (Step 1)")
    parser.add_argument(
        "--tokenizer",
        default="llada",
        choices=list(TOKENIZER_PRESETS.keys()) + ["all_unique"],
        help="Tokenizer preset or all_unique (llada + qwen families)",
    )
    parser.add_argument("--langs", default="en,de,ru,tr,fi,zh,ko")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("multi_language/results/tokenization_audit.json"))
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Cache corpus per language once.
    corpus: dict[str, list[str]] = {}
    for lang in langs:
        print(f"Loading OPUS-100 for {lang}...")
        corpus[lang] = load_opus_texts(lang, max_samples=args.max_samples)
        print(f"  {len(corpus[lang])} sentences")

    if args.tokenizer == "all_unique":
        tok_names = ["llada", "qwen"]
    else:
        tok_names = [args.tokenizer]

    all_results: list[dict] = []
    for tname in tok_names:
        model_id = TOKENIZER_PRESETS[tname]
        print(f"\nLoading tokenizer {tname} ({model_id})...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        lang_stats: list[LangStats] = []
        for lang in langs:
            print(f"  Auditing {lang}...")
            st = audit_language(lang, tokenizer, tname, corpus[lang])
            lang_stats.append(st)
            all_results.append(asdict(st))
        print_table(lang_stats)

    args.out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

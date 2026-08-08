#!/usr/bin/env python3
"""Count multi-token words on loop-filtered traces."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from analyze_multitoken_words import analyze_trace
from multitoken_word_filters import filter_instances, is_lexical


def has_repetition(text: str, min_ngram: int = 4, min_repeats: int = 3) -> tuple[bool, str | None]:
    toks = text.split()
    if len(toks) >= min_ngram * min_repeats:
        for n in range(min(12, len(toks) // min_repeats), min_ngram - 1, -1):
            for i in range(len(toks) - n * min_repeats + 1):
                phrase = tuple(toks[i : i + n])
                cnt = 1
                j = i + n
                while j + n <= len(toks) and tuple(toks[j : j + n]) == phrase:
                    cnt += 1
                    j += n
                if cnt >= min_repeats:
                    return True, f"phrase×{cnt}: {' '.join(phrase)[:60]}"
    return False, None


def has_artifact(text: str) -> tuple[bool, str | None]:
    patterns = [
        (r"\btarget:\b", "target:"),
        (r"\bcaption;\b", "caption;"),
        (r"\bdynasty;\b", "dynasty;"),
        (r"answer_choices", "answer_choices"),
        (r"\.input:Title:", "input:Title"),
        (r"-lrb-", "-lrb-"),
        (r"-rrb-", "-rrb-"),
    ]
    for pat, name in patterns:
        if re.search(pat, text, re.I):
            return True, name
    return False, None


def is_clean(response: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    rep, detail = has_repetition(response)
    if rep and detail:
        reasons.append(detail)
    art, name = has_artifact(response)
    if art and name:
        reasons.append(f"artifact:{name}")
    if len(response.strip()) < 20:
        reasons.append("too_short")
    return not reasons, reasons


def run_checkpoint(ckpt: Path, tokenizer, kind: str = "alpha", word_tier: str = "lexical") -> None:
    rows = []
    for rank_file in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(line) for line in rank_file.open(encoding="utf-8") if line.strip())
    clean_rows, dirty_rows = [], []
    for row in rows:
        ok, reasons = is_clean(row["response"])
        if ok:
            clean_rows.append(row)
        else:
            dirty_rows.append((row, reasons))

    def collect(rows_subset: list) -> list:
        out = []
        for row in rows_subset:
            trace_path = ckpt / row["trace_path"]
            if trace_path.exists():
                insts = analyze_trace(trace_path, tokenizer, method="ws", kind_filter=kind)
                out.extend(filter_instances(insts, tier=word_tier))
        return out

    all_inst = collect(rows)
    clean_inst = collect(clean_rows)
    hit = sum(
        1
        for row in clean_rows
        if filter_instances(
            analyze_trace(ckpt / row["trace_path"], tokenizer, method="ws", kind_filter=kind),
            tier=word_tier,
        )
    )

    tier_label = f"{kind}/{word_tier}" if word_tier != "none" else kind
    print(f"=== {ckpt.name} ===")
    print(f"traces: {len(rows)}")
    print(f"clean: {len(clean_rows)} ({100 * len(clean_rows) / len(rows):.1f}%)  dirty: {len(dirty_rows)}")
    print(f"multi-token {tier_label} ALL: {len(all_inst)} ({len(all_inst) / len(rows):.1f}/trace)")
    print(
        f"multi-token {tier_label} CLEAN: {len(clean_inst)} "
        f"({len(clean_inst) / max(len(clean_rows), 1):.1f}/clean trace)"
    )
    print(f"clean traces with >=1 multi-token word: {hit}/{len(clean_rows)}")
    print(
        f"  2-tok: {sum(1 for i in clean_inst if len(i.positions) == 2)}  "
        f"3-tok: {sum(1 for i in clean_inst if len(i.positions) == 3)}  "
        f"4+: {sum(1 for i in clean_inst if len(i.positions) >= 4)}"
    )
    print("top dirty reasons:")
    rc: Counter[str] = Counter()
    for _, reasons in dirty_rows:
        for r in reasons:
            rc[r.split(":")[0] if ":" in r else r] += 1
    for key, val in rc.most_common(8):
        print(f"  {key}: {val}")
    print("top clean words:")
    for word, cnt in Counter(i.word for i in clean_inst).most_common(12):
        print(f"  {word!r}: {cnt}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--kind", default="alpha")
    parser.add_argument(
        "--word-tier",
        choices=["none", "alpha", "lexical", "lexical_loose"],
        default="lexical",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    for name in args.checkpoints:
        ckpt = Path("checkpoints") / name
        if not (ckpt / "rank0.jsonl").exists():
            print(f"skip {name}: no rank0.jsonl")
            continue
        run_checkpoint(ckpt, tokenizer, kind=args.kind, word_tier=args.word_tier)


if __name__ == "__main__":
    main()

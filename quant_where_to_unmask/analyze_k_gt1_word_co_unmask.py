#!/usr/bin/env python3
"""
Simulate k>1 unmasking from k=1 traces: how often do top-k positions
fall in the same multi-token word?

Uses confidence rankings logged at each step (same state as actual decode at step 0;
later steps are counterfactual on the k=1 trajectory).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from transformers import AutoTokenizer

from analyze_multitoken_words import char_spans, find_word_instances_ws
from multitoken_word_filters import is_lexical


@dataclass
class StepSim:
    step: int
    k: int
    n_masked: int
    selected: list[int]
    # word stats among selected
    n_words_touched: int = 0
    max_tokens_same_word: int = 0
    words_with_2plus: int = 0
    words_with_full: int = 0  # all tokens of word in top-k
    selected_word_sizes: list[int] = field(default_factory=list)


def pos_to_word(positions_by_word: dict[int, int], pos: int) -> int | None:
    return positions_by_word.get(pos)


def build_word_map(final_ids: list[int], tokenizer) -> tuple[dict[int, int], dict[int, list[int]], list]:
    """pos_comp -> word_id; word_id -> list[pos]; word records."""
    spans, full = char_spans(final_ids, tokenizer)
    words = []
    pos_to_wid: dict[int, int] = {}
    wid_to_pos: dict[int, list[int]] = {}
    for wi, (word, kind, positions) in enumerate(find_word_instances_ws(full, spans)):
        if not is_lexical(word):
            continue
        words.append((word, positions))
        for p in positions:
            pos_to_wid[p] = wi
        wid_to_pos[wi] = positions
    return pos_to_wid, wid_to_pos, words


def topk_masked_positions(step: dict, k: int) -> list[int]:
    comp = step["completion"]
    ranked = [
        (comp["confidence"][i], i)
        for i, m in enumerate(comp["masked"])
        if m and comp["confidence"][i] is not None
    ]
    ranked.sort(key=lambda x: -x[0])
    return [i for _, i in ranked[:k]]


def simulate_step(step: dict, k: int, pos_to_wid: dict[int, int], wid_to_pos: dict[int, list[int]]) -> StepSim:
    n_masked = sum(1 for m in step["completion"]["masked"] if m)
    k_eff = min(k, n_masked)
    selected = topk_masked_positions(step, k_eff)

    by_word: Counter[int] = Counter()
    for p in selected:
        w = pos_to_word(pos_to_wid, p)
        if w is not None:
            by_word[w] += 1

    words_2plus = sum(1 for c in by_word.values() if c >= 2)
    words_full = sum(
        1 for w, cnt in by_word.items()
        if cnt == len(wid_to_pos[w])
    )

    return StepSim(
        step=step["step"],
        k=k_eff,
        n_masked=n_masked,
        selected=selected,
        n_words_touched=len(by_word),
        max_tokens_same_word=max(by_word.values(), default=0),
        words_with_2plus=words_2plus,
        words_with_full=words_full,
        selected_word_sizes=[len(wid_to_pos[w]) for w in by_word],
    )


def load_traces(checkpoint: Path, limit: int | None) -> list[dict]:
    traces = []
    for rank_file in sorted(checkpoint.glob("rank*.jsonl")):
        for line in rank_file.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            rel = row.get("trace_path")
            if not rel:
                continue
            path = checkpoint / rel
            if not path.exists():
                continue
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as f:
                traces.append(json.load(f))
            if limit and len(traces) >= limit:
                return traces
    return traces


def k_for_steps(gen_length: int, steps: int, step_idx: int = 0) -> int:
    """LLaDA schedule: base = n_masked // steps, +1 for first `remainder` steps."""
    n_masked = gen_length  # step 0: full block masked
    base = n_masked // steps
    rem = n_masked % steps
    return base + (1 if step_idx < rem else 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/results_wikitext_fp16_g64_n256"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=None,
        help="Explicit k values to simulate (default: derived from steps schedule)",
    )
    parser.add_argument(
        "--steps-schedule",
        type=int,
        nargs="+",
        default=[64, 32, 16, 8, 4],
        help="steps values → implied k at step 0 for gen-length",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    traces = load_traces(args.checkpoint, args.limit)
    if not traces:
        raise SystemExit(f"No traces in {args.checkpoint}")

    if args.k_values:
        k_list = args.k_values
        k_labels = {k: f"k={k}" for k in k_list}
    else:
        k_list = []
        k_labels = {}
        for steps in args.steps_schedule:
            k = k_for_steps(args.gen_length, steps, step_idx=0)
            k_list.append(k)
            k_labels[k] = f"steps={steps} → k={k}"

    # aggregate per k
    stats: dict[int, list[StepSim]] = {k: [] for k in k_list}
    step0_only: dict[int, list[StepSim]] = {k: [] for k in k_list}

    n_traces = 0
    n_words_per_trace: list[int] = []

    for tr in traces:
        final_ids = tr["steps_trace"][-1]["completion_tokens"]
        pos_to_wid, wid_to_pos, words = build_word_map(final_ids, tokenizer)
        n_traces += 1
        n_words_per_trace.append(len(words))

        for step in tr["steps_trace"]:
            for k in k_list:
                sim = simulate_step(step, k, pos_to_wid, wid_to_pos)
                stats[k].append(sim)
                if step["step"] == 0:
                    step0_only[k].append(sim)

    def summarize(rows: list[StepSim], label: str) -> None:
        if not rows:
            return
        n = len(rows)
        print(f"\n{'=' * 72}")
        print(label)
        print(f"n={n} step-samples")

        has_word = [r for r in rows if r.n_words_touched > 0]
        print(f"  top-k touches ≥1 multi-token word: {len(has_word)/n:.1%}")

        ge2 = [r for r in rows if r.max_tokens_same_word >= 2]
        print(f"  ≥2 tokens same word in top-k:     {len(ge2)/n:.1%}")

        ge3 = [r for r in rows if r.max_tokens_same_word >= 3]
        print(f"  ≥3 tokens same word in top-k:     {len(ge3)/n:.1%}")

        full = [r for r in rows if r.words_with_full >= 1]
        print(f"  full word (all tokens) in top-k:  {len(full)/n:.1%}")

        if ge2:
            print(f"  mean max same-word count (when ≥2): "
                  f"{statistics.mean(r.max_tokens_same_word for r in ge2):.2f}")

        # fraction of selected positions that are word tokens
        word_pos = sum(
            sum(1 for p in r.selected if p in {pp for wid in range(999) for pp in []})
            for r in rows
        )
        total_sel = sum(len(r.selected) for r in rows)
        # recompute word positions fraction properly
        word_frac = []
        for r in rows:
            # count selected that belong to any lexical word — use n_words_touched proxy
            word_frac.append(r.n_words_touched / max(len(r.selected), 1))
        print(f"  mean distinct words touched / k:    {statistics.mean(word_frac):.2f}")

        two_plus_rate = [r.words_with_2plus for r in rows]
        print(f"  mean words with ≥2 tokens picked:   {statistics.mean(two_plus_rate):.3f}")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Traces: {n_traces}, lexical words/trace: "
          f"{statistics.mean(n_words_per_trace):.2f} (median {statistics.median(n_words_per_trace):.0f})")

    for k in k_list:
        summarize(step0_only[k], f"STEP 0 ONLY — {k_labels[k]}")
    for k in k_list:
        summarize(stats[k], f"ALL STEPS — {k_labels[k]}")

    # detailed step 0 for k=2,4,8
    print(f"\n{'=' * 72}")
    print("STEP 0 — distribution of max same-word count")
    for k in k_list:
        ctr = Counter(r.max_tokens_same_word for r in step0_only[k])
        print(f"\n  {k_labels[k]}:")
        for cnt in sorted(ctr):
            print(f"    max_same_word={cnt}: {ctr[cnt]/len(step0_only[k]):.1%} ({ctr[cnt]})")

    # Example: step 0, k=4, cases with 2+ same word
    print(f"\n{'=' * 72}")
    print("EXAMPLES step 0, k=4, ≥2 tokens same word (first 5 traces)")
    shown = 0
    k_ex = 4 if 4 in k_list else k_list[min(1, len(k_list)-1)]
    for tr in traces[:50]:
        final_ids = tr["steps_trace"][-1]["completion_tokens"]
        spans, full = char_spans(final_ids, tokenizer)
        pos_to_wid, wid_to_pos, words = build_word_map(final_ids, tokenizer)
        step0 = tr["steps_trace"][0]
        sim = simulate_step(step0, k_ex, pos_to_wid, wid_to_pos)
        if sim.max_tokens_same_word < 2:
            continue
        # show which words
        by_word: dict[int, list[int]] = defaultdict(list)
        for p in sim.selected:
            w = pos_to_wid.get(p)
            if w is not None:
                by_word[w].append(p)
        print(f"\n  trace step0, selected pos={sim.selected}")
        for w, ps in sorted(by_word.items(), key=lambda x: -len(x[1])):
            if len(ps) >= 2:
                word, _pos = words[w]
                confs = [step0["completion"]["confidence"][p] for p in ps]
                toks = [tokenizer.decode([final_ids[p]]) for p in ps]
                print(f"    word {word!r}: pos={ps} tokens={toks} conf={[round(c,3) for c in confs]}")
        shown += 1
        if shown >= 5:
            break


if __name__ == "__main__":
    main()

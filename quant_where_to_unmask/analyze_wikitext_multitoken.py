#!/usr/bin/env python3
"""Aggregate multi-token unmasking analysis across WikiText g64 checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from analyze_multitoken_words import WordInstance, analyze_trace, order_pattern, step_gaps
from filter_loops_and_count import has_repetition
from multitoken_word_filters import is_lexical
    return f"{100 * n / d:.1f}%" if d else "n/a"


def load_instances(checkpoints: list[Path], tokenizer, *, no_loop: bool) -> tuple[list[WordInstance], int, int]:
    instances: list[WordInstance] = []
    n_traces = 0
    n_clean = 0
    for ckpt in checkpoints:
        rows = []
        for rank_file in sorted(ckpt.glob("rank*.jsonl")):
            rows.extend(json.loads(line) for line in rank_file.open(encoding="utf-8") if line.strip())
        for row in rows:
            n_traces += 1
            if no_loop and has_repetition(row["response"])[0]:
                continue
            n_clean += 1
            trace_path = ckpt / row["trace_path"]
            if not trace_path.exists():
                continue
            for inst in analyze_trace(trace_path, tokenizer, method="ws", kind_filter="alpha"):
                if is_lexical(inst.word):
                    instances.append(inst)
    return instances, n_traces, n_clean


def sibling_stats(instances: list[WordInstance]) -> dict:
    rows = []
    word_ok = []
    for inst in instances:
        final = {t.pos_comp: t.token_id for t in inst.tokens}
        sibs = []
        for s in inst.sibling_conf_at_first:
            if not s["still_masked"]:
                continue
            ok = s["predicted_token_id"] == final[s["pos_comp"]]
            rows.append((s["confidence"], ok))
            sibs.append(ok)
        if sibs:
            word_ok.append(all(sibs))
    out = {"n_siblings": len(rows), "n_words": len(word_ok)}
    if rows:
        out["sibling_acc"] = sum(ok for _, ok in rows) / len(rows)
        out["all_sibs_correct"] = sum(word_ok) / len(word_ok)
        for th in (0.5, 0.7, 0.8, 0.9):
            sub = [ok for c, ok in rows if c is not None and c >= th]
            if sub:
                out[f"co_unmask_{th}"] = (sum(sub) / len(sub), len(sub))
    return out


def print_report(instances: list[WordInstance], *, n_traces: int, n_clean: int, title: str) -> None:
    n = len(instances)
    print(f"\n{'=' * 80}")
    print(title)
    print(f"traces={n_traces}  no-loop traces={n_clean}")
    print(f"real multi-token word instances={n}")
    if not n:
        return

    hit_traces = len({id(i) for i in instances})  # wrong - need trace id
    print(f"avg per clean trace: {n / max(n_clean, 1):.1f}")

    print("\nLAYOUT")
    print(f"  consecutive positions: {pct(sum(i.consecutive for i in instances), n)}")
    print(f"  same-step unmask:      {pct(sum(i.same_step for i in instances), n)}")
    spans = [i.step_span for i in instances]
    print(
        f"  step_span: mean={statistics.mean(spans):.2f} median={statistics.median(spans):.0f} "
        f"p90={sorted(spans)[int(0.9 * n)]} max={max(spans)}"
    )

    order = Counter()
    for inst in instances:
        first = min(inst.tokens, key=lambda t: (t.step, t.pos_comp))
        if first.pos_comp == inst.positions[0]:
            order["leftmost"] += 1
        elif first.pos_comp == inst.positions[-1]:
            order["rightmost"] += 1
        else:
            order["middle"] += 1
    print("FIRST UNMASKED TOKEN")
    for k in ("leftmost", "middle", "rightmost"):
        print(f"  {k}: {pct(order[k], n)}")

    sib = sibling_stats(instances)
    print("\nCO-UNMASK SAFETY (sibling pred at first unmask)")
    if sib.get("n_siblings"):
        print(f"  sibling pred == final: {pct(int(sib['sibling_acc'] * sib['n_siblings']), sib['n_siblings'])}")
        print(f"  all siblings correct (per word): {pct(int(sib['all_sibs_correct'] * sib['n_words']), sib['n_words'])}")
        for th in (0.5, 0.7, 0.8, 0.9):
            key = f"co_unmask_{th}"
            if key in sib:
                acc, cnt = sib[key]
                print(f"  co-unmask if conf>={th}: acc={acc:.1%} (n={cnt})")

    two = [i for i in instances if len(i.positions) == 2]
    three = [i for i in instances if len(i.positions) == 3]
    fourp = [i for i in instances if len(i.positions) >= 4]
    print("\nBY TOKEN COUNT")
    for label, group in [("2", two), ("3", three), ("4+", fourp)]:
        if not group:
            continue
        print(f"  {label} tokens: n={len(group)} ({pct(len(group), n)})")
        if label == "2":
            gaps = [step_gaps(i)[0] for i in group]
            print(f"    consecutive steps: {pct(sum(g == 1 for g in gaps), len(group))}")
            print(f"    order LR: {pct(sum(order_pattern(i) == 'LR' for i in group), len(group))}")
            print(f"    order RL: {pct(sum(order_pattern(i) == 'RL' for i in group), len(group))}")
        if label == "3":
            print(f"    all 3 consecutive: {pct(sum(all(g == 1 for g in step_gaps(i)) for i in group), len(group))}")
            print(f"    strict LMR: {pct(sum(order_pattern(i) == 'LMR' for i in group), len(group))}")
            print(f"    strict RLM: {pct(sum(order_pattern(i) == 'RLM' for i in group), len(group))}")
            print(f"    mixed: {pct(sum(order_pattern(i) not in ('LMR', 'RLM') for i in group), len(group))}")
            for pat, c in Counter(order_pattern(i) for i in group).most_common(5):
                print(f"      {pat}: {c}")

    print("\nTOP WORDS")
    for word, c in Counter(i.word for i in instances).most_common(15):
        print(f"  {word!r}: {c}")

    print("\nEXAMPLES")
    ranked = []
    for inst in instances:
        final = {t.pos_comp: t.token_id for t in inst.tokens}
        wrong = sum(
            1
            for s in inst.sibling_conf_at_first
            if s["still_masked"] and s["predicted_token_id"] != final[s["pos_comp"]]
        )
        ranked.append((wrong, inst.step_span, len(inst.positions), inst))
    for inst in [x[3] for x in sorted(ranked, key=lambda t: t[:3], reverse=True)[:6]]:
        print(f"\n  {inst.word!r} ({len(inst.positions)} tok) pattern={order_pattern(inst)} span={inst.step_span}")
        for t in sorted(inst.tokens, key=lambda x: (x.step, x.pos_comp)):
            print(f"    step={t.step:3d} pos={t.pos_comp:3d} {t.token!r} conf={t.confidence:.3f}")
        final = {t.pos_comp: t.token_id for t in inst.tokens}
        for s in inst.sibling_conf_at_first:
            if not s["still_masked"]:
                continue
            mark = "OK" if s["predicted_token_id"] == final[s["pos_comp"]] else "WRONG"
            print(
                f"    sibling pos={s['pos_comp']}: pred={s['predicted_token']!r} "
                f"conf={s['confidence']:.3f} final={tokenizer.decode([final[s['pos_comp']]])!r} [{mark}]"
            )


tokenizer = None


def main() -> None:
    global tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "results_wikitext_fp16_g64_n256",
            "results_wikitext_fp16_g64_train_seed43_n512",
            "results_wikitext_fp16_g64_train_seed1043_n512",
        ],
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    ckpts = [Path("checkpoints") / name for name in args.checkpoints]

    all_inst, n_traces, _ = load_instances(ckpts, tokenizer, no_loop=False)
    clean_inst, _, n_clean = load_instances(ckpts, tokenizer, no_loop=True)

    print("WikiText g64 multi-token unmasking analysis")
    print_report(all_inst, n_traces=n_traces, n_clean=n_traces, title="ALL TRACES (real words)")
    print_report(clean_inst, n_traces=n_traces, n_clean=n_clean, title="NO-LOOP TRACES (real words)")


if __name__ == "__main__":
    main()

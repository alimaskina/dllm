#!/usr/bin/env python3
"""Detailed multi-token unmask stats on WikiText g64, no-loop traces only."""

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
from multitoken_word_filters import is_alpha_multi, is_lexical
    return f"{100 * n / d:.1f}%" if d else "n/a"


def pctile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(int(len(ys) * p), len(ys) - 1)]


def load_data(checkpoints: list[Path], tokenizer) -> tuple[list[WordInstance], list[WordInstance], int, int]:
    alpha: list[WordInstance] = []
    lexical: list[WordInstance] = []
    n_traces = 0
    n_clean = 0
    for ckpt in checkpoints:
        rows = []
        for rank_file in sorted(ckpt.glob("rank*.jsonl")):
            rows.extend(json.loads(line) for line in rank_file.open(encoding="utf-8") if line.strip())
        for row in rows:
            n_traces += 1
            if has_repetition(row["response"])[0]:
                continue
            n_clean += 1
            trace_path = ckpt / row["trace_path"]
            if not trace_path.exists():
                continue
            for inst in analyze_trace(trace_path, tokenizer, method="ws", kind_filter="alpha"):
                if is_alpha_multi(inst.word):
                    alpha.append(inst)
                if is_lexical(inst.word):
                    lexical.append(inst)
    return alpha, lexical, n_traces, n_clean


def first_token(inst: WordInstance):
    return min(inst.tokens, key=lambda t: (t.step, t.pos_comp))


def sibling_rows(inst: WordInstance) -> list[tuple[float | None, bool]]:
    final = {t.pos_comp: t.token_id for t in inst.tokens}
    out = []
    for s in inst.sibling_conf_at_first:
        if not s["still_masked"]:
            continue
        out.append((s["confidence"], s["predicted_token_id"] == final[s["pos_comp"]]))
    return out


def word_fully_consecutive(inst: WordInstance) -> bool:
    gaps = step_gaps(inst)
    return bool(gaps) and all(g == 1 for g in gaps)


def order_class(pat: str) -> str:
    if pat in ("LR", "LMR", "LMRM", "LMMR") or pat.replace("M", "") == "LR":
        if pat == "LR" or pat == "LMR":
            return "monotone_LR"
    if pat in ("RL", "RLM", "RML", "RMML") or pat == "RL" or pat == "RLM":
        return "monotone_RL"
    if all(c in "LR" for c in pat):
        if pat == "LR":
            return "monotone_LR"
        if pat == "RL":
            return "monotone_RL"
    # strict checks
    if pat == "LR" or pat == "LMR":
        return "monotone_LR"
    if pat == "RL" or pat == "RLM":
        return "monotone_RL"
    return "mixed"


def report(instances: list[WordInstance], *, label: str, n_traces: int, n_clean: int) -> None:
    n = len(instances)
    print(f"\n{'=' * 88}")
    print(label)
    print(f"no-loop traces: {n_clean}/{n_traces} ({pct(n_clean, n_traces)})")
    print(f"instances: {n}  ({n / max(n_clean, 1):.2f}/trace)")

    if not n:
        return

    # span distribution
    spans = [i.step_span for i in instances]
    print("\nSTEP SPAN (first→last token)")
    print(
        f"  mean={statistics.mean(spans):.2f}  median={statistics.median(spans):.0f}  "
        f"p10={pctile(spans,0.1):.0f}  p25={pctile(spans,0.25):.0f}  "
        f"p75={pctile(spans,0.75):.0f}  p90={pctile(spans,0.9):.0f}  max={max(spans)}"
    )
    for v in range(0, 8):
        c = sum(1 for s in spans if s == v)
        if c:
            print(f"    span={v}: {c} ({pct(c, n)})")
    c8p = sum(1 for s in spans if s >= 8)
    if c8p:
        print(f"    span>=8: {c8p} ({pct(c8p, n)})")

    print(f"\n  full word unmasked in consecutive steps: {pct(sum(word_fully_consecutive(i) for i in instances), n)}")

    # first position
    pos_ctr = Counter()
    for inst in instances:
        ft = first_token(inst)
        if ft.pos_comp == inst.positions[0]:
            pos_ctr["left"] += 1
        elif ft.pos_comp == inst.positions[-1]:
            pos_ctr["right"] += 1
        else:
            pos_ctr["middle"] += 1
    print("\nFIRST UNMASKED POSITION")
    for k in ("left", "middle", "right"):
        print(f"  {k}: {pos_ctr[k]} ({pct(pos_ctr[k], n)})")

    first_confs = [first_token(i).confidence for i in instances]
    print("\nFIRST TOKEN CONFIDENCE")
    print(
        f"  mean={statistics.mean(first_confs):.3f}  median={statistics.median(first_confs):.3f}  "
        f"p10={pctile(first_confs,0.1):.3f}  p90={pctile(first_confs,0.9):.3f}"
    )
    for lo, hi, name in [(0, 0.3, "<0.3"), (0.3, 0.7, "0.3-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, ">=0.9")]:
        sub = [i for i in instances if lo <= first_token(i).confidence < hi]
        if not sub:
            continue
        sacc = _sibling_acc(sub)
        print(f"  first_conf {name}: n={len(sub)}  sibling_acc={sacc:.1%}  all_sibs_ok={_all_sibs_ok(sub):.1%}")

    # sibling stats
    srows = [(c, ok) for i in instances for c, ok in sibling_rows(i)]
    print("\nSIBLING PRED AT FIRST UNMASK")
    print(f"  observations: {len(srows)}")
    if srows:
        confs = [c for c, _ in srows if c is not None]
        print(
            f"  sibling conf: mean={statistics.mean(confs):.3f} median={statistics.median(confs):.3f} "
            f"p10={pctile(confs,0.1):.3f} p90={pctile(confs,0.9):.3f}"
        )
        print(f"  pred==final: {sum(ok for _, ok in srows)}/{len(srows)} ({pct(sum(ok for _, ok in srows), len(srows))})")
        print(f"  all siblings ok (per word): {_all_sibs_ok(instances):.1%}")
        for lo, hi, name in [(0, 0.3, "<0.3"), (0.3, 0.5, "0.3-0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, ">=0.9")]:
            sub = [(c, ok) for c, ok in srows if c is not None and lo <= c < hi]
            if sub:
                print(f"    conf {name}: acc={sum(ok for _, ok in sub)/len(sub):.1%} (n={len(sub)})")

    print("\nCO-UNMASK POLICIES")
    for th in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        sub = [ok for c, ok in srows if c is not None and c >= th]
        if sub:
            wrong = len(sub) - sum(sub)
            print(f"  sibling conf>={th}: acc={sum(sub)/len(sub):.1%}  n={len(sub)}  wrong={wrong}")

    # per token count
    print("\nBY TOKEN COUNT")
    for ntok in [2, 3, 4, 5]:
        grp = [i for i in instances if len(i.positions) == ntok]
        if not grp:
            continue
        print(f"  {ntok} tokens: n={len(grp)} ({pct(len(grp), n)})  span_med={statistics.median(i.step_span for i in grp):.0f}  sib_acc={_sibling_acc(grp):.1%}")
        if ntok == 2:
            gaps = [step_gaps(i)[0] for i in grp]
            for g in sorted(set(gaps)):
                if sum(1 for x in gaps if x == g) >= 10:
                    print(f"    gap={g}: {sum(1 for x in gaps if x == g)} ({pct(sum(1 for x in gaps if x == g), len(grp))})")
            print(f"    LR={pct(sum(order_pattern(i)=='LR' for i in grp), len(grp))}  RL={pct(sum(order_pattern(i)=='RL' for i in grp), len(grp))}")
        if ntok == 3:
            print(f"    consecutive 3 steps: {pct(sum(word_fully_consecutive(i) for i in grp), len(grp))}")
            for pat, c in Counter(order_pattern(i) for i in grp).most_common(6):
                print(f"      {pat}: {c} ({pct(c, len(grp))})")
    grp4p = [i for i in instances if len(i.positions) >= 4]
    if grp4p:
        print(f"  4+ tokens: n={len(grp4p)} ({pct(len(grp4p), n)})  span_mean={statistics.mean(i.step_span for i in grp4p):.2f}  sib_acc={_sibling_acc(grp4p):.1%}")
        oc = Counter(order_class(order_pattern(i)) for i in grp4p)
        for k, c in oc.most_common():
            print(f"    {k}: {c} ({pct(c, len(grp4p))})")

    print("\nTOP WORDS")
    for w, c in Counter(i.word for i in instances).most_common(12):
        print(f"  {w!r}: {c}")


def _sibling_acc(instances: list[WordInstance]) -> float:
    rows = [ok for _, ok in ((c, ok) for i in instances for c, ok in sibling_rows(i))]
    return sum(rows) / len(rows) if rows else 0.0


def _all_sibs_ok(instances: list[WordInstance]) -> float:
    ok_words = []
    for inst in instances:
        rows = sibling_rows(inst)
        if rows:
            ok_words.append(all(ok for _, ok in rows))
    return sum(ok_words) / len(ok_words) if ok_words else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "results_wikitext_fp16_g64_n256",
            "results_wikitext_fp16_g64_train_seed43_n512",
            "results_wikitext_fp16_g64_train_seed1043_n512",
            "results_wikitext_fp16_g64_train_seed2043_n512",
            "results_wikitext_fp16_g64_train_seed3043_n512",
            "results_wikitext_fp16_g64_train_seed4043_n512",
            "results_wikitext_fp16_g64_train_seed5043_n512",
        ],
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    ckpts = [Path("checkpoints") / name for name in args.checkpoints]
    alpha, lexical, n_traces, n_clean = load_data(ckpts, tokenizer)

    print("WikiText g64 — detailed unmask stats (NO-LOOP ONLY)")
    report(alpha, label="TIER A: alpha multi-token (incl. infobox tokens)", n_traces=n_traces, n_clean=n_clean)
    report(lexical, label="TIER B: strict lexical (no word+punctuation)", n_traces=n_traces, n_clean=n_clean)


if __name__ == "__main__":
    main()

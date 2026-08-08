#!/usr/bin/env python3
"""Analyze how multi-token words unmask step-by-step in generation traces."""

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

from multitoken_word_filters import filter_instances


@dataclass
class TokenUnmask:
    pos_comp: int
    token_id: int
    token: str
    step: int
    confidence: float


@dataclass
class WordInstance:
    word: str
    kind: str  # alpha | digit | alnum | other
    positions: list[int]
    tokens: list[TokenUnmask]
    consecutive: bool
    same_step: bool
    step_span: int
    first_pos: int
    first_step: int
    sibling_conf_at_first: list[dict] = field(default_factory=list)


def load_trace(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_unmask_timeline(trace: dict) -> dict[int, TokenUnmask]:
    timeline: dict[int, TokenUnmask] = {}
    for step in trace["steps_trace"]:
        for u in step["unmasked"]:
            timeline[u["pos_comp"]] = TokenUnmask(
                pos_comp=u["pos_comp"],
                token_id=u["token_id"],
                token=u["token"],
                step=step["step"],
                confidence=u["confidence"],
            )
    return timeline


def char_spans(final_ids: list[int], tokenizer) -> tuple[list[tuple[int, int, int, str]], str]:
    full = ""
    spans: list[tuple[int, int, int, str]] = []
    for i, tid in enumerate(final_ids):
        piece = tokenizer.decode([tid])
        start = len(full)
        full += piece
        spans.append((i, start, start + len(piece), piece))
    return spans, full


def classify_word(text: str) -> str:
    core = re.sub(r"^[^\w']+|[^\w']+$", "", text)
    if not core:
        return "other"
    if core.isdigit():
        return "digit"
    if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", core):
        return "alpha"
    if re.search(r"[A-Za-z]", core) and re.search(r"\d", core):
        return "alnum"
    return "other"


def find_word_instances_strict(full: str, spans: list[tuple[int, int, int, str]]) -> list[tuple[str, str, list[int]]]:
    out: list[tuple[str, str, list[int]]] = []
    for m in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", full):
        word = m.group()
        ws, we = m.start(), m.end()
        pos = [i for i, st, en, _ in spans if not (en <= ws or st >= we)]
        if len(pos) < 2:
            continue
        kind = "digit" if word.isdigit() else "alpha"
        out.append((word, kind, pos))
    return out


def find_word_instances_ws(full: str, spans: list[tuple[int, int, int, str]]) -> list[tuple[str, str, list[int]]]:
    out: list[tuple[str, str, list[int]]] = []
    for m in re.finditer(r"\S+", full):
        text = m.group()
        ws, we = m.start(), m.end()
        pos = [i for i, st, en, _ in spans if not (en <= ws or st >= we)]
        if len(pos) < 2:
            continue
        out.append((text, classify_word(text), pos))
    return out


def sibling_confidence_at_first_unmask(trace: dict, positions: list[int], first_pos: int, first_step: int) -> list[dict]:
    step_data = next(s for s in trace["steps_trace"] if s["step"] == first_step)
    comp = step_data["completion"]
    out = []
    for pos in positions:
        if pos == first_pos:
            continue
        if not comp["masked"][pos]:
            out.append(
                {
                    "pos_comp": pos,
                    "still_masked": False,
                    "confidence": None,
                    "predicted_token_id": comp["predicted_token_id"][pos],
                    "predicted_token": None,
                }
            )
            continue
        tid = comp["predicted_token_id"][pos]
        out.append(
            {
                "pos_comp": pos,
                "still_masked": True,
                "confidence": comp["confidence"][pos],
                "predicted_token_id": tid,
                "predicted_token": None,
            }
        )
    return out


def analyze_trace(
    trace_path: Path,
    tokenizer,
    *,
    method: str = "ws",
    min_word_len: int = 2,
    kind_filter: str | None = None,
) -> list[WordInstance]:
    trace = load_trace(trace_path)
    final_ids = trace["steps_trace"][-1]["completion_tokens"]
    timeline = build_unmask_timeline(trace)
    spans, full = char_spans(final_ids, tokenizer)
    finder = find_word_instances_ws if method == "ws" else find_word_instances_strict

    instances: list[WordInstance] = []
    for word, kind, positions in finder(full, spans):
        core = re.sub(r"^[^\w']+|[^\w']+$", "", word)
        if len(core) < min_word_len:
            continue
        if kind_filter and kind != kind_filter:
            continue
        tokens = [timeline[p] for p in positions]
        first = min(tokens, key=lambda t: (t.step, t.pos_comp))
        sib = sibling_confidence_at_first_unmask(trace, positions, first.pos_comp, first.step)
        for s in sib:
            if s["predicted_token_id"] is not None:
                s["predicted_token"] = tokenizer.decode([s["predicted_token_id"]])

        instances.append(
            WordInstance(
                word=word,
                kind=kind,
                positions=positions,
                tokens=tokens,
                consecutive=all(positions[i] + 1 == positions[i + 1] for i in range(len(positions) - 1)),
                same_step=len({t.step for t in tokens}) == 1,
                step_span=max(t.step for t in tokens) - min(t.step for t in tokens),
                first_pos=first.pos_comp,
                first_step=first.step,
                sibling_conf_at_first=sib,
            )
        )
    return instances


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def order_pattern(inst: WordInstance) -> str:
    by_pos = sorted(inst.tokens, key=lambda t: t.pos_comp)
    by_time = sorted(inst.tokens, key=lambda t: (t.step, t.pos_comp))
    poss = [t.pos_comp for t in by_pos]
    labels = {
        p: ("L" if p == poss[0] else "R" if p == poss[-1] else "M")
        for p in poss
    }
    return "".join(labels[t.pos_comp] for t in by_time)


def step_gaps(inst: WordInstance) -> list[int]:
    steps = [t.step for t in sorted(inst.tokens, key=lambda t: (t.step, t.pos_comp))]
    return [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]


def print_report(instances: list[WordInstance], *, n_traces: int, checkpoint: Path, method: str, kind: str) -> None:
    n = len(instances)
    print(f"checkpoint={checkpoint}")
    print(f"method={method}  kind={kind}")
    print(f"traces={n_traces}")
    print(f"multi-token instances={n}")
    if not n:
        return

    print(f"\n{'=' * 80}")
    print("LAYOUT")
    print(f"  consecutive positions: {pct(sum(i.consecutive for i in instances), n)}")
    print(f"  same-step unmask:      {pct(sum(i.same_step for i in instances), n)}")
    spans = [i.step_span for i in instances]
    print(
        f"  step_span: mean={statistics.mean(spans):.2f} median={statistics.median(spans):.0f} "
        f"p90={sorted(spans)[int(0.9 * n)]} max={max(spans)}"
    )

    print(f"\n{'=' * 80}")
    print("CO-UNMASK SAFETY (sibling pred at first unmask vs final)")
    sib_rows = []
    word_rows = []
    for inst in instances:
        final = {t.pos_comp: t.token_id for t in inst.tokens}
        sibs = []
        for s in inst.sibling_conf_at_first:
            if not s["still_masked"]:
                continue
            ok = s["predicted_token_id"] == final[s["pos_comp"]]
            sib_rows.append((s["confidence"], ok))
            sibs.append(ok)
        if sibs:
            word_rows.append(all(sibs))
    if sib_rows:
        correct = sum(ok for _, ok in sib_rows)
        print(f"  sibling pred == final: {correct}/{len(sib_rows)} ({correct/len(sib_rows):.1%})")
        print(f"  all siblings correct (per word): {sum(word_rows)}/{len(word_rows)} ({sum(word_rows)/len(word_rows):.1%})")
        for th in (0.5, 0.7, 0.8, 0.9):
            sub = [ok for c, ok in sib_rows if c is not None and c >= th]
            if sub:
                print(f"  if co-unmask when conf>={th}: acc={sum(sub)/len(sub):.1%} (n={len(sub)})")

    two = [i for i in instances if len(i.positions) == 2]
    three = [i for i in instances if len(i.positions) == 3]
    if two:
        print(f"\n{'=' * 80}")
        print(f"2-TOKEN WORDS (n={len(two)})")
        gaps = [step_gaps(i)[0] for i in two]
        print(f"  consecutive steps (gap=1): {pct(sum(g == 1 for g in gaps), len(two))}")
        print(f"  order LR: {pct(sum(order_pattern(i) == 'LR' for i in two), len(two))}")
        print(f"  order RL: {pct(sum(order_pattern(i) == 'RL' for i in two), len(two))}")
    if three:
        print(f"\n{'=' * 80}")
        print(f"3-TOKEN WORDS (n={len(three)})")
        print(f"  all 3 consecutive steps: {pct(sum(all(g == 1 for g in step_gaps(i)) for i in three), len(three))}")
        print(f"  strict LMR: {pct(sum(order_pattern(i) == 'LMR' for i in three), len(three))}")
        print(f"  strict RLM: {pct(sum(order_pattern(i) == 'RLM' for i in three), len(three))}")
        print(f"  mixed: {pct(sum(order_pattern(i) not in ('LMR', 'RLM') for i in three), len(three))}")
        for pat, c in Counter(order_pattern(i) for i in three).most_common(6):
            print(f"    {pat}: {c}")

    print(f"\n{'=' * 80}")
    print("TOP WORDS")
    for word, c in Counter(i.word for i in instances).most_common(20):
        print(f"  {word!r}: {c}")

    print(f"\n{'=' * 80}")
    print("EXAMPLES (interesting order / wrong sibling pred)")
    ranked = []
    for inst in instances:
        wrong = 0
        final = {t.pos_comp: t.token_id for t in inst.tokens}
        for s in inst.sibling_conf_at_first:
            if s["still_masked"] and s["predicted_token_id"] != final[s["pos_comp"]]:
                wrong += 1
        ranked.append((wrong, inst.step_span, len(inst.positions), inst))
    for inst in [x[3] for x in sorted(ranked, key=lambda x: x[:3], reverse=True)[:8]]:
        print(f"\n--- {inst.word!r} ({len(inst.positions)} tok) pattern={order_pattern(inst)} ---")
        for t in sorted(inst.tokens, key=lambda x: (x.step, x.pos_comp)):
            print(f"  step={t.step:3d} pos={t.pos_comp:3d} {t.token!r} conf={t.confidence:.4f}")
        final = {t.pos_comp: t.token_id for t in inst.tokens}
        for s in inst.sibling_conf_at_first:
            if not s["still_masked"]:
                continue
            mark = "OK" if s["predicted_token_id"] == final[s["pos_comp"]] else "WRONG"
            print(
                f"  sibling pos={s['pos_comp']}: pred={s['predicted_token']!r} conf={s['confidence']:.4f} "
                f"final={tokenizer.decode([final[s['pos_comp']]])!r} [{mark}]"
            )


tokenizer = None  # set in main for examples


def main() -> None:
    global tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_gsm8k_fp16_n256_fast_2gpu")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--method", choices=["ws", "strict"], default="ws")
    parser.add_argument("--kind", choices=["all", "alpha", "digit", "alnum"], default="alpha")
    parser.add_argument(
        "--word-tier",
        choices=["none", "alpha", "lexical", "lexical_loose"],
        default="lexical",
        help="Filter word instances after trace parse (default: strict lexical words)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kind_filter = None if args.kind == "all" else args.kind

    trace_files = sorted(ckpt.glob("traces/rank*/*.json.gz"))
    if args.limit:
        trace_files = trace_files[: args.limit]

    all_instances: list[WordInstance] = []
    for path in trace_files:
        all_instances.extend(
            analyze_trace(path, tokenizer, method=args.method, kind_filter=kind_filter)
        )
    all_instances = filter_instances(all_instances, tier=args.word_tier)

    report_kind = args.kind if args.word_tier == "none" else f"{args.kind}/{args.word_tier}"
    if args.report or args.kind != "all" or args.word_tier != "none":
        print_report(
            all_instances,
            n_traces=len(trace_files),
            checkpoint=ckpt,
            method=args.method,
            kind=report_kind,
        )
        return

    summarize = print_report  # fallback
    summarize(all_instances, n_traces=len(trace_files), checkpoint=ckpt, method=args.method, kind=report_kind)


if __name__ == "__main__":
    main()

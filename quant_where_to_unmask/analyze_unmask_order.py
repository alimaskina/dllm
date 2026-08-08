#!/usr/bin/env python3
"""Detailed FP16 vs INT4 unmasking order analysis from traces."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from lm_eval import tasks
from lm_eval.tasks import TaskManager

from tasks.sudoku4.utils import clean_generation

DIGIT_POS = {1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19}
FMT_POS = {0, 5, 10, 15, 20, 21, 22, 23}
FMT_PREFIX = [0, 5, 10, 15, 20]
POS_LABEL = {
    0: "sp",
    5: "nl1",
    10: "nl2",
    15: "nl3",
    20: "nl4",
    21: "tnl1",
    22: "tnl2",
    23: "tnl3",
    1: "r1c1",
    2: "r1c2",
    3: "r1c3",
    4: "r1c4",
    6: "r2c1",
    7: "r2c2",
    8: "r2c3",
    9: "r2c4",
    11: "r3c1",
    12: "r3c2",
    13: "r3c3",
    14: "r3c4",
    16: "r4c1",
    17: "r4c2",
    18: "r4c3",
    19: "r4c4",
}


def row_of(pos: int) -> int:
    if pos in (1, 2, 3, 4):
        return 1
    if pos in (6, 7, 8, 9):
        return 2
    if pos in (11, 12, 13, 14):
        return 3
    if pos in (16, 17, 18, 19):
        return 4
    return 0


def load_order(ckpt_dir: Path, row: dict) -> list[int]:
    with gzip.open(ckpt_dir / row["trace_path"], "rt", encoding="utf-8") as f:
        tr = json.load(f)
    return [u["pos_comp"] for step in tr["steps_trace"] for u in step["unmasked"]]


def rank_map(order: list[int]) -> dict[int, int]:
    return {pos: i for i, pos in enumerate(order)}


def kendall_tau(order_a: list[int], order_b: list[int]) -> float:
    ra, rb = rank_map(order_a), rank_map(order_b)
    conc = disc = 0
    for x, y in combinations(ra, 2):
        dx, dy = ra[x] - ra[y], rb[x] - rb[y]
        if dx == 0 or dy == 0:
            continue
        if (dx > 0) == (dy > 0):
            conc += 1
        else:
            disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else 1.0


def prefix_match(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n


def classify_outcome(fp16_resp: str, int4_resp: str, gold: str) -> str:
    ex_f = clean_generation(fp16_resp).replace("\n", "") == gold.replace("\n", "")
    ex_i = clean_generation(int4_resp).replace("\n", "") == gold.replace("\n", "")
    if ex_f and ex_i:
        return "both_ok"
    if ex_f:
        return "fp16_only"
    if ex_i:
        return "int4_only"
    return "both_wrong"


def load_samples(task_name: str, fp16_dir: Path, int4_dir: Path) -> list[dict]:
    tm = TaskManager(include_path="tasks")
    docs = {
        i: d
        for i, d in enumerate(tasks.get_task_dict([task_name], task_manager=tm)[task_name].test_docs())
    }
    fp16_rows = {
        json.loads(line)["doc_id"]: json.loads(line)
        for path in sorted(fp16_dir.glob("rank*.jsonl"))
        for line in path.open(encoding="utf-8")
        if line.strip()
    }
    int4_rows = {
        json.loads(line)["doc_id"]: json.loads(line)
        for path in sorted(int4_dir.glob("rank*.jsonl"))
        for line in path.open(encoding="utf-8")
        if line.strip()
    }

    samples = []
    for doc_id in sorted(docs):
        if doc_id not in fp16_rows or doc_id not in int4_rows:
            continue
        o_f = load_order(fp16_dir, fp16_rows[doc_id])
        o_i = load_order(int4_dir, int4_rows[doc_id])
        d_f = [p for p in o_f if p in DIGIT_POS]
        d_i = [p for p in o_i if p in DIGIT_POS]
        samples.append(
            {
                "doc_id": doc_id,
                "outcome": classify_outcome(
                    fp16_rows[doc_id]["response"],
                    int4_rows[doc_id]["response"],
                    docs[doc_id]["target"],
                ),
                "o_f": o_f,
                "o_i": o_i,
                "d_f": d_f,
                "d_i": d_i,
                "prefix": prefix_match(o_f, o_i),
                "tau": kendall_tau(o_f, o_i),
                "tau_digit": kendall_tau(d_f, d_i),
            }
        )
    return samples


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def summarize_group(samples: list[dict], title: str) -> None:
    n = len(samples)
    if not n:
        return
    print(f"\n{'=' * 72}")
    print(title)
    print(f"n={n}")
    print(f"  identical 24-step order:     {sum(s['o_f'] == s['o_i'] for s in samples) / n:.1%}")
    print(f"  shared prefix length:      mean={mean([s['prefix'] for s in samples]):.1f}  median={statistics.median([s['prefix'] for s in samples]):.0f}")
    print(f"  Kendall tau (24 positions): mean={mean([s['tau'] for s in samples]):.3f}  median={statistics.median([s['tau'] for s in samples]):.3f}")
    print(f"  Kendall tau (16 digits):    mean={mean([s['tau_digit'] for s in samples]):.3f}  median={statistics.median([s['tau_digit'] for s in samples]):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sudoku4_2shot_large")
    parser.add_argument("--fp16", default="checkpoints/results_sudoku_fp16_large_4x4_2shot_2gpu")
    parser.add_argument("--int4", default="checkpoints/results_sudoku_int4_large_4x4_2shot_2gpu")
    args = parser.parse_args()

    samples = load_samples(args.task, Path(args.fp16), Path(args.int4))
    n = len(samples)
    print(f"task={args.task}  samples={n}")

    summarize_group(samples, "ALL")
    for cat in ("both_ok", "fp16_only", "int4_only", "both_wrong"):
        summarize_group([s for s in samples if s["outcome"] == cat], f"outcome={cat}")

    # Phase 1: format prefix
    fmt_same = sum(1 for s in samples if s["o_f"][:5] == FMT_PREFIX and s["o_i"][:5] == FMT_PREFIX)
    print(f"\n{'=' * 72}")
    print("PHASE 1: format tokens")
    print(f"  both start [0,5,10,15,20]: {fmt_same}/{n} ({fmt_same/n:.1%})")
    print("  interpretation: first 5 steps almost always unmask leading space + 3 internal newlines + row4 newline")

    # Per-step agreement
    print(f"\n{'=' * 72}")
    print("PER-STEP position agreement")
    print("step | agree | phase")
    for step in range(24):
        agree = sum(1 for s in samples if s["o_f"][step] == s["o_i"][step]) / n
        pos = samples[0]["o_f"][step] if samples else 0
        phase = "format" if step < 5 else ("digit/tail" if step < 20 else "tail")
        print(f" {step:2d}  | {agree:5.1%} | {phase:10s} typical_fp16_pos={POS_LABEL.get(samples[0]['o_f'][step], '?')}")

    # Digit subsequence
    print(f"\n{'=' * 72}")
    print("PHASE 2: digit unmask order (16 steps)")
    first_digit_step_f = statistics.mean(
        next(i for i, p in enumerate(s["o_f"]) if p in DIGIT_POS) for s in samples
    )
    first_digit_step_i = statistics.mean(
        next(i for i, p in enumerate(s["o_i"]) if p in DIGIT_POS) for s in samples
    )
    print(f"  first digit unmask at step: FP16 mean={first_digit_step_f:.1f}  INT4 mean={first_digit_step_i:.1f}")
    print("  first digit row:")
    print("    FP16:", dict(Counter(row_of(next(p for p in s["o_f"] if p in DIGIT_POS)) for s in samples)))
    print("    INT4:", dict(Counter(row_of(next(p for p in s["o_i"] if p in DIGIT_POS)) for s in samples)))

    digit_step_agree = []
    for k in range(16):
        agree = sum(1 for s in samples if s["d_f"][k] == s["d_i"][k]) / n
        digit_step_agree.append(agree)
    print("  digit-step agreement (k-th digit unmask):")
    for k, agree in enumerate(digit_step_agree):
        print(f"    k={k:2d}: {agree:5.1%}")

    # Rank shifts for digit positions
    print(f"\n{'=' * 72}")
    print("DIGIT POSITION RANK SHIFTS (mean rank_fp16 - rank_int4)")
    rank_shift = defaultdict(list)
    for s in samples:
        rf, ri = rank_map(s["o_f"]), rank_map(s["o_i"])
        for pos in DIGIT_POS:
            rank_shift[pos].append(rf[pos] - ri[pos])
    for pos in sorted(DIGIT_POS):
        shifts = rank_shift[pos]
        print(
            f"  {POS_LABEL[pos]:5s} (pos {pos:2d}): mean_shift={mean(shifts):+5.2f}  "
            f"INT4 earlier={sum(x>0 for x in shifts)/len(shifts):.1%}"
        )

    # First divergence details
    print(f"\n{'=' * 72}")
    print("FIRST DIVERGENCE")
    swaps = Counter()
    swap_type = Counter()
    for s in samples:
        i = s["prefix"]
        if i >= 24:
            continue
        a, b = s["o_f"][i], s["o_i"][i]
        swaps[(a, b)] += 1
        if a in FMT_POS and b in FMT_POS:
            swap_type["fmt-fmt"] += 1
        elif a in DIGIT_POS and b in DIGIT_POS:
            swap_type["digit-digit"] += 1
        elif a in DIGIT_POS and b in FMT_POS:
            swap_type["fp16_digit_int4_fmt"] += 1
        elif a in FMT_POS and b in DIGIT_POS:
            swap_type["fp16_fmt_int4_digit"] += 1
        else:
            swap_type["other"] += 1
    print(f"  first diff at step: mean={mean([s['prefix'] for s in samples]):.1f} median={statistics.median([s['prefix'] for s in samples]):.0f}")
    print(f"  first diff types: {dict(swap_type)}")
    print("  top swaps at first diff (fp16_pos -> int4_pos):")
    for (a, b), c in swaps.most_common(15):
        print(f"    {POS_LABEL.get(a, str(a)):5s} -> {POS_LABEL.get(b, str(b)):5s}: {c}")

    # fp16_only deep dive
    fp16_only = [s for s in samples if s["outcome"] == "fp16_only"]
    print(f"\n{'=' * 72}")
    print(f"FP16_ONLY DEEP DIVE (n={len(fp16_only)})")
    if fp16_only:
        print(f"  tau_digit mean={mean([s['tau_digit'] for s in fp16_only]):.3f}")
        early = mean(
            sum(1 for a, b in zip(s["d_f"][:8], s["d_i"][:8]) if a == b) / 8 for s in fp16_only
        )
        late = mean(
            sum(1 for a, b in zip(s["d_f"][8:], s["d_i"][8:]) if a == b) / 8 for s in fp16_only
        )
        print(f"  digit order agree first 8: {early:.1%}")
        print(f"  digit order agree last 8:  {late:.1%}")
        print("  examples:")
        for s in fp16_only[:5]:
            i = s["prefix"]
            print(f"\n  #{s['doc_id']} prefix={i} tau_digit={s['tau_digit']:.2f}")
            print(f"    FP16: {' '.join(POS_LABEL.get(p, str(p)) for p in s['o_f'])}")
            print(f"    INT4: {' '.join(POS_LABEL.get(p, str(p)) for p in s['o_i'])}")
            if i < 24:
                print(
                    f"    fork@{i}: FP16={POS_LABEL.get(s['o_f'][i], s['o_f'][i])} "
                    f"INT4={POS_LABEL.get(s['o_i'][i], s['o_i'][i])}"
                )


if __name__ == "__main__":
    main()

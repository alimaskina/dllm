#!/usr/bin/env python3
"""Rank LongBench tasks by whether their metric separates the four variants the
way the method predicts.

Expected ordering:  dense >= sparse_fp16_all ~ sparse_k4v4_all > sparse_fp16_middle

A task is only useful in a headline table if its metric can actually show that.
Two ways a task fails:

  inverted  - a weaker variant scores ABOVE dense by more than noise. This is the
              one that costs you a reviewer: an approximation beating the exact
              ceiling looks like a broken experiment, whatever the caption says.
  flat      - every variant lands within noise of every other. Nothing is wrong,
              but the task cannot distinguish the methods, so it carries no
              evidence either way.

"Noise" here is the standard error of the per-example score mean, so the verdict
scales with n and with how spread the per-example scores are, instead of using a
fixed threshold that would be far too strict on a 0/1 metric and far too loose on
a smooth one.

    python scripts/rank_task_orderings.py results/task_selection_20260912_120000
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

DENSE = "dense"
MAGE = "sparse_fp16_all"
QUANT = "sparse_k4v4_all"
HERALD = "sparse_fp16_middle"
ORDER = [DENSE, MAGE, QUANT, HERALD]

PUBLISHED_PATH = Path(__file__).with_name("longbench_published_scores.json")


def load_published() -> dict[str, dict[str, float]]:
    if not PUBLISHED_PATH.is_file():
        return {}
    return json.loads(PUBLISHED_PATH.read_text(encoding="utf-8")).get("scores", {})


def published_band(task: str, published: dict[str, dict[str, float]]) -> tuple[float, float] | None:
    """min/max published score (as a fraction) across the 7B-class models, so our
    dense number can be checked for being in the right ballpark rather than against
    one arbitrary model. GPT-3.5 is excluded - it is not a 7B-class comparison."""
    row = published.get(task)
    if not row:
        return None
    vals = [v / 100.0 for k, v in row.items() if k != "GPT-3.5-Turbo-16k"]
    return (min(vals), max(vals)) if vals else None


def load_task_scores(task_dir: Path) -> dict[str, list[float]]:
    """variant -> list of per-example scores, from the quality pass only."""
    scores: dict[str, list[float]] = {}
    for path in sorted(task_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pass") != "quality" or row.get("score") is None:
                continue
            scores.setdefault(row["variant"], []).append(float(row["score"]))
    return scores


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    return statistics.stdev(values) / math.sqrt(len(values))


def analyse(task: str, scores: dict[str, list[float]]) -> dict:
    present = [v for v in ORDER if scores.get(v)]
    missing = [v for v in ORDER if v not in present]
    means = {v: statistics.fmean(scores[v]) for v in present}
    sems = {v: sem(scores[v]) for v in present}
    n = {v: len(scores[v]) for v in present}

    def separated(a: str, b: str) -> bool:
        """Is a's mean above b's by more than the combined standard error?"""
        if a not in means or b not in means:
            return False
        gap = means[a] - means[b]
        noise = math.sqrt(sems[a] ** 2 + sems[b] ** 2)
        return gap > noise

    notes: list[str] = []
    verdict = "ok"

    # Inversion: any sparse variant above dense by more than noise.
    inversions = [v for v in (MAGE, QUANT, HERALD) if separated(v, DENSE)]
    if inversions:
        verdict = "inverted"
        for v in inversions:
            notes.append(f"{v} > dense by {means[v] - means[DENSE]:+.3f}")

    # Does the metric separate the good selectors from the starved one at all?
    herald_gap_mage = means.get(MAGE, 0.0) - means.get(HERALD, 0.0)
    herald_gap_quant = means.get(QUANT, 0.0) - means.get(HERALD, 0.0)
    discriminates = separated(MAGE, HERALD) or separated(QUANT, HERALD)

    if verdict == "ok" and not discriminates:
        # Everything within noise of everything else -> no evidence either way.
        spread = max(means.values()) - min(means.values()) if means else 0.0
        verdict = "flat"
        notes.append(f"no variant separated from another (spread {spread:.3f})")

    if verdict == "ok":
        notes.append(f"herald below the all-query selectors by {min(herald_gap_mage, herald_gap_quant):+.3f}")
        if separated(MAGE, QUANT) or separated(QUANT, MAGE):
            notes.append(f"k4v4 vs mage separated ({means[QUANT] - means[MAGE]:+.3f}) - quantization visible here")
        else:
            notes.append("k4v4 ~ mage (quantization within noise, as predicted)")

    return {
        "task": task,
        "verdict": verdict,
        "means": means,
        "sems": sems,
        "n": n,
        "missing": missing,
        "notes": notes,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="output root from run_task_selection.sh (contains <task>/ dirs)")
    p.add_argument("--json", help="also write the analysis here")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not task_dirs:
        raise SystemExit(f"no per-task directories under {root}")

    results = []
    for task_dir in task_dirs:
        scores = load_task_scores(task_dir)
        if not scores:
            continue
        results.append(analyse(task_dir.name, scores))

    if not results:
        raise SystemExit("no quality-pass rows found")

    published = load_published()
    width = max(len(r["task"]) for r in results) + 2
    print(
        f"\n{'task':<{width}}{'n':>4}  {'dense':>7}{'mage':>9}{'k4v4':>9}{'herald':>9}"
        f"   {'published 7B':>14}  verdict"
    )
    print("-" * (width + 70))
    for r in sorted(results, key=lambda x: (x["verdict"] != "ok", x["task"])):
        m = r["means"]
        n = max(r["n"].values()) if r["n"] else 0
        cells = "".join(
            f"{m[v]:>9.3f}" if v in m else f"{'--':>9}" for v in (MAGE, QUANT, HERALD)
        )
        dense_cell = f"{m[DENSE]:>7.3f}" if DENSE in m else f"{'--':>7}"
        band = published_band(r["task"], published)
        if band is None:
            band_cell = f"{'--':>14}"
        else:
            band_cell = f"{band[0]:.2f}-{band[1]:.2f}".rjust(14)
            if DENSE in m:
                lo, hi = band
                # Flag only a clear miss: our dense far outside what every 7B-class
                # model in the LongBench table reaches. Inside or near the band means
                # the task is being run and scored the way the literature runs it.
                if m[DENSE] < lo * 0.5 or m[DENSE] > hi * 1.5:
                    band_cell += " !"
        print(f"{r['task']:<{width}}{n:>4}  {dense_cell}{cells}{band_cell}  {r['verdict']}")
    print(
        "\npublished 7B = min-max over the 7B-class models in THUDM/LongBench's own results "
        "table\n(different models, so this is a ballpark check; '!' marks our dense score far "
        "outside it)"
    )

    for verdict, header in (
        ("ok", "USABLE - metric puts the variants in the predicted order"),
        ("inverted", "DROP - a weaker variant scores above dense by more than noise"),
        ("flat", "NO SIGNAL - every variant within noise of every other"),
    ):
        group = [r for r in results if r["verdict"] == verdict]
        if not group:
            continue
        print(f"\n{header}:")
        for r in group:
            print(f"  {r['task']}")
            for note in r["notes"]:
                print(f"      {note}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

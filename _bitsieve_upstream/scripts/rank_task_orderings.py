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

Every variant runs the SAME examples, so comparisons are paired: the verdict is a
paired t-statistic on the per-example differences, which removes between-example
difficulty variance (that variance dominates - on narrativeqa the unpaired SE of a
variant mean is ~0.05 against ~0.03 for the paired difference). A fixed score
threshold is not used; it would be far too strict on a 0/1 metric and far too loose
on a smooth one.

The report also prints a pooled terseness check, because the k4v4-vs-mage pair is
close to a null (those two differ only by quantizing the cache) and on QA-F1 tasks
a perturbation that merely shortens the answer raises precision, and so the score.

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


def load_task_predictions(task_dir: Path) -> dict[str, dict[str, tuple[float, int]]]:
    """variant -> {example_id: (score, prediction length in words)}."""
    out: dict[str, dict[str, tuple[float, int]]] = {}
    for path in sorted(task_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pass") != "quality" or row.get("score") is None:
                continue
            pred = row.get("prediction")
            if pred is None:
                continue
            out.setdefault(row["variant"], {})[str(row["id"])] = (
                float(row["score"]), len(pred.split())
            )
    return out


def length_deltas(preds: dict[str, dict[str, tuple[float, int]]]) -> tuple[list[int], list[float]]:
    """Does this task's metric pay for terseness? Measured on the k4v4-vs-mage pair.

    Those two variants differ only by quantizing the cache, and empirically leave ~60%
    of answers byte-identical, so any score difference between them is close to a pure
    null: it is not the method doing better, it is the metric reacting to a perturbed
    answer. Correlating that difference with the change in answer LENGTH therefore
    isolates the artifact, with no arbitrary threshold and no confound from a variant
    genuinely losing information.

    Strongly NEGATIVE = shorter answers score higher, so any degradation that trims a
    trailing clause reads as an improvement. That is how an approximation ends up
    "beating" dense on a QA-F1 task, and it is the property that decides whether a task
    can be trusted to rank the variants.

    (Correlating against dense instead would conflate this with the honest direction -
    a sparse variant that drops information answers shorter AND scores worse, giving a
    positive correlation that hides the artifact.)
    """
    a, b = preds.get(QUANT), preds.get(MAGE)
    if not a or not b:
        return [], []
    dl: list[int] = []
    ds: list[float] = []
    for key in set(a) & set(b):
        d_score = a[key][0] - b[key][0]
        if abs(d_score) < 1e-9:
            continue
        ds.append(d_score)
        dl.append(a[key][1] - b[key][1])
    return dl, ds


def report_length_bias(results: list[dict]) -> None:
    """Print the terseness check POOLED over tasks.

    Deliberately not a per-task column: at n=20 a task has only ~7-10 examples where
    k4v4 and mage differ at all, and a single large content flip swings the correlation
    from -0.6 to +0.7. Pooled, it is answering a question about the metric family rather
    than about one task, which is what it is actually evidence for.
    """
    dl: list[int] = []
    ds: list[float] = []
    for r in results:
        dl.extend(r.get("length_deltas", []))
        ds.extend(r.get("score_deltas", []))
    if len(ds) < 8:
        return
    mx, my = statistics.fmean(dl), statistics.fmean(ds)
    num = sum((x - mx) * (y - my) for x, y in zip(dl, ds))
    den = math.sqrt(sum((x - mx) ** 2 for x in dl) * sum((y - my) ** 2 for y in ds))
    corr = num / den if den else float("nan")
    shorter_and_better = sum(1 for x, y in zip(dl, ds) if x < 0 and y > 0)
    better = sum(1 for y in ds if y > 0)
    print(
        f"\nterseness check (pooled, n={len(ds)} examples where k4v4 and mage differ):"
        f"\n  corr(length change, score change) = {corr:+.2f}"
        f"\n  k4v4 scored higher on {better}/{len(ds)}; {shorter_and_better} of those came with a"
        f" SHORTER answer"
        f"\n  k4v4 differs from mage only by quantizing the cache, so this pair is close to a"
        f"\n  null - a negative correlation here means the metric pays for terseness, which is"
        f"\n  how a degraded variant ends up 'beating' a better one on QA-F1."
    )


def load_task_scores(task_dir: Path) -> dict[str, dict[str, float]]:
    """variant -> {example_id: score}, quality pass only.

    Keyed by example rather than appended to a list because every variant runs the
    SAME examples, which makes the variant comparison a paired one - see paired_diff.
    """
    scores: dict[str, dict[str, float]] = {}
    for path in sorted(task_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pass") != "quality" or row.get("score") is None:
                continue
            scores.setdefault(row["variant"], {})[str(row["id"])] = float(row["score"])
    return scores


def load_task_coverage(task_dir: Path) -> dict[str, tuple[float, float, int]]:
    """variant -> (mean coverage mass, mean index overlap, n), coverage pass only.

    Coverage is measured against an independent fp16 all-masked-query reference,
    so it says how much of the attention mass a selector actually kept - which is
    the quantity the method is about. The score is a downstream proxy for it and a
    lossy one: a selector can drop 5% of the mass and still produce the same
    answer. When a task is `flat` in score, coverage is what says whether that is
    because the selectors really are equivalent at this budget or because the
    metric cannot see a difference that is there.

    Absent unless the run was made with COVERAGE=1 (run_task_selection.sh).
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for path in sorted(task_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pass") != "coverage":
                continue
            cov = (row.get("runtime") or {}).get("coverage")
            if not cov or cov.get("mass_mean") is None:
                continue
            out.setdefault(row["variant"], []).append(
                (float(cov["mass_mean"]), float(cov.get("overlap_mean") or "nan"))
            )
    return {
        v: (
            statistics.fmean(m for m, _ in rows),
            statistics.fmean(o for _, o in rows),
            len(rows),
        )
        for v, rows in out.items()
        if rows
    }


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    return statistics.stdev(values) / math.sqrt(len(values))


def paired_diff(a: dict[str, float], b: dict[str, float]) -> tuple[float, float, int]:
    """(mean difference a-b, its standard error, n) over the examples both ran.

    All variants see identical examples, so pairing removes between-example
    difficulty variance - which dominates here. On narrativeqa the unpaired SE of
    each variant's mean is ~0.05 while the SE of the paired difference is ~0.03,
    so the unpaired test would call a real gap 'noise' far more often.
    """
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return 0.0, float("inf"), len(common)
    diffs = [a[k] - b[k] for k in common]
    sd = statistics.stdev(diffs)
    return statistics.fmean(diffs), sd / math.sqrt(len(diffs)), len(diffs)


def n_differing(a: dict[str, float], b: dict[str, float]) -> tuple[int, int]:
    """(examples where the two arms scored differently, examples compared).

    On a discrete metric at a small n, a headline gap can be one flipped
    example: lsht at n=20 showed `mage-herald` -0.050, which was a single
    example of 20 with the other 19 tied. trec at the same n showed +0.250 from
    5 differing examples, all 5 the same way. The means look like results of the
    same kind and are not, so the count belongs next to them.
    """
    common = sorted(set(a) & set(b))
    return sum(1 for k in common if a[k] != b[k]), len(common)


# |t| above this counts as a real separation rather than noise. 2.0 is ~95% for a
# two-sided paired t-test at these n.
T_THRESHOLD = 2.0


def analyse(task: str, scores: dict[str, dict[str, float]]) -> dict:
    present = [v for v in ORDER if scores.get(v)]
    missing = [v for v in ORDER if v not in present]
    means = {v: statistics.fmean(scores[v].values()) for v in present}
    sems = {v: sem(list(scores[v].values())) for v in present}
    n = {v: len(scores[v]) for v in present}

    def compare(a: str, b: str) -> tuple[float, float, float]:
        """(paired mean diff a-b, se, t). t>0 means a scores above b."""
        if a not in scores or b not in scores:
            return 0.0, float("inf"), 0.0
        diff, se, _ = paired_diff(scores[a], scores[b])
        return diff, se, (diff / se if se and math.isfinite(se) else 0.0)

    notes: list[str] = []
    verdict = "ok"

    # Unequal example counts mean at least one variant is still running (or failed
    # partway). Comparing a 20-example mean against a 1-example mean produces
    # confident-looking nonsense, so say so instead of ranking it.
    if missing or (n and max(n.values()) != min(n.values())):
        counts = ", ".join(f"{v}={n.get(v, 0)}" for v in ORDER)
        return {
            "task": task,
            "verdict": "incomplete",
            "means": means,
            "sems": sems,
            "n": n,
            "missing": missing,
            "notes": [f"variants have unequal example counts ({counts})"],
            "paired": {},
        }

    # Inversion: a sparse variant scores above dense by more than noise.
    inversions = []
    for v in (MAGE, QUANT, HERALD):
        diff, _se, t = compare(v, DENSE)
        if t > T_THRESHOLD:
            inversions.append((v, diff, t))
    if inversions:
        verdict = "inverted"
        for v, diff, t in inversions:
            notes.append(f"{v} > dense by {diff:+.3f} (t={t:+.1f})")

    # Can the metric tell the all-query selectors from the single-query one at all?
    mage_gap, _, mage_t = compare(MAGE, HERALD)
    quant_gap, _, quant_t = compare(QUANT, HERALD)
    discriminates = mage_t > T_THRESHOLD or quant_t > T_THRESHOLD

    if verdict == "ok" and not discriminates:
        verdict = "flat"
        notes.append(
            f"herald not separated from the all-query selectors "
            f"(mage-herald {mage_gap:+.3f}, t={mage_t:+.1f}; "
            f"k4v4-herald {quant_gap:+.3f}, t={quant_t:+.1f})"
        )
        # How many examples WOULD resolve it? Useful for deciding whether a task is
        # hopeless or merely under-sampled at this n.
        for label, diff, t in (("mage-herald", mage_gap, mage_t), ("k4v4-herald", quant_gap, quant_t)):
            if t and abs(diff) > 1e-9:
                have = max(n.values())
                need = have * (T_THRESHOLD / abs(t)) ** 2
                notes.append(f"    {label} would need n~{need:.0f} at this effect size")

    if verdict == "ok":
        notes.append(
            f"herald below the all-query selectors (mage {mage_gap:+.3f} t={mage_t:+.1f}, "
            f"k4v4 {quant_gap:+.3f} t={quant_t:+.1f})"
        )
        qm_gap, _, qm_t = compare(QUANT, MAGE)
        if abs(qm_t) > T_THRESHOLD:
            notes.append(f"k4v4 vs mage separated ({qm_gap:+.3f}, t={qm_t:+.1f}) - quantization visible here")
        else:
            notes.append(f"k4v4 ~ mage ({qm_gap:+.3f}, t={qm_t:+.1f}) - quantization within noise, as predicted")

    return {
        "task": task,
        "verdict": verdict,
        "means": means,
        "sems": sems,
        "n": n,
        "missing": missing,
        "notes": notes,
        "paired": {
            "mage_vs_dense": compare(MAGE, DENSE),
            "k4v4_vs_dense": compare(QUANT, DENSE),
            "herald_vs_dense": compare(HERALD, DENSE),
            "mage_vs_herald": compare(MAGE, HERALD),
            "k4v4_vs_herald": compare(QUANT, HERALD),
            "k4v4_vs_mage": compare(QUANT, MAGE),
        },
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
        record = analyse(task_dir.name, scores)
        dl, ds = length_deltas(load_task_predictions(task_dir))
        record["length_deltas"] = dl
        record["score_deltas"] = ds
        record["coverage"] = load_task_coverage(task_dir)
        record["differing"] = {
            label: n_differing(scores.get(x, {}), scores.get(y, {}))
            for label, x, y in (
                ("dense-mage", DENSE, MAGE),
                ("k4v4-mage", QUANT, MAGE),
                ("mage-herald", MAGE, HERALD),
            )
            if scores.get(x) and scores.get(y)
        }
        results.append(record)

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
        counts = r.get("differing") or {}
        if counts:
            # Flagged when a comparison rests on very few examples: on a discrete
            # metric that is where a mean turns into one flipped row.
            parts = []
            for label, (nd, tot) in counts.items():
                mark = " !" if 0 < nd <= 2 else ""
                parts.append(f"{label} {nd}/{tot}{mark}")
            print(f"{'':<{width}}{'':>4}  differing examples: " + ",  ".join(parts))
    covered = [r for r in results if r.get("coverage")]
    if covered:
        print(
            f"\n{'coverage mass vs fp16 reference':<{width}}{'n':>4}  "
            f"{'mage':>9}{'k4v4':>9}{'herald':>9}   what it says"
        )
        print("-" * (width + 70))
        for r in sorted(covered, key=lambda x: x["task"]):
            cov = r["coverage"]
            n = max(v[2] for v in cov.values())
            cells = "".join(
                f"{cov[v][0]:>9.4f}" if v in cov else f"{'--':>9}"
                for v in (MAGE, QUANT, HERALD)
            )
            # The diagnosis a flat score cannot give on its own: is herald keeping
            # the same mass as the all-query selector, or is the metric blind to a
            # gap that is there?
            if MAGE in cov and HERALD in cov:
                gap = cov[MAGE][0] - cov[HERALD][0]
                if gap < 0.01:
                    says = f"herald keeps the same mass (gap {gap:+.4f}) - budget too loose to differ"
                elif r["verdict"] == "flat":
                    says = f"herald drops {gap:.4f} of the mass but the metric cannot see it"
                else:
                    says = f"herald drops {gap:.4f} of the mass"
            else:
                says = ""
            print(f"{r['task']:<{width}}{n:>4}  {cells}   {says}")
        print(
            "\nCoverage is the quantity the method is about; the score is a lossy proxy for it. "
            "\nA task that is flat in score AND flat in coverage is not under-sampled - the "
            "\nselectors are genuinely equivalent at that budget, and no n fixes that."
        )

    print(
        "\n'!' on a differing-examples count marks a comparison resting on <=2 examples - on a "
        "\ndiscrete metric that is a mean built from one or two flipped rows, not an effect."
    )
    print(
        "\npublished 7B = min-max over the 7B-class models in THUDM/LongBench's own results "
        "table\n(different models, so this is a ballpark check; '!' marks our dense score far "
        "outside it)"
    )
    report_length_bias(results)

    for verdict, header in (
        ("ok", "USABLE - metric puts the variants in the predicted order"),
        ("inverted", "DROP - a weaker variant scores above dense by more than noise"),
        ("flat", "NO SIGNAL - herald not separated from the all-query selectors"),
        ("incomplete", "NOT YET COMPARABLE - still running or partially failed"),
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

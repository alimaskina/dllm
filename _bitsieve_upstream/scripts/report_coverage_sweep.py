#!/usr/bin/env python3
"""Report a precision-vs-budget coverage sweep.

The question the sweep asks is not "how much does quantization cost" but "at a
fixed cache budget, is it better to keep more entries at lower precision". So the
table is ordered by memory, and the iso-memory groups underneath it are where the
answer actually is: within a group every arm costs the same bits, and the one
retaining the most attention mass wins.

`mass_abs` is the share of the full prefix attention mass an arm's selection
retains, so it is comparable across arms with different budgets. The relative
`mass` is not - it normalises by each arm's own budget and so divides out the
trade being measured; it is shown only as a selector-quality diagnostic.

    python scripts/report_coverage_sweep.py results/coverage_sweep
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys


def paired(a: list[float], b: list[float]) -> tuple[float, float, int, int]:
    """(mean difference, t, wins for a, wins for b) over paired examples."""
    diffs = [x - y for x, y in zip(a, b)]
    md = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    t = md / (sd / math.sqrt(len(diffs))) if sd else float("nan")
    return md, t, sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d < 0)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/coverage_sweep")
    files = sorted(root.glob("*.jsonl"))
    if not files:
        print(f"no sweep files under {root}")
        return 1

    for path in files:
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        benchmark = rows[0]["benchmark"]
        names = list(rows[0]["arms"])
        # Per-arm series, paired by example.
        series = {n: [r["arms"][n]["mass_abs_mean"] for r in rows if n in r["arms"]] for n in names}
        bits = {n: rows[0]["arms"][n]["bits"] for n in names}
        ks = {n: statistics.fmean(r["arms"][n]["mean_selected_k"] for r in rows) for n in names}
        # Memory comes from each arm's NOMINAL budget, not the entries it actually
        # selected. The realized count is clamped to the live prefix (a 4x arm on a
        # 114-token prompt gets 124, not 128), which would jitter the ratio and stop
        # arms that cost exactly the same from grouping together.
        nominal = {
            str(a["name"]): float(a["topk"] if a.get("topk") is not None else a["topk_percent"])
            for a in rows[0]["config"]["coverage_arms"]
        }
        base_budget = nominal.get("fp16", float("nan"))
        mem = {n: (bits[n] * nominal[n]) / (16.0 * base_budget) for n in names}

        n_ex = len(rows)
        print(f"\n=== {benchmark}  (n={n_ex}, "
              f"mean prompt {statistics.fmean(r['prompt_tokens'] for r in rows):.0f} tokens) ===")
        print(f"  {'arm':<10}{'bits':>5}{'entries':>9}{'memory':>8}"
              f"{'mass_abs':>10}{'ceiling':>9}{'overlap':>9}")
        for n in sorted(names, key=lambda x: (-mem[x], -bits[x])):
            a = [r["arms"][n] for r in rows]
            print(f"  {n:<10}{bits[n]:>5}{ks[n]:>9.0f}{mem[n]:>8.3f}"
                  f"{statistics.fmean(series[n]):>10.4f}"
                  f"{statistics.fmean(x['ceiling_abs_mean'] for x in a):>9.4f}"
                  f"{statistics.fmean(x['overlap_mean'] for x in a):>9.4f}")

        # The comparison the sweep exists for: equal bits, different split
        # between precision and count.
        groups: dict[float, list[str]] = {}
        for n in names:
            groups.setdefault(round(mem[n], 6), []).append(n)
        printed = False
        for lvl in sorted((g for g, v in groups.items() if len(v) > 1), reverse=True):
            arms = sorted(groups[lvl], key=lambda x: -bits[x])
            if not printed:
                print("\n  same memory, different precision/count split:")
                printed = True
            best = max(arms, key=lambda x: statistics.fmean(series[x]))
            cells = ",  ".join(f"{a} {statistics.fmean(series[a]):.4f}" for a in arms)
            print(f"    memory {lvl:.3f}:  {cells}    -> {best} keeps more")
            if len(arms) == 2:
                lo, hi = arms[0], arms[1]     # lo = more bits/fewer entries
                md, t, w, l = paired(series[hi], series[lo])
                print(f"{'':>17}{hi} - {lo} = {md:+.4f}  t={t:+.1f}  "
                      f"wins {w}:{l} of {len(series[hi])}")
        if not printed:
            print("\n  (no two arms share a memory budget - nothing to compare)")

    print(
        "\nmass_abs = share of the full prefix attention mass the selection keeps;"
        "\n  comparable across arms. ceiling = the most any selector could keep at that"
        "\n  budget. overlap = agreement with the fp16 reference top-k."
        "\n\nMemory is arithmetic (bits x entries), not measured: 3-bit arms are scored"
        "\n  through a round-tripped quantizer because no packed kernel addresses 3 bits."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Per-context-length report for LongBench-E runs.

LongBench-E resamples each task so 0-4k / 4-8k / 8k+ contexts are evenly represented
and is scored per bucket. That cut is the point of running it here: the claim behind a
KV-cache selector is that dropping entries costs more as the prefix grows, and a pooled
average cannot show that - a selector could be free at 4k and ruinous at 8k+ and still
land on the same mean as one that is mildly bad everywhere.

    python scripts/report_longbench_e.py results/longbench_e
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

DENSE, MAGE, QUANT, HERALD = (
    "dense", "sparse_fp16_all", "sparse_k4v4_all", "sparse_fp16_middle"
)
ORDER = [DENSE, MAGE, QUANT, HERALD]
SHORT = {DENSE: "dense", MAGE: "mage", QUANT: "k4v4", HERALD: "herald"}
BUCKETS = ["0-4k", "4-8k", "8k+"]


def bucket(length: int | None) -> str | None:
    if length is None:
        return None
    if length < 4000:
        return "0-4k"
    if length < 8000:
        return "4-8k"
    return "8k+"


def load(task_dir: Path) -> dict[str, dict[str, tuple[float, str | None]]]:
    """variant -> {example_id: (score, bucket)} for the quality pass."""
    out: dict[str, dict[str, tuple[float, str | None]]] = {}
    for path in sorted(task_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pass") != "quality" or row.get("score") is None:
                continue
            meta = row.get("metadata") or {}
            out.setdefault(row["variant"], {})[str(row["id"])] = (
                float(row["score"]), bucket(meta.get("length"))
            )
    return out


def paired(a: dict[str, tuple[float, str | None]], b: dict[str, tuple[float, str | None]],
           keep: str | None) -> tuple[float, float, int]:
    keys = [k for k in set(a) & set(b) if keep is None or a[k][1] == keep]
    if len(keys) < 2:
        return float("nan"), float("nan"), len(keys)
    diffs = [a[k][0] - b[k][0] for k in keys]
    md = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(len(diffs)) if sd else 0.0
    return md, (md / se if se else 0.0), len(diffs)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/longbench_e")
    task_dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if not task_dirs:
        print(f"no task directories under {root}")
        return 1

    for task_dir in task_dirs:
        data = load(task_dir)
        if not data:
            continue
        print(f"\n=== {task_dir.name} ===")
        print(f"{'bucket':<8}{'n':>4}" + "".join(f"{SHORT[v]:>9}" for v in ORDER)
              + f"{'mage-herald':>14}{'dense-mage':>13}")
        for b in BUCKETS + [None]:
            label = b or "ALL"
            per = {}
            for v in ORDER:
                rows = data.get(v, {})
                vals = [s for s, bk in rows.values() if b is None or bk == b]
                per[v] = statistics.fmean(vals) if vals else float("nan")
            n = sum(1 for s, bk in data.get(DENSE, {}).values() if b is None or bk == b)
            mh, mh_t, _ = paired(data.get(MAGE, {}), data.get(HERALD, {}), b)
            dm, dm_t, _ = paired(data.get(DENSE, {}), data.get(MAGE, {}), b)
            cells = "".join(
                ("       --" if per[v] != per[v] else f"{per[v]:>9.3f}") for v in ORDER
            )
            f = lambda d, t: "          --" if d != d else f"{d:>+7.3f} t={t:>+4.1f}"
            print(f"{label:<8}{n:>4}{cells}  {f(mh, mh_t):>12}{f(dm, dm_t):>12}")
    print(
        "\nThe question this table answers: does the gap to herald WIDEN with context length?"
        "\nIf it does not, the selector's advantage is not a long-context effect and the"
        "\ntask is carrying the claim on something else."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bitsieve_fastdllm.eval.benchmarks import LONGBENCH_E_BUCKETS, length_bucket

DENSE, MAGE, QUANT, HERALD = (
    "dense", "sparse_fp16_all", "sparse_k4v4_all", "sparse_fp16_middle"
)
ORDER = [DENSE, MAGE, QUANT, HERALD]
SHORT = {DENSE: "dense", MAGE: "mage", QUANT: "k4v4", HERALD: "herald"}
# Imported rather than restated: the loader balances the sample against these
# same edges, and a copy here could drift from them without anything failing.
BUCKETS = [name for _, _, name in LONGBENCH_E_BUCKETS]


def bucket(length: int | None) -> str | None:
    return length_bucket(length) if isinstance(length, int) else None


def load(task_dir: Path) -> dict[str, dict[str, tuple[float, str | None, int]]]:
    """variant -> {example_id: (score, bucket, prompt_tokens)} for the quality pass."""
    out: dict[str, dict[str, tuple[float, str | None, int]]] = {}
    for path in sorted(task_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pass") != "quality" or row.get("score") is None:
                continue
            meta = row.get("metadata") or {}
            out.setdefault(row["variant"], {})[str(row["id"])] = (
                float(row["score"]),
                bucket(meta.get("length")),
                int(row.get("prompt_tokens") or 0),
            )
    return out


def paired(a: dict[str, tuple[float, str | None, int]],
           b: dict[str, tuple[float, str | None, int]],
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
        print(
            f"{'bucket':<8}{'n':>4}{'tok':>7}"
            + "".join(f"{SHORT[v]:>9}" for v in ORDER)
            + f"{'dense-mage':>14}{'k4v4-mage':>14}{'mage-herald':>14}"
        )
        for b in BUCKETS + [None]:
            label = b or "ALL"
            per = {}
            for v in ORDER:
                rows = data.get(v, {})
                vals = [s for s, bk, _ in rows.values() if b is None or bk == b]
                per[v] = statistics.fmean(vals) if vals else float("nan")
            dense_rows = [
                (s, bk, tok) for s, bk, tok in data.get(DENSE, {}).values()
                if b is None or bk == b
            ]
            n = len(dense_rows)
            # Mean prompt length actually fed to the model. A percentage budget
            # is a very different absolute budget at 3k than at 20k, so this is
            # the column that explains a gap changing across buckets.
            toks = statistics.fmean([tok for _, _, tok in dense_rows]) if dense_rows else 0
            dm, dm_t, _ = paired(data.get(DENSE, {}), data.get(MAGE, {}), b)
            qm, qm_t, _ = paired(data.get(QUANT, {}), data.get(MAGE, {}), b)
            mh, mh_t, _ = paired(data.get(MAGE, {}), data.get(HERALD, {}), b)
            cells = "".join(
                ("       --" if per[v] != per[v] else f"{per[v]:>9.3f}") for v in ORDER
            )
            f = lambda d, t: "            --" if d != d else f"{d:>+8.3f} t={t:>+4.1f}"
            print(
                f"{label:<8}{n:>4}{toks:>7.0f}{cells}"
                f"{f(dm, dm_t)}{f(qm, qm_t)}{f(mh, mh_t)}"
            )
    print(
        "\nWhat to read here:"
        "\n  dense-mage  : the price of sparsity. Should be small and positive."
        "\n  k4v4-mage   : the price of quantisation on top. Should be ~0 either way."
        "\n  mage-herald : the value of the all-query selector over a single middle query."
        "\n                Should be positive, and the open question is whether it WIDENS"
        "\n                with context length. If it does not, the selector's advantage is"
        "\n                not a long-context effect and the task carries the claim on"
        "\n                something else. Read it against the tok column: a percentage"
        "\n                budget is far tighter in absolute terms in the 0-4k bucket."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

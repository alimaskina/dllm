#!/usr/bin/env python3
"""Unified report for handoff sweep (LongBench + benchmarks, one results.jsonl)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from handoff_sweep_configs import HANDOFF_TOPK_PCTS  # noqa: E402
from run_longbench_sweep import summarize_sweep, write_sweep_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    jsonl_path = out_dir / "results.jsonl"
    meta_path = out_dir / "sweep_meta.json"

    if not jsonl_path.exists():
        raise SystemExit(f"No results at {jsonl_path}")

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    analysis = summarize_sweep(records)
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    topk_pcts = HANDOFF_TOPK_PCTS
    model = "?"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        topk_pcts = tuple(meta.get("topk_pcts") or HANDOFF_TOPK_PCTS)
        model = meta.get("model", "?")

    title = f"Handoff Sweep Report ({model})"
    write_sweep_report(
        analysis,
        out_dir / "report.md",
        (),
        topk_pcts,
        title=title,
        handoff=True,
    )
    n = len(records)
    print(f"Records: {n}")
    print(f"Report → {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

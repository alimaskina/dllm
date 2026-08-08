#!/usr/bin/env python3
"""Compare stabilization across bd_size runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_stabilization import analyze_record, aggregate, format_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", dest="sweep_root", default="checkpoints/sweep_bd_n64_sbeq")
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    root = Path(args.sweep_root)
    summaries = []
    lines = ["# Stabilization comparison (sbeq sweep)\n"]

    for d in sorted(root.glob("n*_bd*_sbeq")):
        traces = d / "traces.jsonl"
        if not traces.exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        rows = []
        for line in traces.open():
            rows.extend(analyze_record(json.loads(line)))
        s = aggregate(rows)
        bd = meta["bd_size"]
        summaries.append((bd, s, meta))
        lines.append(f"## bd_size={bd}\n")
        lines.append(format_report(s, meta))

    lines.append("\n## Cross-bd summary\n")
    lines.append("| bd | mean stab step | median | frac step 0 | frac stab before unmask | mean gap | never stab |")
    lines.append("|---:|---------------:|-------:|------------:|------------------------:|---------:|-----------:|")
    for bd, s, _ in summaries:
        lines.append(
            f"| {bd} | {s.get('mean_stabilize_step', 0):.2f} | {s.get('median_stabilize_step', '-')} | "
            f"{s.get('frac_stable_from_first', 0):.1%} | {s.get('frac_stabilized_before_unmask', 0):.1%} | "
            f"{s.get('mean_gap_stabilize_to_unmask', 0):.2f} | {s.get('frac_never_stabilized', 0):.1%} |"
        )

    out = Path(args.out_md or root / "stabilization_compare.md")
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare volatility sweeps across bd_size values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def format_compare_table(summaries: list[tuple[int, dict]]) -> str:
    lines = ["# Block size sweep comparison\n"]
    lines.append("| bd_size | accuracy | mean vol (blocks 0-5) | mean vol (all blocks) | max block |")
    lines.append("|--------:|---------:|------------------------:|------------------------:|----------:|")

    for bd_size, s in summaries:
        by_block = s.get("by_block", [])
        early = [b["mean_frac_changed_vs_first"] for b in by_block if b["block_idx"] <= 5]
        all_blocks = [b["mean_frac_changed_vs_first"] for b in by_block]
        early_mean = sum(early) / len(early) if early else 0.0
        all_mean = sum(all_blocks) / len(all_blocks) if all_blocks else 0.0
        max_block = max((b["block_idx"] for b in by_block), default=-1)
        lines.append(
            f"| {bd_size} | {s.get('accuracy', 0):.1%} | {early_mean:.3f} | {all_mean:.3f} | {max_block} |"
        )

    lines.append("\n## Per-block volatility by bd_size\n")
    lines.append(
        "Fraction of masked positions whose argmax prediction changed vs the first pass.\n"
    )
    max_block = max(
        (b["block_idx"] for _, s in summaries for b in s.get("by_block", [])),
        default=-1,
    )
    header = "| block | " + " | ".join(str(bd) for bd, _ in summaries) + " |"
    sep = "|------:|" + "|".join(["--------:"] * len(summaries)) + "|"
    lines.append(header)
    lines.append(sep)

    for block_idx in range(max_block + 1):
        row = [f"| {block_idx} |"]
        for _, s in summaries:
            match = next(
                (b for b in s.get("by_block", []) if b["block_idx"] == block_idx), None
            )
            if match is None:
                row.append(" — |")
            else:
                n = match["n_samples_with_block"]
                val = match["mean_frac_changed_vs_first"]
                row.append(f" {val:.3f} (n={n}) |")
        lines.append("".join(row))

    for bd_size, s in summaries:
        lines.append(f"\n## bd_size={bd_size}: all blocks (n={s.get('n_samples', '?')})\n")
        lines.append(
            "| block | samples | frac changed | frac committed ≠ 1st | mean flips/pos | inner steps |"
        )
        lines.append("|------:|--------:|-------------:|---------------------:|---------------:|------------:|")
        for b in s.get("by_block", []):
            lines.append(
                f"| {b['block_idx']} | {b['n_samples_with_block']} | "
                f"{b['mean_frac_changed_vs_first']:.3f} | "
                f"{b['mean_frac_committed_differs']:.3f} | "
                f"{b['mean_pred_changes_per_pos']:.3f} | "
                f"{b['mean_inner_steps']:.1f} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default="checkpoints/sweep_bd")
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    root = Path(args.sweep_root)
    summaries: list[tuple[int, dict]] = []
    for d in sorted(root.glob("n*_bd*")):
        summary_path = d / "volatility_summary.json"
        traces_path = d / "traces.jsonl"
        if summary_path.exists():
            s = load_summary(summary_path)
        elif traces_path.exists():
            from analyze_volatility import aggregate

            records = [json.loads(line) for line in traces_path.open()]
            if not records:
                continue
            s = aggregate(records)
        else:
            continue
        bd = s.get("bd_size")
        if bd is None:
            meta = json.loads((d / "meta.json").read_text())
            bd = meta["bd_size"]
        summaries.append((bd, s))

    summaries.sort(key=lambda x: x[0])
    md = format_compare_table(summaries)
    out_md = Path(args.out_md or root / "compare_report.md")
    out_md.write_text(md)
    print(out_md)
    print(md)


if __name__ == "__main__":
    main()

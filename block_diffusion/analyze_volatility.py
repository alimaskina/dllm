#!/usr/bin/env python3
"""Aggregate block-level token prediction volatility from run_volatility.py traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_traces(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def aggregate(records: list[dict]) -> dict:
    by_block: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        for block in rec["trace"]["blocks"]:
            by_block[block["block_idx"]].append(block)

    block_stats = []
    for block_idx in sorted(by_block.keys()):
        blocks = by_block[block_idx]
        frac_changed = [b["frac_changed_vs_first"] for b in blocks]
        frac_committed = [b["frac_committed_differs"] for b in blocks]
        mean_changes = [b["mean_pred_changes"] for b in blocks]
        n_positions = [b["n_positions"] for b in blocks]
        inner_steps = [b["inner_steps"] for b in blocks]

        block_stats.append(
            {
                "block_idx": block_idx,
                "n_samples_with_block": len(blocks),
                "mean_frac_changed_vs_first": float(np.mean(frac_changed)),
                "std_frac_changed_vs_first": float(np.std(frac_changed)),
                "mean_frac_committed_differs": float(np.mean(frac_committed)),
                "mean_pred_changes_per_pos": float(np.mean(mean_changes)),
                "mean_positions_in_block": float(np.mean(n_positions)),
                "mean_inner_steps": float(np.mean(inner_steps)),
            }
        )

    pos_in_block_buckets: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        for block in rec["trace"]["blocks"]:
            for pos in block["positions"]:
                if pos["n_forward_passes"] <= 1:
                    continue
                pos_in_block_buckets[pos["pos_in_block"]].append(
                    1.0 if pos["changed_vs_first"] else 0.0
                )

    pos_stats = []
    for pos_in_block in sorted(pos_in_block_buckets.keys()):
        vals = pos_in_block_buckets[pos_in_block]
        pos_stats.append(
            {
                "pos_in_block": pos_in_block,
                "n_observations": len(vals),
                "frac_changed_vs_first": float(np.mean(vals)),
            }
        )

    accuracy = float(np.mean([r["correct"] for r in records])) if records else 0.0

    return {
        "n_samples": len(records),
        "accuracy": accuracy,
        "by_block": block_stats,
        "by_pos_in_block": pos_stats,
    }


def format_markdown(summary: dict, meta: dict | None) -> str:
    lines = ["# Block diffusion token volatility (GSM8K)\n"]
    if meta:
        lines.append(
            f"Model: `{meta.get('model_path')}` | "
            f"n={summary['n_samples']} | "
            f"accuracy={summary['accuracy']:.1%} | "
            f"threshold={meta.get('threshold')} | "
            f"bd_size={meta.get('bd_size')}\n"
        )
    lines.append("## Volatility by generation block\n")
    lines.append(
        "| block | samples | frac changed vs 1st pass | frac committed ≠ 1st | mean flips/pos | mean inner steps |"
    )
    lines.append("|------:|--------:|-------------------------:|---------------------:|---------------:|-----------------:|")
    for row in summary["by_block"]:
        lines.append(
            f"| {row['block_idx']} | {row['n_samples_with_block']} | "
            f"{row['mean_frac_changed_vs_first']:.3f} | "
            f"{row['mean_frac_committed_differs']:.3f} | "
            f"{row['mean_pred_changes_per_pos']:.3f} | "
            f"{row['mean_inner_steps']:.1f} |"
        )
    lines.append("\n## Volatility by position inside block\n")
    lines.append("| pos_in_block | observations | frac changed vs 1st pass |")
    lines.append("|-------------:|-------------:|-------------------------:|")
    for row in summary["by_pos_in_block"]:
        lines.append(
            f"| {row['pos_in_block']} | {row['n_observations']} | {row['frac_changed_vs_first']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces",
        default="checkpoints/gsm8k_volatility_n32/traces.jsonl",
        help="Path to traces.jsonl from run_volatility.py",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    traces_path = Path(args.traces)
    records = load_traces(traces_path)
    summary = aggregate(records)

    meta_path = traces_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None

    out_json = Path(args.out_json or traces_path.parent / "volatility_summary.json")
    out_md = Path(args.out_md or traces_path.parent / "volatility_report.md")

    out_json.write_text(json.dumps(summary, indent=2))
    out_md.write_text(format_markdown(summary, meta))
    print(out_json)
    print(out_md)
    print(format_markdown(summary, meta))


if __name__ == "__main__":
    main()

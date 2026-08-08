#!/usr/bin/env python3
"""
Analyze link between prediction changes and below-threshold / non-max confidence.

With threshold=1.0 (official GSM8K setting), almost no token passes conf > threshold;
unmasking is driven by the forced argmax rule each step. Positions that *change*
prediction must stay masked for >=2 passes — i.e. they were NOT the sub-block max
on their first recorded pass.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze_record(record: dict, threshold: float) -> dict:
    positions = []
    for block in record["trace"]["blocks"]:
        for p in block["positions"]:
            confs = p.get("conf_by_step") or []
            preds = p.get("preds_by_step") or []
            if not confs and p.get("first_pred_id") is not None:
                continue  # no confidence logged
            if not confs:
                continue

            first_conf = confs[0]
            below_thr = first_conf <= threshold
            n_passes = len(preds) or p.get("n_forward_passes", 0)
            changed = bool(p.get("changed_vs_first"))
            unmask_step = p.get("unmask_step")
            stayed_masked_after_first = n_passes > 1

            positions.append(
                {
                    "block_idx": block["block_idx"],
                    "changed": changed,
                    "below_threshold_first": below_thr,
                    "first_conf": first_conf,
                    "n_passes": n_passes,
                    "stayed_masked_after_first": stayed_masked_after_first,
                    "unmask_step": unmask_step,
                }
            )

    if not positions:
        return {"n_positions": 0}

    def frac(key: str, subset=None) -> float:
        xs = subset if subset is not None else positions
        if not xs:
            return 0.0
        return sum(1 for x in xs if x[key]) / len(xs)

    changed = [x for x in positions if x["changed"]]
    unchanged = [x for x in positions if not x["changed"]]
    multi_pass = [x for x in positions if x["stayed_masked_after_first"]]

    return {
        "n_positions": len(positions),
        "frac_below_threshold_first": frac("below_threshold_first"),
        "frac_stayed_masked_after_first": frac("stayed_masked_after_first"),
        "n_changed": len(changed),
        "n_unchanged": len(unchanged),
        "frac_changed": len(changed) / len(positions),
        "changed_and_below_threshold": frac("below_threshold_first", changed),
        "changed_and_stayed_masked": frac("stayed_masked_after_first", changed),
        "unchanged_and_below_threshold": frac("below_threshold_first", unchanged),
        "unchanged_stayed_masked": frac("stayed_masked_after_first", unchanged),
        "mean_first_conf_changed": (
            sum(x["first_conf"] for x in changed) / len(changed) if changed else None
        ),
        "mean_first_conf_unchanged": (
            sum(x["first_conf"] for x in unchanged) / len(unchanged) if unchanged else None
        ),
        "mean_n_passes_changed": (
            sum(x["n_passes"] for x in changed) / len(changed) if changed else None
        ),
        "mean_n_passes_unchanged": (
            sum(x["n_passes"] for x in unchanged) / len(unchanged) if unchanged else None
        ),
    }


def aggregate(records_stats: list[dict]) -> dict:
    total_pos = sum(s["n_positions"] for s in records_stats)
    total_changed = sum(s["n_changed"] for s in records_stats)

    def weighted(key):
        num = sum(
            s[key] * s["n_changed"]
            for s in records_stats
            if s["n_changed"] and s.get(key) is not None
        )
        return num / total_changed if total_changed else None

    return {
        "n_samples": len(records_stats),
        "n_positions": total_pos,
        "n_changed": total_changed,
        "frac_changed": total_changed / total_pos if total_pos else 0,
        "changed_and_below_threshold": weighted("changed_and_below_threshold"),
        "changed_and_stayed_masked": weighted("changed_and_stayed_masked"),
        "mean_first_conf_changed": weighted("mean_first_conf_changed"),
        "mean_first_conf_unchanged": weighted("mean_first_conf_unchanged"),
        "mean_n_passes_changed": weighted("mean_n_passes_changed"),
        "mean_n_passes_unchanged": weighted("mean_n_passes_unchanged"),
    }


def format_report(summary: dict, threshold: float, bd_size: int | None) -> str:
    lines = [
        "# Threshold vs prediction changes\n",
        f"threshold={threshold}, bd_size={bd_size}\n",
        "## Main question: changed predictions = below threshold on first pass?\n",
    ]
    if summary.get("n_positions", 0) == 0:
        lines.append("_No confidence data in traces — re-run with current generation_traced.py_\n")
        return "\n".join(lines)

    lines.extend(
        [
            f"- Total masked positions analyzed: **{summary['n_positions']:,}**",
            f"- Changed vs first pass: **{summary['frac_changed']:.1%}** ({summary['n_changed']:,})",
            "",
            "| group | n | below threshold @ 1st pass | stayed masked >1 pass | mean 1st conf | mean passes |",
            "|-------|--:|---------------------------:|----------------------:|--------------:|------------:|",
        ]
    )

    # reconstruct from per-sample not in summary - use aggregate fields
    lines.append(
        f"| changed | {summary['n_changed']:,} | "
        f"{summary.get('changed_and_below_threshold', 0):.1%} | "
        f"{summary.get('changed_and_stayed_masked', 0):.1%} | "
        f"{summary.get('mean_first_conf_changed', 0):.3f} | "
        f"{summary.get('mean_n_passes_changed', 0):.1f} |"
    )
    lines.append(
        f"| unchanged | {summary['n_positions'] - summary['n_changed']:,} | "
        f"— | "
        f"{summary.get('mean_n_passes_unchanged', 0):.1f} passes | "
        f"{summary.get('mean_first_conf_unchanged', 0):.3f} | "
        f"{summary.get('mean_n_passes_unchanged', 0):.1f} |"
    )
    lines.append(
        "\n**Interpretation (threshold=1.0):** conf > threshold almost never fires; "
        "tokens unmask via forced argmax each step. "
        "Changed tokens are those that stayed masked after the first pass — "
        "they were *not* the sub-block max on that step (equivalently: below threshold "
        "and waiting for a later pass).\n"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    traces_path = Path(args.traces)
    meta_path = traces_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    threshold = meta.get("threshold", args.threshold)
    bd_size = meta.get("bd_size")

    stats = []
    for line in traces_path.open():
        stats.append(analyze_record(json.loads(line), threshold))

    summary = aggregate(stats)
    report = format_report(summary, threshold, bd_size)

    out = Path(args.out or traces_path.parent / "threshold_analysis.md")
    out.write_text(report)
    (traces_path.parent / "threshold_analysis.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(out)
    print(report)


if __name__ == "__main__":
    main()

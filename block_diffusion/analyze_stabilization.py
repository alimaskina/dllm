#!/usr/bin/env python3
"""Per-token prediction stabilization step analysis.

For each masked position, find the first inner step where argmax prediction
equals the final committed token and stays unchanged until unmask (or trace end).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def stabilize_step(preds: list[int], final_pred: int | None) -> int | None:
    """First step where pred == final_pred for all remaining masked passes."""
    if not preds or final_pred is None:
        return None
    for s in range(len(preds)):
        if preds[s] != final_pred:
            continue
        if all(preds[t] == final_pred for t in range(s, len(preds))):
            return s
    return None


def analyze_position(p: dict) -> dict | None:
    preds = list(p.get("preds_by_step") or [])
    if not preds:
        return None
    final_pred = p.get("final_pred_id")
    if final_pred is None:
        return None

    stab = stabilize_step(preds, final_pred)
    unmask = p.get("unmask_step")
    n_passes = len(preds)

    gap = None
    if stab is not None and unmask is not None:
        gap = unmask - stab

    return {
        "block_idx": None,  # filled by caller
        "pos_in_block": p["pos_in_block"],
        "stabilize_step": stab,
        "unmask_step": unmask,
        "n_passes": n_passes,
        "gap_stabilize_to_unmask": gap,
        "stabilized_before_unmask": gap is not None and gap > 0,
        "never_stabilized": stab is None,
        "stable_from_first": stab == 0,
        "final_differs_from_first": bool(p.get("committed_differs_from_first")),
    }


def analyze_record(record: dict) -> list[dict]:
    out = []
    for block in record["trace"]["blocks"]:
        for p in block["positions"]:
            row = analyze_position(p)
            if row is None:
                continue
            row["block_idx"] = block["block_idx"]
            out.append(row)
    return out


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n_positions": 0}

    stab_steps = [r["stabilize_step"] for r in rows if r["stabilize_step"] is not None]
    gaps = [r["gap_stabilize_to_unmask"] for r in rows if r["gap_stabilize_to_unmask"] is not None]
    gaps_pos = [g for g in gaps if g > 0]

    by_block: dict[int, list] = defaultdict(list)
    by_pos: dict[int, list] = defaultdict(list)
    for r in rows:
        by_block[r["block_idx"]].append(r)
        by_pos[r["pos_in_block"]].append(r)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    block_stats = []
    for bi in sorted(by_block.keys()):
        xs = by_block[bi]
        stabs = [x["stabilize_step"] for x in xs if x["stabilize_step"] is not None]
        gps = [x["gap_stabilize_to_unmask"] for x in xs if x["gap_stabilize_to_unmask"] is not None]
        block_stats.append(
            {
                "block_idx": bi,
                "n": len(xs),
                "mean_stabilize_step": mean(stabs),
                "mean_gap_to_unmask": mean(gps),
                "frac_stabilized_before_unmask": sum(x["stabilized_before_unmask"] for x in xs) / len(xs),
                "frac_never_stabilized": sum(x["never_stabilized"] for x in xs) / len(xs),
                "frac_stable_from_first": sum(x["stable_from_first"] for x in xs) / len(xs),
            }
        )

    pos_stats = []
    for pi in sorted(by_pos.keys()):
        xs = by_pos[pi]
        stabs = [x["stabilize_step"] for x in xs if x["stabilize_step"] is not None]
        pos_stats.append(
            {
                "pos_in_block": pi,
                "n": len(xs),
                "mean_stabilize_step": mean(stabs),
                "frac_stabilized_before_unmask": sum(x["stabilized_before_unmask"] for x in xs) / len(xs),
            }
        )

    # histogram of stabilize_step (0..31 cap)
    hist = defaultdict(int)
    for s in stab_steps:
        hist[s] += 1

    return {
        "n_positions": len(rows),
        "mean_stabilize_step": mean(stab_steps),
        "median_stabilize_step": sorted(stab_steps)[len(stab_steps) // 2] if stab_steps else None,
        "mean_gap_stabilize_to_unmask": mean(gaps),
        "mean_gap_when_positive": mean(gaps_pos),
        "frac_stabilized_before_unmask": sum(r["stabilized_before_unmask"] for r in rows) / len(rows),
        "frac_never_stabilized": sum(r["never_stabilized"] for r in rows) / len(rows),
        "frac_stable_from_first": sum(r["stable_from_first"] for r in rows) / len(rows),
        "stabilize_step_hist": {str(k): hist[k] for k in sorted(hist.keys())},
        "by_block": block_stats,
        "by_pos_in_block": pos_stats,
    }


def format_report(summary: dict, meta: dict) -> str:
    lines = [
        "# Token stabilization step analysis\n",
        f"bd_size={meta.get('bd_size')}  n_samples={meta.get('n_samples')}  "
        f"small_block={meta.get('small_block_size')}\n",
        "For each masked position: first inner step where argmax == final token "
        "and stays unchanged until unmask.\n",
    ]
    if summary.get("n_positions", 0) == 0:
        lines.append("_No preds_by_step in traces._\n")
        return "\n".join(lines)

    lines.extend(
        [
            "## Global\n",
            f"- Positions analyzed: **{summary['n_positions']:,}**",
            f"- Mean stabilize step: **{summary['mean_stabilize_step']:.2f}**",
            f"- Median stabilize step: **{summary['median_stabilize_step']}**",
            f"- Stable from step 0: **{summary['frac_stable_from_first']:.1%}**",
            f"- Stabilized before unmask (gap ≥ 1): **{summary['frac_stabilized_before_unmask']:.1%}**",
            f"- Mean gap stabilize→unmask: **{summary['mean_gap_stabilize_to_unmask']:.2f}** steps",
            f"- Mean gap when >0: **{summary.get('mean_gap_when_positive') or 0:.2f}** steps",
            f"- Never stabilized (pred kept flipping): **{summary['frac_never_stabilized']:.1%}**",
            "",
            "## Stabilize step histogram\n",
            "| step | count |",
            "|-----:|------:|",
        ]
    )
    total = sum(summary["stabilize_step_hist"].values())
    for k, v in summary["stabilize_step_hist"].items():
        lines.append(f"| {k} | {v} ({100*v/total:.1f}%) |")

    lines.extend(
        [
            "",
            "## By generation block\n",
            "| block | n | mean stab step | mean gap→unmask | frac stab before unmask | frac never stab |",
            "|------:|--:|---------------:|----------------:|------------------------:|----------------:|",
        ]
    )
    for b in summary["by_block"]:
        lines.append(
            f"| {b['block_idx']} | {b['n']} | {b['mean_stabilize_step']:.2f} | "
            f"{b['mean_gap_to_unmask']:.2f} | {b['frac_stabilized_before_unmask']:.1%} | "
            f"{b['frac_never_stabilized']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## By position inside block\n",
            "| pos | n | mean stab step | frac stab before unmask |",
            "|----:|--:|---------------:|------------------------:|",
        ]
    )
    for p in summary["by_pos_in_block"]:
        m = p["mean_stabilize_step"]
        lines.append(
            f"| {p['pos_in_block']} | {p['n']} | {m:.2f} | {p['frac_stabilized_before_unmask']:.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    traces_path = Path(args.traces)
    meta_path = traces_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    rows = []
    for line in traces_path.open():
        rows.extend(analyze_record(json.loads(line)))

    summary = aggregate(rows)
    report = format_report(summary, meta)

    out_md = Path(args.out_md or traces_path.parent / "stabilization_report.md")
    out_md.write_text(report)
    (traces_path.parent / "stabilization_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(out_md)
    print(report)


if __name__ == "__main__":
    main()

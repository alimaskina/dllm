#!/usr/bin/env python3
"""
Compare probe NLL: NLL_IID - NLL_WORD / NLL_SPAN by (k, t, whole/partial).

Positive delta => intervention lowered NLL vs IID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_buckets(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {k: v["mean_nll"] for k, v in data["buckets"].items()}


def compare(
    baseline: dict[str, float],
    target: dict[str, float],
    *,
    baseline_name: str,
    target_name: str,
) -> list[dict]:
    rows = []
    keys = sorted(set(baseline) & set(target))
    for key in keys:
        b = baseline[key]
        t = target[key]
        rows.append(
            {
                "bucket": key,
                "nll_baseline": b,
                "nll_target": t,
                f"delta_{baseline_name}_minus_{target_name}": b - t,
                "rel_improvement": (b - t) / b if b > 0 else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "causal_pilot_comparison.json",
    )
    args = parser.parse_args()

    rd = args.results_dir
    iid = load_buckets(rd / "nll_probe_iid.json")
    word = load_buckets(rd / "nll_probe_word.json")
    span = load_buckets(rd / "nll_probe_span.json")
    base = load_buckets(rd / "nll_probe_base.json") if (rd / "nll_probe_base.json").exists() else iid

    word_cmp = compare(iid, word, baseline_name="iid", target_name="word")
    span_cmp = compare(iid, span, baseline_name="iid", target_name="span")
    word_vs_span = compare(span, word, baseline_name="span", target_name="word")

    # Highlight whole high-k @ low t
    def filter_whole_high_k(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            b = r["bucket"]
            if "whole" not in b:
                continue
            if "k2|" in b:
                continue
            if not any(x in b for x in ("t0.2", "t0.3", "t0.4")):
                continue
            out.append(r)
        return out

    payload = {
        "interpretation": {
            "WORD > SPAN > IID on whole k3/k4+": "lexical fragmentation signal",
            "WORD ≈ SPAN > IID": "general local-fragmentation GO",
            "all ≈ 0": "causal exposure hypothesis weak",
        },
        "iid_minus_word": word_cmp,
        "iid_minus_span": span_cmp,
        "span_minus_word": word_vs_span,
        "summary_whole_high_k": {
            "iid_minus_word": filter_whole_high_k(word_cmp),
            "iid_minus_span": filter_whole_high_k(span_cmp),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

    print("\n=== Key buckets: NLL_IID - NLL_WORD (whole, k3/k4+, low t) ===")
    for r in filter_whole_high_k(word_cmp):
        d = r["delta_iid_minus_word"]
        print(f"  {r['bucket']}: delta={d:+.3f}  (IID={r['nll_baseline']:.3f} WORD={r['nll_target']:.3f})")


if __name__ == "__main__":
    main()

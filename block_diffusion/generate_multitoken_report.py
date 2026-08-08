#!/usr/bin/env python3
"""Markdown report: multi-token word unmasking on block-diffusion GSM8K traces."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from analyze_multitoken_words import analyze_row, has_step_series, load_traces

QWT = Path(__file__).resolve().parents[1] / "quant_where_to_unmask"
sys.path.insert(0, str(QWT))

from analyze_multitoken_words import order_pattern, step_gaps  # noqa: E402
from generate_wikitext_report import (  # noqa: E402
    first_token,
    layout_stats,
    pct,
    sibling_pred_stats,
    word_fully_consecutive,
)
from multitoken_word_filters import filter_instances, is_lexical  # noqa: E402


def generate_report(rows: list[dict], instances, tokenizer, out: Path, *, meta: dict) -> None:
    n = len(instances)
    lay = layout_stats(instances)
    sp = sibling_pred_stats(instances)
    n_traces = len(rows)

    lines = [
        f"# Multi-token word unmasking — block diffusion GSM8K\n\n",
        f"*Сгенерировано: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n\n",
        "## Setup\n\n",
        f"| Параметр | Значение |\n|----------|----------|\n",
        f"| Model | {meta.get('model_path', '?')} |\n",
        f"| bd_size | {meta.get('bd_size', '?')} |\n",
        f"| small_block_size | {meta.get('small_block_size', '?')} |\n",
        f"| threshold | {meta.get('threshold', '?')} |\n",
        f"| n_samples | {n_traces} |\n",
        f"| accuracy | {100 * sum(r.get('correct') for r in rows) / max(n_traces, 1):.1f}% |\n",
        f"| Word tier | strict lexical (`is_lexical`) |\n",
        f"| Word instances | {n} ({n / max(n_traces, 1):.2f}/trace) |\n\n",
        "## Summary\n\n",
        f"- step_span median: **{statistics.median(lay['spans']):.0f}**, mean: **{statistics.mean(lay['spans']):.2f}**\n",
        f"- full consecutive (gap=1): **{pct(lay['full_consecutive_steps'], n)}**\n",
        f"- first=left/right/middle: "
        f"**{pct(lay['pos_ctr']['left'], n)}** / **{pct(lay['pos_ctr']['right'], n)}** / "
        f"**{pct(lay['pos_ctr']['middle'], n)}**\n",
        f"- sibling pred==final: **{sp.get('acc', 0):.1%}** ({sp.get('n_siblings', 0)} obs)\n",
        f"- all siblings ok: **{sp.get('all_ok', 0):.1%}**\n",
    ]
    co = sp.get("co_unmask_0.8")
    if co:
        lines.append(f"- co-unmask @conf≥0.8: **{co[0]:.1%}** ({co[1]} cases, {co[2]} errors)\n")

    lines += ["\n## By ntok\n\n| ntok | count | span_med | sib_acc |\n|------|-------|----------|--------|\n"]
    by_ntok = Counter(len(i.positions) for i in instances)
    for ntok in sorted(by_ntok):
        grp = [i for i in instances if len(i.positions) == ntok]
        sacc = sibling_pred_stats(grp).get("acc", 0)
        med = statistics.median(i.step_span for i in grp)
        lines.append(f"| {ntok} | {len(grp)} | {med:.0f} | {sacc:.1%} |\n")

    lines += ["\n## Top words\n\n| word | count |\n|------|-------|\n"]
    for word, cnt in Counter(i.word for i in instances).most_common(25):
        lines.append(f"| `{word}` | {cnt} |\n")

    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", default="multitoken_report.md")
    parser.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    args = parser.parse_args()

    path = Path(args.traces)
    rows = load_traces(path)
    meta = {}
    meta_path = path.parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    instances = []
    for row in rows:
        instances.extend(analyze_row(row, tokenizer, kind_filter="alpha"))
    instances = filter_instances(instances, tier="lexical")

    if sum(has_step_series(r["trace"]) for r in rows) < len(rows):
        print("WARNING: re-run run_volatility.py for preds_by_step/conf_by_step in traces")

    out = Path(args.out)
    generate_report(rows, instances, tokenizer, out, meta=meta)
    print(f"Wrote {out} ({len(instances)} words from {len(rows)} traces)")


if __name__ == "__main__":
    main()

"""Analyze sweep JSONL: coverage, accuracy, idealized speedup vs cache length."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _block_avg_coverage(block: dict) -> float | None:
    masses = [
        layer.get("attention_mass_captured")
        for layer in block.get("layers", {}).values()
        if layer.get("attention_mass_captured") is not None
    ]
    if not masses:
        return None
    return sum(masses) / len(masses)


def extract_example_metrics(record: dict) -> dict[str, Any]:
    """Per-example coverage + speedup-by-cache-len."""
    blocks = record.get("blocks", [])
    coverages_by_len: dict[int, list[float]] = defaultdict(list)
    for blk in blocks:
        cache_len = blk.get("old_cache_len_at_start", 0)
        cov = _block_avg_coverage(blk)
        if cov is not None and cache_len > 0:
            coverages_by_len[cache_len].append(cov)

    cost = record.get("cost", {})
    by_cache_len = cost.get("by_cache_len", {})

    return {
        "example_id": record.get("example_id"),
        "correct": record.get("correct"),
        "gen_tokens": record.get("gen_tokens"),
        "avg_coverage": (
            sum(c for bl in coverages_by_len.values() for c in bl)
            / max(sum(len(v) for v in coverages_by_len.values()), 1)
            if coverages_by_len
            else None
        ),
        "coverage_by_cache_len": {
            str(k): sum(v) / len(v) for k, v in sorted(coverages_by_len.items())
        },
        "speedup_by_cache_len": {
            k: v.get("speedup_vs_dense_fp16", {})
            for k, v in by_cache_len.items()
        },
    }


def aggregate_config_records(records: list[dict]) -> dict[str, Any]:
    cfg = records[0].get("config", {})
    name = cfg.get("name", "unknown")
    correct = sum(1 for r in records if r.get("correct"))
    n = len(records)

    all_coverages: list[float] = []
    coverage_by_len: dict[int, list[float]] = defaultdict(list)
    speedup_qk_by_len: dict[int, list[float]] = defaultdict(list)
    speedup_kv_by_len: dict[int, list[float]] = defaultdict(list)
    speedup_bwqk_by_len: dict[int, list[float]] = defaultdict(list)
    sparsity_by_len: dict[int, list[float]] = defaultdict(list)

    for rec in records:
        m = extract_example_metrics(rec)
        if m["avg_coverage"] is not None:
            all_coverages.append(m["avg_coverage"])
        for k, cov in m.get("coverage_by_cache_len", {}).items():
            coverage_by_len[int(k)].append(cov)
        cost = rec.get("cost", {})
        for k, bucket in cost.get("by_cache_len", {}).items():
            kl = int(k)
            sp = bucket.get("speedup_vs_dense_fp16", {})
            if sp.get("qk_macs"):
                speedup_qk_by_len[kl].append(sp["qk_macs"])
            if sp.get("kv_bytes_read"):
                speedup_kv_by_len[kl].append(sp["kv_bytes_read"])
            if sp.get("bit_weighted_qk"):
                speedup_bwqk_by_len[kl].append(sp["bit_weighted_qk"])
            sparsity_by_len[kl].append(bucket.get("effective_sparsity", 1.0))

    def _mean_dict(d: dict[int, list[float]]) -> dict[str, float]:
        return {str(k): sum(v) / len(v) for k, v in sorted(d.items()) if v}

    # Trend: speedup at early vs late cache lengths
    cache_lens = sorted(speedup_qk_by_len.keys())
    early_speedup = late_speedup = None
    if len(cache_lens) >= 2:
        early = cache_lens[: max(1, len(cache_lens) // 3)]
        late = cache_lens[-max(1, len(cache_lens) // 3) :]
        early_qk = [x for k in early for x in speedup_qk_by_len[k]]
        late_qk = [x for k in late for x in speedup_qk_by_len[k]]
        if early_qk:
            early_speedup = sum(early_qk) / len(early_qk)
        if late_qk:
            late_speedup = sum(late_qk) / len(late_qk)

    cost_agg = records[0].get("cost", {}) if records else {}
    ideal = cost_agg.get("idealized_speedup_per_step", {})

    return {
        "config_name": name,
        "config": cfg,
        "num_examples": n,
        "accuracy": correct / max(n, 1),
        "avg_topk_coverage": sum(all_coverages) / max(len(all_coverages), 1) if all_coverages else None,
        "avg_idealized_speedup_exec_qk": ideal.get("exec_qk_macs"),
        "avg_idealized_speedup_exec_kv": ideal.get("exec_kv_bytes_read"),
        "avg_idealized_speedup_bit_weighted_qk": ideal.get("exec_bit_weighted_qk"),
        "coverage_by_cache_len_mean": _mean_dict(coverage_by_len),
        "speedup_qk_by_cache_len_mean": _mean_dict(speedup_qk_by_len),
        "speedup_kv_by_cache_len_mean": _mean_dict(speedup_kv_by_len),
        "speedup_bit_weighted_qk_by_cache_len_mean": _mean_dict(speedup_bwqk_by_len),
        "effective_sparsity_by_cache_len_mean": _mean_dict(sparsity_by_len),
        "speedup_trend": {
            "early_cache_qk_speedup": early_speedup,
            "late_cache_qk_speedup": late_speedup,
            "speedup_growth_ratio": (
                late_speedup / early_speedup if early_speedup and late_speedup else None
            ),
        },
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def analyze_sweep_jsonl(jsonl_path: Path) -> dict[str, Any]:
    records = load_jsonl(jsonl_path)
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        name = r.get("config", {}).get("name", "unknown")
        by_config[name].append(r)

    summaries = [aggregate_config_records(recs) for recs in by_config.values()]
    summaries.sort(key=lambda x: x["config_name"])

    # Global: speedup vs cache len pooled across sparse configs
    pooled_qk: dict[int, list[float]] = defaultdict(list)
    pooled_cov: dict[int, list[float]] = defaultdict(list)
    for s in summaries:
        if s["config_name"].startswith("baseline"):
            continue
        for k, v in s.get("speedup_qk_by_cache_len_mean", {}).items():
            pooled_qk[int(k)].append(v)
        for k, v in s.get("coverage_by_cache_len_mean", {}).items():
            pooled_cov[int(k)].append(v)

    global_by_len = []
    for cache_len in sorted(pooled_qk):
        global_by_len.append(
            {
                "old_cache_len": cache_len,
                "mean_qk_speedup": sum(pooled_qk[cache_len]) / len(pooled_qk[cache_len]),
                "mean_coverage": (
                    sum(pooled_cov[cache_len]) / len(pooled_cov[cache_len])
                    if pooled_cov[cache_len]
                    else None
                ),
            }
        )

    baseline = next((s for s in summaries if s["config_name"].startswith("baseline")), None)

    return {
        "num_configs": len(summaries),
        "num_records": len(records),
        "baseline_accuracy": baseline["accuracy"] if baseline else None,
        "config_summaries": summaries,
        "global_speedup_vs_cache_len": global_by_len,
    }


def write_sweep_report(analysis: dict[str, Any], path: Path) -> None:
    lines = [
        "# Sparse KV Sweep Report (GSM8K)",
        "",
        f"- configs analyzed: {analysis['num_configs']}",
        f"- records: {analysis['num_records']}",
        f"- baseline accuracy: {analysis.get('baseline_accuracy')}",
        f"- skipped configs: {len(analysis.get('skipped_configs', []))}",
        "",
    ]
    if analysis.get("skip_messages"):
        lines.append("## Skipped (expensive, extreme already ideal)")
        lines.append("")
        for msg in analysis["skip_messages"]:
            lines.append(f"- {msg}")
        lines.append("")
    lines.extend([
        "## Config summary (accuracy / coverage / speedup)",
        "",
        "| Config | Acc | Coverage | QK speedup | KV speedup | BW-QK speedup | Early→Late QK |",
        "|--------|-----|----------|------------|------------|---------------|---------------|",
    ])
    for s in analysis["config_summaries"]:
        trend = s.get("speedup_trend", {})
        early = trend.get("early_cache_qk_speedup")
        late = trend.get("late_cache_qk_speedup")
        trend_s = (
            f"{early:.2f}→{late:.2f}"
            if isinstance(early, (int, float)) and isinstance(late, (int, float))
            else "—"
        )

        def _f(x: Any) -> str:
            return f"{x:.3f}" if isinstance(x, (int, float)) else "—"

        lines.append(
            f"| {s['config_name']} | {_f(s['accuracy'])} | {_f(s['avg_topk_coverage'])} "
            f"| {_f(s['avg_idealized_speedup_exec_qk'])} | {_f(s['avg_idealized_speedup_exec_kv'])} "
            f"| {_f(s['avg_idealized_speedup_bit_weighted_qk'])} | {trend_s} |"
        )

    lines.extend(
        [
            "",
            "## Idealized QK speedup vs old-cache length (pooled over sparse configs)",
            "",
            "| old_cache_len | mean QK speedup | mean coverage |",
            "|---------------|-----------------|---------------|",
        ]
    )
    for row in analysis.get("global_speedup_vs_cache_len", []):
        cov = row.get("mean_coverage")
        cov_s = f"{cov:.3f}" if isinstance(cov, (int, float)) else "—"
        lines.append(
            f"| {row['old_cache_len']} | {row['mean_qk_speedup']:.3f} | {cov_s} |"
        )

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

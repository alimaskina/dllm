"""JSONL + markdown reporting for sparse KV experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)


def write_markdown_report(
    path: Path,
    *,
    title: str,
    runs: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Aggregate",
        "",
    ]
    for k, v in aggregate.items():
        lines.append(f"- **{k}**: {v}")
    lines.extend(["", "## Runs", ""])
    for run in runs:
        lines.append(f"### {run.get('config_name', run.get('name', '?'))}")
        lines.append("")
        lines.append(f"- accuracy: {run.get('accuracy', 'n/a')}")
        lines.append(f"- examples: {run.get('num_examples', 'n/a')}")
        if "cost_aggregate" in run:
            ca = run["cost_aggregate"]
            ratio = ca.get("ratio_vs_dense_fp16_per_step", ca.get("ratio_vs_dense_fp16", {})) if isinstance(ca, dict) else {}
            qk_r = ratio.get("exec_qk_macs", "n/a")
            kv_r = ratio.get("exec_kv_bytes_read", "n/a")
            qk_s = f"{qk_r:.4f}" if isinstance(qk_r, (int, float)) else str(qk_r)
            kv_s = f"{kv_r:.4f}" if isinstance(kv_r, (int, float)) else str(kv_r)
            lines.append(f"- QK MAC ratio vs dense FP16: {qk_s}")
            lines.append(f"- KV bytes read ratio: {kv_s}")
            lines.append(f"- avg top-k coverage: {run.get('avg_topk_coverage', 'n/a')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_run(
    *,
    config_name: str,
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    cost_aggregate: dict[str, Any],
) -> dict[str, Any]:
    correct = sum(1 for e in examples if e.get("correct"))
    coverages = []
    for ex in examples:
        for blk in ex.get("blocks", []):
            for layer in blk.get("layers", {}).values():
                if "attention_mass_captured" in layer:
                    coverages.append(layer["attention_mass_captured"])
    return {
        "config_name": config_name,
        "config": config,
        "num_examples": len(examples),
        "accuracy": correct / max(len(examples), 1),
        "avg_topk_coverage": sum(coverages) / max(len(coverages), 1) if coverages else None,
        "cost_aggregate": cost_aggregate,
        "examples": examples,
    }

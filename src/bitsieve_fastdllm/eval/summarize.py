from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Summarize quality, end-to-end performance, and microbenchmark JSONL files.')
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSONL files, directories (searched recursively), or shell-style glob patterns",
    )
    parser.add_argument("--output-prefix", required=True)
    return parser


def _expand_inputs(items: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.jsonl")))
            continue
        if path.is_file():
            paths.append(path)
            continue
        matches = [Path(value) for value in glob.glob(item, recursive=True)]
        for match in matches:
            if match.is_dir():
                paths.extend(sorted(match.rglob("*.jsonl")))
            elif match.is_file():
                paths.append(match)

    return list(dict.fromkeys(path.resolve() for path in paths))


def _mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def _median(values: list[float]) -> float | str:
    return statistics.median(values) if values else ""


def _value(row: dict[str, Any], key: str) -> Any:
    runtime = row.get("runtime")
    if isinstance(runtime, dict) and key in runtime:
        return runtime.get(key)
    return row.get(key)


def _floats(rows: list[dict[str, Any]], key: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        value = _value(row, key)
        if value is None or value == "":
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _config_name(row: dict[str, Any]) -> str:
    direct = row.get("config_name")
    if direct:
        return str(direct)
    config = row.get("config")
    if isinstance(config, dict) and config.get("name"):
        return str(config["name"])
    return "unknown"


def _schedule(row: dict[str, Any]) -> tuple[str, str]:
    config = row.get("config")
    if not isinstance(config, dict):
        return "", ""
    generation = config.get("generation")
    if not isinstance(generation, dict):
        return "", ""
    schedule = str(generation.get("schedule", ""))
    steps = generation.get("fixed_steps_per_block", "") if schedule == "fixed" else ""
    return schedule, str(steps)


def _classify(row: dict[str, Any]) -> str:
    if "operation" in row and "median_ms" in row:
        return "microbenchmark"
    if "score" in row and "benchmark" in row:
        return "quality"
    return "performance"


def _group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    kind = _classify(row)
    if kind == "quality":
        return kind, str(row.get("benchmark", "unknown")), _config_name(row)
    if kind == "microbenchmark":
        return (
            kind,
            str(row.get("operation", "unknown")),
            int(row.get("context_tokens", 0)),
            int(row.get("batch", 0)),
            int(row.get("selector_queries", 0)),
            int(row.get("topk", 0)),
            int(row.get("k_bits", 0)),
            int(row.get("v_bits", 0)),
            int(row.get("residual", row.get("residual_tokens", 0)) or 0),
        )
    schedule, fixed_steps = _schedule(row)
    return (
        kind,
        _config_name(row),
        int(row.get("context_tokens_per_request", row.get("context_tokens", 0)) or 0),
        int(row.get("batch_size", row.get("batch", 0)) or 0),
        schedule,
        fixed_steps,
    )


def _summarize_group(key: tuple[Any, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    kind = str(key[0])
    if kind == "quality":
        _, benchmark, config = key
        return {
            "kind": kind,
            "benchmark": benchmark,
            "config": config,
            "context_tokens": "",
            "batch_size": "",
            "schedule": "",
            "fixed_steps": "",
            "operation": "",
            "selector_queries": "",
            "topk": "",
            "k_bits": "",
            "v_bits": "",
            "residual": "",
            "n": len(rows),
            "mean_score": _mean(_floats(rows, "score")),
            "median_decode_ms": _median(_floats(rows, "decode_ms")),
            "median_tpob_ms": _median(_floats(rows, "mean_tpob_ms")),
            "median_tokens_per_second": _median(_floats(rows, "tokens_per_second")),
            "median_peak_allocated_bytes": _median(_floats(rows, "peak_cuda_allocated_bytes")),
            "median_peak_reserved_bytes": _median(_floats(rows, "peak_cuda_reserved_bytes")),
            "median_cache_bytes": _median(
                [
                    float(value)
                    for row in rows
                    for value in [
                        (
                            (_value(row, "packed_cache") or {}).get("total")
                            if isinstance(_value(row, "packed_cache"), dict)
                            else _value(row, "native_cache_bytes")
                        )
                    ]
                    if value is not None
                ]
            ),
            "median_compression_ratio": _median(_floats(rows, "cache_compression_ratio")),
            "median_operation_ms": "",
            "median_reported_p10_ms": "",
            "median_reported_p90_ms": "",
        }

    if kind == "microbenchmark":
        (
            _,
            operation,
            context,
            batch,
            selector_queries,
            topk,
            k_bits,
            v_bits,
            residual,
        ) = key
        return {
            "kind": kind,
            "benchmark": "",
            "config": "",
            "context_tokens": context,
            "batch_size": batch,
            "schedule": "",
            "fixed_steps": "",
            "operation": operation,
            "selector_queries": selector_queries,
            "topk": topk,
            "k_bits": k_bits,
            "v_bits": v_bits,
            "residual": residual,
            "n": len(rows),
            "mean_score": "",
            "median_decode_ms": "",
            "median_tpob_ms": "",
            "median_tokens_per_second": "",
            "median_peak_allocated_bytes": "",
            "median_peak_reserved_bytes": "",
            "median_cache_bytes": "",
            "median_compression_ratio": "",
            "median_operation_ms": _median(_floats(rows, "median_ms")),
            "median_reported_p10_ms": _median(_floats(rows, "p10_ms")),
            "median_reported_p90_ms": _median(_floats(rows, "p90_ms")),
        }

    _, config, context, batch, schedule, fixed_steps = key
    cache_values: list[float] = []
    for row in rows:
        packed = row.get("packed_cache")
        if isinstance(packed, dict) and packed.get("total") is not None:
            cache_values.append(float(packed["total"]))
        elif row.get("native_cache_bytes") is not None:
            cache_values.append(float(row["native_cache_bytes"]))
    return {
        "kind": kind,
        "benchmark": "",
        "config": config,
        "context_tokens": context,
        "batch_size": batch,
        "schedule": schedule,
        "fixed_steps": fixed_steps,
        "operation": "",
        "selector_queries": "",
        "topk": "",
        "k_bits": "",
        "v_bits": "",
        "residual": "",
        "n": len(rows),
        "mean_score": "",
        "median_decode_ms": _median(_floats(rows, "decode_ms")),
        "median_tpob_ms": _median(_floats(rows, "mean_tpob_ms")),
        "median_tokens_per_second": _median(_floats(rows, "tokens_per_second")),
        "median_peak_allocated_bytes": _median(_floats(rows, "peak_cuda_allocated_bytes")),
        "median_peak_reserved_bytes": _median(_floats(rows, "peak_cuda_reserved_bytes")),
        "median_cache_bytes": _median(cache_values),
        "median_compression_ratio": _median(_floats(rows, "cache_compression_ratio")),
        "median_operation_ms": "",
        "median_reported_p10_ms": "",
        "median_reported_p90_ms": "",
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    paths = _expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("No JSONL inputs were found")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    parse_errors: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"{path}:{line_number}: {exc}")
                    continue
                if not isinstance(row, dict):
                    continue
                groups[_group_key(row)].append(row)

    summary = [
        _summarize_group(key, rows)
        for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))
    ]

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    errors_path = prefix.with_suffix(".errors.txt")

    columns = list(summary[0]) if summary else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        if columns:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summary)

    if summary:
        lines = [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join(["---"] * len(columns)) + "|",
        ]
        for row in summary:
            lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        md_path.write_text("No rows found.\n", encoding="utf-8")

    if parse_errors:
        errors_path.write_text("\n".join(parse_errors) + "\n", encoding="utf-8")
    elif errors_path.exists():
        errors_path.unlink()

    print(f"inputs={len(paths)} groups={len(summary)} parse_errors={len(parse_errors)}")
    print(csv_path)
    print(md_path)
    if parse_errors:
        print(errors_path)


if __name__ == "__main__":
    main()

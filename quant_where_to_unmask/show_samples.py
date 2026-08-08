#!/usr/bin/env python3
"""Render per-sample outputs from official lm-eval --log_samples JSONL files."""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_samples_file(output_path: str) -> Path:
    base = Path(output_path)
    if base.is_dir():
        candidates = sorted(base.rglob("samples_gsm8k_*.jsonl"))
    else:
        candidates = sorted(base.parent.rglob("samples_gsm8k_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No samples_gsm8k_*.jsonl under {output_path}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="results_gsm8k_*.json dir or file")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    samples_path = find_samples_file(args.output_path)
    rows = list(load_jsonl(samples_path))

    out = Path(args.out or samples_path.with_suffix(".md"))
    lines = [f"# GSM8K samples from {samples_path.name}\n", f"n={len(rows)}\n"]

    for i, row in enumerate(rows):
        doc = row.get("doc", {})
        q = doc.get("question", "?")
        target = row.get("target", "")
        # resps: list of lists of generations per repeat
        resp = row["resps"][0][0] if row.get("resps") else ""
        filt = row.get("filtered_resps", [""])[0]
        metrics = {k: v for k, v in row.items() if k.startswith("exact_match")}

        lines.append(f"\n## Example {i+1}\n")
        lines.append(f"**Question:** {q}\n")
        lines.append(f"**Target:** {target[:200]}{'...' if len(str(target))>200 else ''}\n")
        lines.append(f"**Metrics:** {metrics}\n")
        lines.append(f"**Filtered:** {filt}\n")
        trunc = str(resp)[:2000]
        if len(str(resp)) > 2000:
            trunc += "...(truncated)"
        lines.append(f"\n### Generation\n```\n{trunc}\n```\n")

    out.write_text("".join(lines), encoding="utf-8")
    print(f"Read {len(rows)} samples from {samples_path}")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

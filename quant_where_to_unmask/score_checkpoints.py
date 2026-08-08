#!/usr/bin/env python3
"""Score partial/final GSM8K results from per-rank checkpoint jsonl files."""

import argparse
import json
import re
from pathlib import Path

from lm_eval import tasks


def extract_gold(answer: str) -> str | None:
    m = re.search(r"####\s*(-?[\d.,]+)", answer)
    return m.group(1).replace(",", "") if m else None


def extract_pred(text: str) -> str | None:
    m = re.search(r"####\s*(-?[\d.,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def load_rows(checkpoint_dir: Path):
    rows = []
    for path in sorted(checkpoint_dir.glob("rank*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    rows = load_rows(ckpt)
    if not rows:
        print(f"No checkpoints in {ckpt}")
        return

    task = tasks.get_task_dict(["gsm8k"])["gsm8k"]
    docs = list(task.test_docs())
    if args.limit:
        docs = docs[: args.limit]

    correct = 0
    total = 0
    for row in rows:
        doc_id = row.get("doc_id", row["idx"])
        if doc_id >= len(docs):
            continue
        gold = extract_gold(task.doc_to_target(docs[doc_id]))
        pred = extract_pred(row["response"])
        ok = gold is not None and pred == gold
        correct += int(ok)
        total += 1

    print(f"Checkpoint: {ckpt}")
    print(f"Lines: {len(rows)}  scored: {total}  limit: {args.limit or 'all'}")
    if total:
        print(f"exact_match (interim): {correct/total:.1%}  ({correct}/{total})")
    for p in sorted(ckpt.glob("rank*.jsonl")):
        n = sum(1 for _ in p.open())
        print(f"  {p.name}: {n} samples")


if __name__ == "__main__":
    main()

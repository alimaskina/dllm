#!/usr/bin/env python3
"""Preflight: the 200 LongBench examples must cover training and evaluation.

LongBench has no train split, so recovery training takes its examples from the
front of the test split and evaluation has to start past them. Getting that
arithmetic wrong is silent until the first trained LongBench cell, which is
hours into a full run -- and the failure mode if the offset is too *small* is
worse than a crash: the score would be measured on training examples.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bitsieve_fastdllm.eval.benchmarks import load_benchmark  # noqa: E402
from bitsieve_fastdllm.training.longbench_data import DEFAULT_TRAIN_TASKS  # noqa: E402


def main() -> int:
    per_task, heldout, offset, limit = (int(x) for x in sys.argv[1:5])
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "Efficient-Large-Model/Fast_dLLM_v2_7B", trust_remote_code=True
    )
    bad = False
    for task in DEFAULT_TRAIN_TASKS:
        total = len(load_benchmark(task, tokenizer=tok, limit=None, split="test"))
        used = per_task + heldout
        available = total - offset
        print(
            f"[preflight] {task}: {total} examples | training uses the first {used} "
            f"| eval starts at {offset}, leaving {available}"
        )
        if used > offset:
            print(
                f"  ERROR: training consumes {used} examples but evaluation only skips "
                f"{offset} -- the score would be measured on training examples."
            )
            bad = True
        if available < limit:
            print(f"  ERROR: only {available} examples remain but LIMIT_LONGBENCH={limit}.")
            bad = True
    if bad:
        print("\nAdjust LONGBENCH_PER_TASK / EXAMPLE_OFFSET / LIMIT_LONGBENCH and rerun.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lm_eval import tasks
from lm_eval.tasks import TaskManager
from tasks.sudoku4.utils import clean_generation, is_valid_sudoku_dream


def load_ckpt(d: str) -> dict:
    rows = {}
    for p in sorted(Path(d).glob("rank*.jsonl")):
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            rows[r["doc_id"]] = r
    return rows


def exact(row, doc) -> bool:
    return clean_generation(row["response"]).replace("\n", "") == doc["target"].replace("\n", "")


def main():
    root = Path(__file__).resolve().parent
    tm = TaskManager(include_path=str(root / "tasks"))
    docs = list(tasks.get_task_dict(["sudoku4_2shot_large"], tm)["sudoku4_2shot_large"].test_docs())
    n = len(docs)

    configs = [
        ("g24 FP16", root / "checkpoints/results_sudoku_fp16_large_4x4_2shot_2gpu"),
        ("g24 INT4", root / "checkpoints/results_sudoku_int4_large_4x4_2shot_2gpu"),
        ("g20 FP16", root / "checkpoints/results_sudoku_fp16_large_4x4_2shot_2gpu_g20"),
        ("g20 INT4", root / "checkpoints/results_sudoku_int4_large_4x4_2shot_2gpu_g20"),
        ("g21 FP16", root / "checkpoints/results_sudoku_fp16_large_4x4_2shot_2gpu_g21"),
        ("g21 INT4", root / "checkpoints/results_sudoku_int4_large_4x4_2shot_2gpu_g21"),
    ]
    print(f"{'Config':12s}  exact   valid   full16")
    for name, path in configs:
        rows = load_ckpt(path)
        ex = sum(exact(rows[i], docs[i]) for i in range(n))
        val = sum(
            is_valid_sudoku_dream(docs[i]["puzzle_input"], clean_generation(rows[i]["response"]))
            for i in range(n)
        )
        full = sum(len(re.sub(r"[^1-4]", "", clean_generation(rows[i]["response"]))) == 16 for i in range(n))
        print(f"{name:12s}  {ex/n*100:5.1f}%  {val/n*100:5.1f}%  {full/n*100:5.1f}%")

    for quant, p24, p20 in [
        ("FP16", configs[0][1], configs[2][1]),
        ("INT4", configs[1][1], configs[3][1]),
    ]:
        r24, r20 = load_ckpt(p24), load_ckpt(p20)
        both = g24o = g20o = 0
        for i, doc in enumerate(docs):
            e24, e20 = exact(r24[i], doc), exact(r20[i], doc)
            if e24 and e20:
                both += 1
            elif e24:
                g24o += 1
            elif e20:
                g20o += 1
        print(f"\n{quant} g24 vs g20: both={both}  g24_only={g24o}  g20_only={g20o}")

    trace = root / "checkpoints/results_sudoku_fp16_large_4x4_2shot_2gpu_g20/traces/rank0/0.json.gz"
    tr = json.load(gzip.open(trace, "rt"))
    order = [u["pos_comp"] for step in tr["steps_trace"] for u in step["unmasked"]]
    print(f"\ng20 sample order ({len(order)} steps): {order}")


if __name__ == "__main__":
    main()

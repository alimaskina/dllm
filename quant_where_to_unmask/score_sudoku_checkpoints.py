#!/usr/bin/env python3
"""Score partial/final Sudoku results from per-rank checkpoint jsonl files."""

import argparse
import json
import re
from pathlib import Path

from lm_eval import tasks
from lm_eval.tasks import TaskManager


def _is_valid_sudoku_dream(puzzle_input: str, prediction: str) -> bool:
    import numpy as np

    prediction = prediction.split("<|endoftext|>")[0].split("\n\n")[0].replace(" ", "")
    prediction = prediction[: len(puzzle_input)]
    try:
        input_array = np.array([list(map(int, row)) for row in puzzle_input.strip().split("\n")])
        grid = np.array([list(map(int, row)) for row in prediction.strip().split("\n")])
        if grid.shape != (4, 4):
            return False
    except (ValueError, TypeError):
        return False

    non_zero_mask = input_array != 0
    if not np.all(input_array[non_zero_mask] == grid[non_zero_mask]):
        return False

    expected_set = {1, 2, 3, 4}
    for row in grid:
        if set(row) != expected_set:
            return False
    for col in range(4):
        if {grid[row][col] for row in range(4)} != expected_set:
            return False
    for start_row in (0, 2):
        for start_col in (0, 2):
            subgrid = {
                grid[r][c]
                for r in range(start_row, start_row + 2)
                for c in range(start_col, start_col + 2)
            }
            if subgrid != expected_set:
                return False
    return True


def _is_valid_sudoku(grid_str: str, size: int = 9) -> bool:
    digits = re.sub(r"[^1-9]" if size == 9 else r"[^1-4]", "", grid_str)
    n = size * size
    if len(digits) != n:
        return False
    g = [int(c) for c in digits]
    exp = list(range(1, size + 1))
    box = 2 if size == 4 else 3
    for i in range(size):
        row = g[i * size : (i + 1) * size]
        col = [g[i + j * size] for j in range(size)]
        box_r, box_c = (i // box) * box, (i % box) * box
        bx = [g[(box_r + r) * size + box_c + c] for r in range(box) for c in range(box)]
        if sorted(row) != exp:
            return False
        if sorted(col) != exp:
            return False
        if sorted(bx) != exp:
            return False
    return True


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
    parser.add_argument("--task", default="sudoku",
                        choices=["sudoku", "sudoku_8shot", "sudoku4_8shot", "sudoku4_4shot", "sudoku4_2shot", "sudoku4_1shot"])
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    rows = load_rows(ckpt)
    if not rows:
        print(f"No checkpoints in {ckpt}")
        return

    include_path = str(Path(__file__).resolve().parent / "tasks")
    task_manager = TaskManager(include_path=include_path)
    task = tasks.get_task_dict([args.task], task_manager=task_manager)[args.task]
    docs = list(task.test_docs())
    if args.limit:
        docs = docs[: args.limit]

    exact = valid = total = 0
    for row in rows:
        doc_id = row.get("doc_id", row["idx"])
        if doc_id >= len(docs):
            continue
        gold_flat = docs[doc_id]["solution_flat"]
        gen = row["response"]
        gen_digits = re.sub(
            r"[^1-4]" if args.task in ("sudoku4_8shot", "sudoku4_4shot", "sudoku4_2shot", "sudoku4_1shot") else r"[^1-9]",
            "",
            gen,
        )
        exact += int(gen_digits == gold_flat)
        if args.task in ("sudoku4_8shot", "sudoku4_4shot", "sudoku4_2shot", "sudoku4_1shot"):
            valid += int(_is_valid_sudoku_dream(docs[doc_id]["puzzle_input"], gen))
        else:
            valid += int(_is_valid_sudoku(gen, size=9))
        total += 1

    print(f"Checkpoint: {ckpt}")
    print(f"Lines: {len(rows)}  scored: {total}  limit: {args.limit or 'all'}")
    if total:
        print(f"exact_match (interim): {exact/total:.1%}  ({exact}/{total})")
        print(f"valid_sudoku (interim): {valid/total:.1%}  ({valid}/{total})")
    for p in sorted(ckpt.glob("rank*.jsonl")):
        n = sum(1 for _ in p.open())
        print(f"  {p.name}: {n} samples")


if __name__ == "__main__":
    main()

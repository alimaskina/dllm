"""Sudoku task utilities for lm-eval (Ritvik19/sudoku-dataset)."""

from __future__ import annotations

import re

import numpy as np

FEWSHOT_INDICES = list(range(8))

INSTRUCTION_9X9 = (
    "Fill the positions where the values are 0 in a 9x9 grid with digits 1-9 so that "
    "each column, each row, and each of the nine 3x3 subgrids that compose the grid "
    "contains all of the digits from 1 to 9."
)


def fmt_grid(s81: str) -> str:
    s81 = str(s81).replace(".", "0")
    return "\n".join(" ".join(s81[r * 9 : (r + 1) * 9]) for r in range(9))


def fmt_compact(s81: str) -> str:
    s81 = str(s81).replace(".", "0")
    return "\n".join(s81[r * 9 : (r + 1) * 9] for r in range(9))


def process_docs(dataset, seed: int = 42, n_samples: int = 256):
    """Pick a fixed random subset and add prompt/answer fields."""
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(dataset))
    idx = rng.choice(len(dataset), n, replace=False).tolist()

    def _map(example):
        puzzle = str(example["puzzle"]).replace(".", "0")
        solution = str(example["solution"])
        return {
            **example,
            "prompt": f"Solve this Sudoku:\n{fmt_grid(puzzle)}",
            "answer": fmt_grid(solution),
            "solution_flat": solution,
        }

    return dataset.select(idx).map(_map)


def process_docs_8shot(dataset, seed: int = 42, n_samples: int = 256):
    """8-shot eval subset; excludes puzzles used as few-shot demonstrations."""
    rng = np.random.default_rng(seed)
    pool = [i for i in range(len(dataset)) if i not in FEWSHOT_INDICES]
    n = min(n_samples, len(pool))
    idx = rng.choice(pool, n, replace=False).tolist()

    def _map(example):
        puzzle = str(example["puzzle"]).replace(".", "0")
        solution = str(example["solution"])
        return {
            **example,
            "input": fmt_compact(puzzle),
            "target": fmt_compact(solution),
            "answer": fmt_compact(solution),
            "solution_flat": solution,
        }

    return dataset.select(idx).map(_map)


def _is_valid_sudoku(grid_str: str) -> bool:
    digits = re.sub(r"[^1-9]", "", grid_str)
    if len(digits) != 81:
        return False
    g = [int(c) for c in digits]
    for i in range(9):
        row = g[i * 9 : (i + 1) * 9]
        col = [g[i + j * 9] for j in range(9)]
        box_r, box_c = (i // 3) * 3, (i % 3) * 3
        box = [g[(box_r + r) * 9 + box_c + c] for r in range(3) for c in range(3)]
        if sorted(row) != list(range(1, 10)):
            return False
        if sorted(col) != list(range(1, 10)):
            return False
        if sorted(box) != list(range(1, 10)):
            return False
    return True


def process_results(doc, results):
    gen = results[0] if results else ""
    gold_flat = doc["solution_flat"]
    gen_digits = re.sub(r"[^1-9]", "", gen)
    exact = int(gen_digits == gold_flat)
    valid = int(_is_valid_sudoku(gen))
    return {"exact_match": exact, "valid_sudoku": valid}

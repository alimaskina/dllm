"""4x4 Sudoku task — Dream eval/data/sudoku_4x4_10.jsonl (8-shot + 100 test)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

INSTRUCTION_4X4 = (
    "Fill the positions where the values are 0 in a 4x4 grid with digits 1-4 so that "
    "each column, each row, and each of the four 2x2 subgrids that compose the grid "
    "contains all of the digits from 1 to 4."
)

DREAM_DATA = Path(__file__).resolve().parent / "data" / "sudoku_4x4_10.jsonl"
N_FEWSHOT = 8
GEN_LENGTH_4X4 = 24  # Dream eval_planning: max_new_tokens=24


def _flat_solution(output: str) -> str:
    return re.sub(r"[^1-4]", "", output)


def _parse_grid(text: str) -> np.ndarray | None:
    try:
        grid = np.array([list(map(int, row)) for row in text.strip().split("\n")])
        if grid.shape != (4, 4):
            return None
        return grid
    except (ValueError, TypeError):
        return None


def is_valid_sudoku_dream(puzzle_input: str, prediction: str) -> bool:
    """Dream sudoku_metric.is_valid_sudoku — preserves given clues + valid 4x4."""
    prediction = prediction[: len(puzzle_input)]
    input_array = _parse_grid(puzzle_input)
    grid = _parse_grid(prediction)
    if input_array is None or grid is None:
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


def clean_generation(gen: str) -> str:
    """Post-process like Dream eval_planning.eval_sudoku."""
    gen = gen.split("<|endoftext|>")[0].split("\n\n")[0].replace(" ", "")
    return gen


def process_docs(dataset, seed: int = 42, n_samples: int = 256):
    """Rows 0..7 are few-shot; rows 8+ are test (Dream 8-shot protocol)."""
    return _process_docs(dataset, n_fewshot=8, n_samples=n_samples, seed=seed)


def process_docs_1shot_large(dataset, seed: int = 42, n_samples: int = 900):
    """Row 0 few-shot; rows 8+ test (900 unique puzzles from all Dream files)."""
    return _process_docs(dataset, n_fewshot=8, n_samples=n_samples, seed=seed, test_start=8)


def process_docs_2shot_large(dataset, seed: int = 42, n_samples: int = 900):
    """Rows 0..1 few-shot; rows 8+ test (900 unique puzzles from all Dream files)."""
    return _process_docs(dataset, n_fewshot=8, n_samples=n_samples, seed=seed, test_start=8)


def process_docs_4shot_large(dataset, seed: int = 42, n_samples: int = 900):
    """Rows 0..3 few-shot; rows 8+ test (900 unique puzzles from all Dream files)."""
    return _process_docs(dataset, n_fewshot=8, n_samples=n_samples, seed=seed, test_start=8)


def process_docs_1shot(dataset, seed: int = 42, n_samples: int = 256):
    """Row 0 few-shot; rows 8+ test (same 100 tests as 8-shot on _10)."""
    return _process_docs(dataset, n_fewshot=8, n_samples=n_samples, seed=seed, test_start=8)


def process_docs_2shot(dataset, seed: int = 42, n_samples: int = 256):
    """Rows 0..1 few-shot; rows 8+ test (same 100 tests as 8-shot on _10)."""
    return _process_docs(dataset, n_fewshot=8, n_samples=n_samples, seed=seed, test_start=8)


def process_docs_4shot(dataset, seed: int = 42, n_samples: int = 256):
    """Rows 0..3 few-shot; rows 8+ test (same 100 tests as 8-shot on _10)."""
    return _process_docs(dataset, n_fewshot=8, n_samples=n_samples, seed=seed, test_start=8)


def process_docs_4shot_file4(dataset, seed: int = 42, n_samples: int = 256):
    """Rows 0..3 few-shot; rows 4+ test (Dream sudoku_4x4_4.jsonl protocol)."""
    return _process_docs(dataset, n_fewshot=4, n_samples=n_samples, seed=seed, test_start=4)


def _process_docs(dataset, n_fewshot: int, n_samples: int, seed: int, test_start: int | None = None):
    _ = seed
    start = test_start if test_start is not None else n_fewshot
    rows = []
    for i, doc in enumerate(dataset):
        if i < start:
            continue
        inp = doc["input"]
        out = doc["output"]
        rows.append(
            {
                "input": inp,
                "target": out,
                "answer": out,
                "puzzle_input": inp,
                "solution_flat": _flat_solution(out),
            }
        )
    if n_samples:
        rows = rows[: min(n_samples, len(rows))]
    from datasets import Dataset

    return Dataset.from_list(rows)


def process_results(doc, results):
    gen = clean_generation(results[0] if results else "")
    gold_flat = doc["solution_flat"]
    gen_digits = re.sub(r"[^1-4]", "", gen)
    exact = int(gen_digits == gold_flat)
    valid = int(is_valid_sudoku_dream(doc["puzzle_input"], gen))
    return {"exact_match": exact, "valid_sudoku": valid}

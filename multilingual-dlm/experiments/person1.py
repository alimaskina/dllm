"""Researcher 1 experiment template."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_model
from data import build_input_ids, get_word_units, load_flores
from trajectories import run_oracle_trajectory


def main() -> None:
    bundle = load_model("llada")
    row = load_flores(max_examples=1)[0]
    text = row["translations"]["en"]
    units = get_word_units(text, bundle.tokenizer, lang="en")
    input_ids = build_input_ids(units)
    traj = run_oracle_trajectory(bundle, input_ids, policy="confidence", steps=32)
    print(f"example_id={row['example_id']} tokens={len(traj)}")


if __name__ == "__main__":
    main()

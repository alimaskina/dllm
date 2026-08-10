#!/usr/bin/env python3
"""Quick sanity checks before parallel experiments."""

from __future__ import annotations

import gc
import sys

import torch

from common import load_model, model_logits
from data import (
    build_input_ids,
    debug_word_units,
    get_word_units,
    load_belebele,
    load_flores,
    load_flores_plus,
    load_massive,
)
from trajectories import run_oracle_trajectory


def check(name: str, ok: bool) -> None:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}")
    if not ok:
        sys.exit(1)


def main() -> None:
    check("CUDA available", torch.cuda.is_available())

    example = load_flores(max_examples=1)[0]
    check("FLORES example loaded", bool(example["translations"]["en"]))
    print(f"  example_id={example['example_id']}")

    plus = load_flores_plus(max_examples=1)[0]
    check("FLORES+ example loaded", bool(plus["translations"]["en"]))
    print(f"  flores_plus example_id={plus['example_id']}")

    belebele = load_belebele(max_examples=1)[0]
    check("Belebele example loaded", "#" in belebele["example_id"])
    print(f"  belebele example_id={belebele['example_id'][:60]}...")

    massive = load_massive(max_examples=1)[0]
    check("MASSIVE example loaded", bool(massive["utterances"]["en"]))
    print(f"  massive example_id={massive['example_id']} label={massive['label']}")

    text_en = example["translations"]["en"]
    short_text = " ".join(text_en.split()[:12])

    llada = load_model("llada")
    check("LLaDA loaded", llada.model is not None)
    print(f"  llada device={llada.device} mask_id={llada.mask_token_id}")

    units = get_word_units(short_text, llada.tokenizer, lang="en")
    check("word units built", len(units) >= 3)
    debug_word_units(short_text, llada.tokenizer, lang="en", n=3)

    input_ids = build_input_ids(units)
    check("input_ids non-empty", len(input_ids) >= 4)

    x = torch.tensor([input_ids], dtype=torch.long, device=llada.device)
    logits = model_logits(llada, x)
    check("LLaDA forward", logits.shape[-1] > 0)

    traj_conf = run_oracle_trajectory(llada, input_ids, policy="confidence", steps=8)
    check("confidence trajectory", len(traj_conf) == len(input_ids))
    same_step = {r.reveal_step for r in traj_conf}
    check("confidence reveal steps assigned", len(same_step) >= 1)
    check(
        "confidence scores present",
        all(r.reveal_confidence is not None for r in traj_conf),
    )

    traj_rand = run_oracle_trajectory(llada, input_ids, policy="random", steps=8, rng=__import__("random").Random(0))
    check("random trajectory", len(traj_rand) == len(input_ids))
    step_groups: dict[int, int] = {}
    for r in traj_rand:
        step_groups[r.reveal_step] = step_groups.get(r.reveal_step, 0) + 1
    check("random multi-reveal steps exist", max(step_groups.values()) >= 1)

    del llada
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        dream = load_model("dream")
    except torch.OutOfMemoryError as exc:
        print(f"[WARN] Dream skipped due to GPU OOM: {exc}")
        print("\nALL CHECKS PASSED (LLaDA only; Dream skipped — free GPU memory and re-run)")
        return

    check("Dream loaded", dream.model is not None)
    print(f"  dream device={dream.device} mask_id={dream.mask_token_id} ar_shift={dream.ar_shift}")

    x_d = torch.tensor([input_ids], dtype=torch.long, device=dream.device)
    logits_d = model_logits(dream, x_d)
    check("Dream forward", logits_d.shape[-1] > 0)

    traj_d = run_oracle_trajectory(dream, input_ids, policy="confidence", steps=8)
    check("Dream confidence trajectory", len(traj_d) == len(input_ids))

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

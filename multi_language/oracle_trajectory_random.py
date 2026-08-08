#!/usr/bin/env python3
"""
Step 3: oracle reverse-trajectory mismatch — random scheduler (negative control).

Simulates revealing gold tokens in random order; P_traj should match P_train for random policy.
Confidence/margin require model forward passes (see oracle_trajectory_model.py).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from data_sources import load_opus_texts
from word_utils import segment_words


@dataclass
class TrajectoryStats:
    lang: str
    tokenizer: str
    policy: str
    n_sentences: int
    n_word_obs: int
    by_k: dict[str, dict[str, float]]


def tokenize_sentence(text: str, lang: str, tokenizer) -> tuple[list[int], list[list[int]]]:
    """Return flat token ids and word → list of token position indices."""
    words = segment_words(text, lang)
    all_ids: list[int] = []
    word_positions: list[list[int]] = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if not ids:
            continue
        pos = list(range(len(all_ids), len(all_ids) + len(ids)))
        all_ids.extend(ids)
        if len(ids) >= 1:
            word_positions.append(pos)
    return all_ids, word_positions


def tokenize_sentence_with_meta(
    text: str,
    lang: str,
    tokenizer,
) -> tuple[list[int], list[list[int]], list[dict]]:
    """Return token ids, word position lists, and per-word metadata."""
    words = segment_words(text, lang)
    all_ids: list[int] = []
    word_positions: list[list[int]] = []
    word_metas: list[dict] = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if not ids:
            continue
        pos = list(range(len(all_ids), len(all_ids) + len(ids)))
        all_ids.extend(ids)
        word_positions.append(pos)
        word_metas.append({"word": w, "positions": pos, "k": len(ids), "char_len": len(w)})
    return all_ids, word_positions, word_metas


def simulate_random_trajectory(
    word_positions: list[list[int]],
    n_tokens: int,
    rng: random.Random,
) -> list[float]:
    """
    Random reveal order. Return list of (k, t) observations:
    for each word at each step when we record global mask ratio t,
    whether whole word is still fully masked.
    """
    masked = set(range(n_tokens))
    order = list(masked)
    rng.shuffle(order)
    observations: list[tuple[int, float, bool]] = []
    for step, pos in enumerate(order):
        t = len(masked) / n_tokens  # mask ratio BEFORE reveal at this step
        for positions in word_positions:
            k = len(positions)
            unresolved = all(p in masked for p in positions)
            observations.append((k, t, unresolved))
        masked.remove(pos)
    return observations


def aggregate_obs(observations: list[tuple[int, float, bool]], t_bins: list[float]) -> dict:
    # Bin t into nearest grid point
    by_k_t: dict[int, dict[float, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for k, t, unresolved in observations:
        t_bin = min(t_bins, key=lambda tb: abs(tb - t))
        by_k_t[k][t_bin].append(unresolved)

    out = {}
    for k in sorted(by_k_t.keys()):
        out[str(k)] = {}
        for t in t_bins:
            vals = by_k_t[k].get(t, [])
            if vals:
                out[str(k)][str(t)] = sum(vals) / len(vals)
    return out


def p_train(k: int, t: float) -> float:
    return t ** k


def compute_mismatch(traj_p: dict, t_bins: list[float]) -> dict:
    mm = {}
    for k_str, t_dict in traj_p.items():
        k = int(k_str)
        mm[k_str] = {}
        for t_str, p_tr in t_dict.items():
            t = float(t_str)
            pt = p_train(k, t)
            mm[k_str][t_str] = (p_tr / pt) if pt > 0 else None
    return mm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="llada")
    parser.add_argument("--langs", default="en,de,ru,tr,fi,zh,ko")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--t-grid", default="0.1,0.2,0.3,0.5,0.7,0.9", help="Mask ratio bins"
    )
    parser.add_argument("--out", type=Path, default=Path("multi_language/results/oracle_random.json"))
    args = parser.parse_args()

    from tokenization_audit import TOKENIZER_PRESETS

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    t_bins = [float(x) for x in args.t_grid.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model_id = TOKENIZER_PRESETS[args.tokenizer]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    rng = random.Random(args.seed)

    results = []
    for lang in langs:
        texts = load_opus_texts(lang, max_samples=args.max_samples)
        all_obs: list[tuple[int, float, bool]] = []
        n_sent = 0
        for text in texts:
            ids, word_pos = tokenize_sentence(text, lang, tokenizer)
            if len(ids) < 4 or not word_pos:
                continue
            obs = simulate_random_trajectory(word_pos, len(ids), rng)
            all_obs.extend(obs)
            n_sent += 1

        traj_p = aggregate_obs(all_obs, t_bins)
        mismatch = compute_mismatch(traj_p, t_bins)
        row = TrajectoryStats(
            lang=lang,
            tokenizer=args.tokenizer,
            policy="random",
            n_sentences=n_sent,
            n_word_obs=len(all_obs),
            by_k=traj_p,
        )
        results.append({"stats": asdict(row), "mismatch_vs_train": mismatch})
        print(f"{lang}: sentences={n_sent}, obs={len(all_obs)}")
        # Sanity: random policy mismatch should ≈ 1.0
        for k in ("2", "3"):
            if k in mismatch and "0.3" in mismatch[k]:
                print(f"  k={k}, t=0.3: Mismatch={mismatch[k]['0.3']:.3f} (expect ~1.0)")

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

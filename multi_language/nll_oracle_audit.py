#!/usr/bin/env python3
"""
Step 4: gold NLL on oracle confidence trajectories.

Are states underrepresented during training (whole word unresolved) also harder?
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data_sources import load_opus_texts
from model_loader import load_llada
from tokenization_audit import TOKENIZER_PRESETS
from nll_analysis import (
    assign_t_bins,
    compute_freq_map,
    enrich_records,
    summarize_by_group,
    summarize_whole_vs_partial,
)
from trajectory_utils import confidence_trajectory, tokenize_sentence


def aggregate_nll(records: list[dict], t_bins: list[float]) -> dict:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        t_bin = min(t_bins, key=lambda tb: abs(tb - r["t"]))
        label = "whole_unresolved" if r["whole_word_unresolved"] else "other"
        buckets[label][f"{r['k']}@{t_bin:.1f}"].append(r["nll"])
        buckets[label][f"all@{t_bin:.1f}"].append(r["nll"])

    out = {}
    for label, groups in buckets.items():
        out[label] = {}
        for key, vals in groups.items():
            out[label][key] = {
                "mean_nll": statistics.mean(vals),
                "median_nll": statistics.median(vals),
                "n": len(vals),
            }
    return out


def summarize_by_k_t(records: list[dict], t_bins: list[float], min_k: int = 2) -> dict:
    """Mean NLL: whole-word-unresolved vs partial, grouped by k and t."""
    groups: dict[tuple[int, float, bool], list[float]] = defaultdict(list)
    for r in records:
        if r["k"] < min_k:
            continue
        t_bin = min(t_bins, key=lambda tb: abs(tb - r["t"]))
        groups[(r["k"], t_bin, r["whole_word_unresolved"])].append(r["nll"])

    out = {}
    for (k, t, whole), vals in sorted(groups.items()):
        key = f"k={k}|t={t:.1f}|whole={whole}"
        out[key] = {"mean_nll": statistics.mean(vals), "n": len(vals)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada")
    parser.add_argument("--langs", default="en,ru,de,fi")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--t-grid", default="0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "nll_oracle_llada.json",
    )
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    t_bins = [float(x) for x in args.t_grid.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model_id = TOKENIZER_PRESETS[args.model]
    mask_id = 126336
    print(f"Loading {model_id} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = load_llada(model_id, args.device)
    model.eval()

    results = []
    for lang in langs:
        texts = load_opus_texts(lang, max_samples=args.max_samples)
        freq_map = compute_freq_map(texts, lang)
        log_freqs = sorted(math.log1p(c) for c in freq_map.values())
        freq_edges = [log_freqs[int(len(log_freqs) * q)] for q in (0.25, 0.5, 0.75)] if log_freqs else [0.0]
        all_nll: list[dict] = []
        n_sent = 0
        for text in tqdm(texts, desc=lang):
            ids, word_pos = tokenize_sentence(text, lang, tokenizer)
            multi = [wp for wp in word_pos if len(wp) >= 2]
            if len(ids) < 8 or len(ids) > 128 or not multi:
                continue
            _, nll_records, _ = confidence_trajectory(
                model,
                ids,
                multi,
                oracle=True,
                mask_id=mask_id,
                steps=args.steps,
                remasking="low_confidence",
                device=args.device,
                collect_nll=True,
            )
            all_nll.extend(nll_records or [])
            n_sent += 1

        assign_t_bins(all_nll, t_bins)
        enrich_records(all_nll, freq_map, freq_edges)
        summary = summarize_whole_vs_partial(all_nll)
        row = {
            "lang": lang,
            "n_sentences": n_sent,
            "n_nll_records": len(all_nll),
            "mean_nll_whole_unresolved": summary["mean_nll_whole"],
            "mean_nll_partial_sibling": summary["mean_nll_partial"],
            "mean_nll_whole_low_t": summary["mean_nll_whole_low_t"],
            "mean_nll_partial_low_t": summary["mean_nll_partial_low_t"],
            "ratio_whole_vs_partial": summary["ratio_whole_vs_partial"],
            "delta_nll_whole_minus_partial": summary["delta_nll_whole_minus_partial"],
            "by_k_t": summarize_by_group(all_nll, ("k", "t_bin"), min_n=30),
            "by_k_t_char": summarize_by_group(all_nll, ("k", "t_bin", "char_bucket"), min_n=40),
            "by_k_t_freq": summarize_by_group(all_nll, ("k", "t_bin", "freq_bucket"), min_n=40),
            "aggregated": aggregate_nll(all_nll, t_bins),
        }
        results.append(row)
        print(f"\n{lang}: n_sent={n_sent}")
        print(
            f"  NLL whole-unresolved={row['mean_nll_whole_unresolved']:.3f}  "
            f"partial={row['mean_nll_partial_sibling']:.3f}  "
            f"ratio={row['ratio_whole_vs_partial']:.2f}x"
            if row["ratio_whole_vs_partial"]
            else f"  {lang}: insufficient data"
        )
        if row["mean_nll_whole_low_t"] and row["mean_nll_partial_low_t"]:
            print(
                f"  low-t (t≤0.35): whole={row['mean_nll_whole_low_t']:.3f}  "
                f"partial={row['mean_nll_partial_low_t']:.3f}"
            )

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 4b: gold NLL on oracle AND free decoding trajectories, with controls.

Separates:
  - whole-word-unresolved (OOD mask state) vs partial sibling visible
  - stratified by k, t, char length, word frequency
  - cross-language delta at matched (k, t) bins
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data_sources import load_opus_texts
from model_loader import load_llada
from nll_analysis import (
    assign_t_bins,
    compute_freq_map,
    cross_lang_k_t_deltas,
    enrich_records,
    summarize_by_group,
    summarize_whole_vs_partial,
)
from tokenization_audit import TOKENIZER_PRESETS
from trajectory_utils import confidence_trajectory, tokenize_sentence_with_meta


def freq_quartile_edges(freq_map: dict[str, int]) -> list[float]:
    if not freq_map:
        return [0.0, 1.0, 2.0, 3.0]
    logs = sorted(math.log1p(c) for c in freq_map.values())
    n = len(logs)
    if n < 4:
        return [logs[-1]] if logs else [0.0]
    return [logs[n // 4], logs[n // 2], logs[3 * n // 4]]


def run_mode(
    model,
    texts: list[str],
    lang: str,
    tokenizer,
    *,
    oracle: bool,
    mask_id: int,
    steps: int,
    remasking: str,
    device: str,
    freq_map: dict[str, int],
    freq_edges: list[float],
    t_bins: list[float],
) -> tuple[list[dict], int]:
    all_nll: list[dict] = []
    n_sent = 0
    for text in tqdm(texts, desc=f"{lang}|{'oracle' if oracle else 'free'}", leave=False):
        ids, word_pos, word_metas = tokenize_sentence_with_meta(text, lang, tokenizer)
        multi = [wp for wp in word_pos if len(wp) >= 2]
        if len(ids) < 8 or len(ids) > 128 or not multi:
            continue
        _, nll_records, _ = confidence_trajectory(
            model,
            ids,
            word_pos,
            oracle=oracle,
            mask_id=mask_id,
            steps=steps,
            remasking=remasking,
            device=device,
            collect_nll=True,
            word_metas=word_metas,
        )
        if nll_records:
            enrich_records(nll_records, freq_map, freq_edges)
            assign_t_bins(nll_records, t_bins)
            all_nll.extend(nll_records)
        n_sent += 1
    return all_nll, n_sent


def build_lang_row(
    lang: str,
    all_nll: list[dict],
    n_sent: int,
    t_bins: list[float],
) -> dict:
    summary = summarize_whole_vs_partial(all_nll)
    return {
        "lang": lang,
        "n_sentences": n_sent,
        "n_nll_records": len(all_nll),
        **summary,
        "by_k_t": summarize_by_group(all_nll, ("k", "t_bin"), min_n=30),
        "by_k_t_char": summarize_by_group(all_nll, ("k", "t_bin", "char_bucket"), min_n=20),
        "by_k_t_freq": summarize_by_group(all_nll, ("k", "t_bin", "freq_bucket"), min_n=20),
        "by_t": summarize_by_group(all_nll, ("t_bin",), min_n=50),
        "by_k": summarize_by_group(all_nll, ("k",), min_n=50),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada")
    parser.add_argument("--langs", default="en,ru,de,fi")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--t-grid", default="0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument("--modes", default="oracle,free", help="comma: oracle,free")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("multi_language/results/nll_decoding_llada.json"),
    )
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    t_bins = [float(x) for x in args.t_grid.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model_id = TOKENIZER_PRESETS[args.model]
    mask_id = 126336
    print(f"Loading {model_id} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = load_llada(model_id, args.device)
    model.eval()

    payload: dict = {"model": args.model, "modes": {}}

    for mode in modes:
        oracle = mode == "oracle"
        print(f"\n=== mode={mode} (oracle={oracle}) ===")
        lang_records: dict[str, list[dict]] = {}
        per_lang_rows = []

        for lang in langs:
            texts = load_opus_texts(lang, max_samples=args.max_samples)
            freq_map = compute_freq_map(texts, lang)
            freq_edges = freq_quartile_edges(freq_map)

            all_nll, n_sent = run_mode(
                model,
                texts,
                lang,
                tokenizer,
                oracle=oracle,
                mask_id=mask_id,
                steps=args.steps,
                remasking="low_confidence",
                device=args.device,
                freq_map=freq_map,
                freq_edges=freq_edges,
                t_bins=t_bins,
            )
            lang_records[lang] = all_nll
            row = build_lang_row(lang, all_nll, n_sent, t_bins)
            per_lang_rows.append(row)

            print(
                f"{lang}: n_sent={n_sent}  "
                f"whole={row['mean_nll_whole']:.3f}  partial={row['mean_nll_partial']:.3f}  "
                f"ratio={row['ratio_whole_vs_partial']:.2f}x  delta={row['delta_nll_whole_minus_partial']:.3f}"
                if row["ratio_whole_vs_partial"]
                else f"{lang}: insufficient data"
            )

        cross = cross_lang_k_t_deltas(lang_records, t_bins)
        # highlight k=3,4 at t≈0.3
        focus_keys = [k for k in cross if "k=3|t_bin=0.3" in k or "k=4|t_bin=0.3" in k]

        payload["modes"][mode] = {
            "oracle": oracle,
            "per_lang": per_lang_rows,
            "cross_lang_k_t_deltas": cross,
            "focus_k3_k4_t03": {k: cross[k] for k in focus_keys if k in cross},
        }

    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

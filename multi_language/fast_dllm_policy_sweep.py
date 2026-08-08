#!/usr/bin/env python3
"""Policy sweep for Fast-dLLM using native block MDM trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data_sources import load_opus_texts
from fast_dllm_trajectory import fast_dllm_confidence_trajectory
from model_loader import FAST_DLLM_MASK_ID, load_dlm
from oracle_trajectory_random import aggregate_obs, compute_mismatch, p_train
from tokenization_audit import TOKENIZER_PRESETS
from trajectory_utils import tokenize_sentence

POLICIES = ("low_confidence", "topk_margin", "random")


def run_lang(
    model,
    texts: list[str],
    lang: str,
    tokenizer,
    *,
    policy: str,
    block_size: int,
    device: str,
    t_bins: list[float],
) -> dict:
    all_obs = []
    n_sent = 0
    for text in tqdm(texts, desc=f"{lang}|{policy}", leave=False):
        ids, word_pos = tokenize_sentence(text, lang, tokenizer)
        multi = [wp for wp in word_pos if len(wp) >= 2]
        if len(ids) < 8 or len(ids) > block_size or not multi:
            continue
        obs, _ = fast_dllm_confidence_trajectory(
            model,
            ids,
            multi,
            oracle=True,
            mask_id=FAST_DLLM_MASK_ID,
            block_size=block_size,
            remasking=policy,
            device=device,
        )
        if obs:
            all_obs.extend(obs)
            n_sent += 1
    traj_p = aggregate_obs(all_obs, t_bins)
    mismatch = compute_mismatch(traj_p, t_bins)
    return {"lang": lang, "policy": policy, "n_sentences": n_sent, "traj_p": traj_p, "mismatch": mismatch}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fast_dllm")
    parser.add_argument("--langs", default="en,ru,de,fi")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--max-samples", type=int, default=60)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--t-grid", default="0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "policy_sweep_fast_dllm_native.json",
    )
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    t_bins = [float(x) for x in args.t_grid.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model_id = TOKENIZER_PRESETS[args.model]
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = load_dlm(args.model, args.device)
    model.eval()

    texts_by_lang = {lang: load_opus_texts(lang, max_samples=args.max_samples) for lang in langs}
    results_by_policy: dict[str, list] = {}

    for policy in policies:
        print(f"\n=== {policy} ===")
        rows = []
        for lang in langs:
            row = run_lang(
                model,
                texts_by_lang[lang],
                lang,
                tokenizer,
                policy=policy,
                block_size=args.block_size,
                device=args.device,
                t_bins=t_bins,
            )
            rows.append(row)
            mm3 = row["mismatch"].get("3", {}).get("0.3")
            mm4 = row["mismatch"].get("4", {}).get("0.3")
            print(f"{lang}: n={row['n_sentences']} k3={mm3} k4={mm4}")
        results_by_policy[policy] = rows

    payload = {"model": args.model, "block_size": args.block_size, "policies": results_by_policy}
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Policy sweep (feasibility §6): oracle mismatch under multiple reveal policies.

Policies: low_confidence, topk_margin, random, l2r.
Random should ≈ 1.0 mismatch; confidence/margin >> 1 at low t and high k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data_sources import load_opus_texts
from model_loader import get_dlm_spec, load_dlm, resolve_mask_id
from oracle_trajectory_random import aggregate_obs, compute_mismatch, p_train
from tokenization_audit import TOKENIZER_PRESETS
from trajectory_utils import confidence_trajectory, tokenize_sentence

POLICIES = ("low_confidence", "topk_margin", "random", "l2r")


def run_policy(
    model,
    texts: list[str],
    lang: str,
    tokenizer,
    *,
    policy: str,
    mask_id: int,
    steps: int,
    device: str,
    t_bins: list[float],
    ar_shift: bool = False,
) -> dict:
    all_obs = []
    n_sent = 0
    for text in tqdm(texts, desc=f"{lang}|{policy}", leave=False):
        ids, word_pos = tokenize_sentence(text, lang, tokenizer)
        multi = [wp for wp in word_pos if len(wp) >= 2]
        if len(ids) < 8 or len(ids) > 128 or not multi:
            continue
        obs, _, _ = confidence_trajectory(
            model,
            ids,
            multi,
            oracle=True,
            mask_id=mask_id,
            steps=steps,
            remasking=policy,
            device=device,
            ar_shift=ar_shift,
        )
        all_obs.extend(obs)
        n_sent += 1
    traj_p = aggregate_obs(all_obs, t_bins)
    mismatch = compute_mismatch(traj_p, t_bins)
    return {"lang": lang, "policy": policy, "n_sentences": n_sent, "traj_p": traj_p, "mismatch": mismatch}


def summarize_cross_lang(results_by_policy: dict, t_focus: str = "0.3") -> dict:
    """Mean mismatch(k,t) across langs for each policy."""
    out = {}
    for policy, lang_rows in results_by_policy.items():
        out[policy] = {}
        for k in ("2", "3", "4"):
            vals = []
            for row in lang_rows:
                mm = row["mismatch"].get(k, {}).get(t_focus)
                if mm is not None:
                    vals.append(mm)
            if vals:
                out[policy][f"k={k}|t={t_focus}"] = sum(vals) / len(vals)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada")
    parser.add_argument("--langs", default="en,ru,de,fi")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--t-grid", default="0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "policy_sweep_llada.json",
    )
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    t_bins = [float(x) for x in args.t_grid.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model_id = TOKENIZER_PRESETS[args.model]
    spec = get_dlm_spec(args.model)
    print(f"Loading {model_id} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mask_id = resolve_mask_id(args.model, tokenizer, spec)
    model = load_dlm(args.model, args.device)
    model.eval()

    texts_by_lang = {lang: load_opus_texts(lang, max_samples=args.max_samples) for lang in langs}
    results_by_policy: dict[str, list] = {}

    for policy in policies:
        print(f"\n=== policy={policy} ===")
        rows = []
        for lang in langs:
            row = run_policy(
                model,
                texts_by_lang[lang],
                lang,
                tokenizer,
                policy=policy,
                mask_id=mask_id,
                steps=args.steps,
                device=args.device,
                t_bins=t_bins,
                ar_shift=spec.ar_shift,
            )
            rows.append(row)
            print(f"{lang}: n_sent={row['n_sentences']}", end="")
            for k in ("3", "4"):
                mm = row["mismatch"].get(k, {}).get("0.3")
                if mm is not None:
                    print(f"  k={k}@t=0.3: {mm:.1f}x", end="")
            print()
        results_by_policy[policy] = rows

    cross = summarize_cross_lang(results_by_policy)
    payload = {
        "model": args.model,
        "policies": results_by_policy,
        "cross_lang_mean_mismatch": cross,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nCross-lang mean mismatch @ t=0.3:")
    for policy, kv in cross.items():
        print(f"  {policy}: " + ", ".join(f"{k}={v:.1f}x" for k, v in sorted(kv.items())))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

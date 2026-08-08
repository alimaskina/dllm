#!/usr/bin/env python3
"""
Step 3c: train–inference mismatch on FREE decoding (non-oracle).

Two setups:
  full     — entire sentence starts masked; reveal model argmax (errors accumulate)
  prompt   — prefix_frac of gold tokens as prompt; generate rest via generate()

Compare with oracle (3b): same metric, but real model predictions / generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

QWT = Path(__file__).resolve().parents[1] / "quant_where_to_unmask"
sys.path.insert(0, str(QWT))
from generate import generate  # noqa: E402

from data_sources import load_opus_texts
from model_loader import load_llada
from oracle_trajectory_random import aggregate_obs, compute_mismatch, p_train
from tokenization_audit import TOKENIZER_PRESETS
from trajectory_utils import (
    confidence_trajectory,
    observations_from_generate_trace,
    tokenize_sentence,
    word_positions_in_gen,
)


@torch.no_grad()
def run_full_free(
    model,
    tokenizer,
    text: str,
    lang: str,
    *,
    mask_id: int,
    steps: int,
    remasking: str,
    device: str,
) -> tuple[list, float | None]:
    ids, word_pos = tokenize_sentence(text, lang, tokenizer)
    multi = [wp for wp in word_pos if len(wp) >= 2]
    if len(ids) < 8 or len(ids) > 128 or not multi:
        return [], None
    obs, _, acc = confidence_trajectory(
        model,
        ids,
        multi,
        oracle=False,
        mask_id=mask_id,
        steps=steps,
        remasking=remasking,
        device=device,
    )
    return obs, acc


@torch.no_grad()
def run_prompt_free(
    model,
    tokenizer,
    text: str,
    lang: str,
    *,
    prefix_frac: float,
    mask_id: int,
    steps: int,
    remasking: str,
) -> tuple[list, float | None]:
    ids, _ = tokenize_sentence(text, lang, tokenizer)
    if len(ids) < 12 or len(ids) > 128:
        return [], None
    prefix_len = max(1, int(len(ids) * prefix_frac))
    if prefix_len >= len(ids) - 4:
        return [], None
    gen_length = len(ids) - prefix_len
    # gen_length must divide block_length in generate()
    block = gen_length
    while block > 1 and gen_length % block != 0:
        block -= 1
    gen_length = (gen_length // block) * block
    if gen_length < 8:
        return [], None
    prefix_len = len(ids) - gen_length

    prompt = torch.tensor([ids[:prefix_len]], dtype=torch.long, device=model.device)
    gold_gen = ids[prefix_len : prefix_len + gen_length]

    out, trace = generate(
        model,
        prompt,
        steps=steps,
        gen_length=gen_length,
        block_length=block,
        remasking=remasking,
        mask_id=mask_id,
        record_trace=True,
        prompt_len=prefix_len,
        tokenizer=tokenizer,
    )
    pred_gen = out[0, prefix_len : prefix_len + gen_length].tolist()
    acc = sum(a == b for a, b in zip(pred_gen, gold_gen)) / len(gold_gen)

    word_pos_gen = word_positions_in_gen(text, lang, tokenizer, prefix_len, prefix_len + gen_length)
    if not word_pos_gen:
        return [], acc
    obs = observations_from_generate_trace(trace[0], word_pos_gen, gen_length)
    return obs, acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada")
    parser.add_argument("--setup", choices=["full", "prompt", "both"], default="both")
    parser.add_argument("--prefix-frac", type=float, default=0.25)
    parser.add_argument("--langs", default="en,ru,de,fi")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--t-grid", default="0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument("--out", type=Path, default=Path("multi_language/results/free_decoding_llada.json"))
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    t_bins = [float(x) for x in args.t_grid.split(",")]
    setups = ["full", "prompt"] if args.setup == "both" else [args.setup]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model_id = TOKENIZER_PRESETS[args.model]
    mask_id = 126336
    print(f"Loading {model_id} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = load_llada(model_id, args.device)
    model.eval()

    results = []
    for setup in setups:
        for lang in langs:
            texts = load_opus_texts(lang, max_samples=args.max_samples)
            all_obs: list = []
            accs: list[float] = []
            n_sent = 0
            for text in tqdm(texts, desc=f"{setup}/{lang}"):
                if setup == "full":
                    obs, acc = run_full_free(
                        model,
                        tokenizer,
                        text,
                        lang,
                        mask_id=mask_id,
                        steps=args.steps,
                        remasking=args.remasking,
                        device=args.device,
                    )
                else:
                    obs, acc = run_prompt_free(
                        model,
                        tokenizer,
                        text,
                        lang,
                        prefix_frac=args.prefix_frac,
                        mask_id=mask_id,
                        steps=args.steps,
                        remasking=args.remasking,
                    )
                if not obs:
                    continue
                all_obs.extend(obs)
                if acc is not None:
                    accs.append(acc)
                n_sent += 1

            traj_p = aggregate_obs(all_obs, t_bins)
            mismatch = compute_mismatch(traj_p, t_bins)
            row = {
                "setup": setup,
                "lang": lang,
                "policy": args.remasking,
                "oracle": False,
                "n_sentences": n_sent,
                "mean_token_acc": sum(accs) / len(accs) if accs else None,
                "traj_p": traj_p,
                "mismatch": mismatch,
            }
            results.append(row)
            print(f"\n[{setup}] {lang}: n_sent={n_sent}, tok_acc={row['mean_token_acc']}")
            for k in ("2", "3", "4"):
                if k in mismatch:
                    for t in ("0.3", "0.5", "0.7"):
                        if t in mismatch[k] and mismatch[k][t] is not None:
                            print(
                                f"  k={k} t={t}: Mismatch={mismatch[k][t]:.2f}x "
                                f"(P_traj={traj_p.get(k, {}).get(t, 0):.4f})"
                            )

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 3 (model): oracle confidence-trajectory mismatch with LLaDA.

At each reverse step the scheduler picks positions to reveal; we inject gold tokens.
Compare P_traj(whole word unresolved | k, t) vs P_train = t^k.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

QWT = Path(__file__).resolve().parents[1] / "quant_where_to_unmask"
sys.path.insert(0, str(QWT))
from generate import get_num_transfer_tokens  # noqa: E402
from model_loader import load_llada  # noqa: E402

from data_sources import load_opus_texts
from oracle_trajectory_random import aggregate_obs, compute_mismatch, p_train, tokenize_sentence
from tokenization_audit import TOKENIZER_PRESETS


@torch.no_grad()
def oracle_confidence_trajectory(
    model,
    token_ids: list[int],
    word_positions: list[list[int]],
    *,
    mask_id: int,
    steps: int = 32,
    remasking: str = "low_confidence",
    device: str = "cuda",
) -> list[tuple[int, float, bool]]:
    n = len(token_ids)
    x = torch.full((1, n), mask_id, dtype=torch.long, device=device)
    gold = torch.tensor([token_ids], dtype=torch.long, device=device)
    masked_set = set(range(n))
    observations: list[tuple[int, float, bool]] = []

    mask_index = torch.ones((1, n), dtype=torch.bool, device=device)
    num_transfer = get_num_transfer_tokens(mask_index, steps)[0].tolist()

    for step_i in range(steps):
        t = len(masked_set) / n
        for positions in word_positions:
            k = len(positions)
            unresolved = all(p in masked_set for p in positions)
            observations.append((k, t, unresolved))

        k_reveal = num_transfer[step_i]
        if k_reveal <= 0 or not masked_set:
            break

        logits = model(x).logits
        x0 = torch.argmax(logits, dim=-1)
        if remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "topk_margin":
            p = F.softmax(logits, dim=-1)
            top2 = torch.topk(p, k=2, dim=-1).values
            conf = top2[..., 0] - top2[..., 1]
        else:
            conf = torch.rand_like(logits[:, :, 0])

        conf = conf.clone()
        conf[x != mask_id] = -float("inf")
        _, idxs = torch.topk(conf[0], k=min(k_reveal, len(masked_set)))
        for pos in idxs.tolist():
            if pos not in masked_set:
                continue
            x[0, pos] = gold[0, pos]
            masked_set.remove(pos)

    return observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada")
    parser.add_argument("--langs", default="en,ru,de,fi,zh")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "topk_margin", "random"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--t-grid", default="0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument("--out", type=Path, default=Path("multi_language/results/oracle_confidence_llada.json"))
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
    device = args.device

    results = []
    for lang in langs:
        texts = load_opus_texts(lang, max_samples=args.max_samples)
        all_obs: list[tuple[int, float, bool]] = []
        n_sent = 0
        for text in tqdm(texts, desc=lang):
            ids, word_pos = tokenize_sentence(text, lang, tokenizer)
            if len(ids) < 8 or len(ids) > 128:
                continue
            multi = [wp for wp in word_pos if len(wp) >= 2]
            if not multi:
                continue
            obs = oracle_confidence_trajectory(
                model,
                ids,
                multi,
                mask_id=mask_id,
                steps=args.steps,
                remasking=args.remasking,
                device=str(device),
            )
            all_obs.extend(obs)
            n_sent += 1

        traj_p = aggregate_obs(all_obs, t_bins)
        mismatch = compute_mismatch(traj_p, t_bins)
        results.append({"lang": lang, "policy": args.remasking, "n_sentences": n_sent, "traj_p": traj_p, "mismatch": mismatch})
        print(f"\n{lang}: n_sent={n_sent}")
        for k in ("2", "3", "4"):
            if k in mismatch:
                for t in ("0.3", "0.5", "0.7"):
                    if t in mismatch[k] and mismatch[k][t] is not None:
                        print(f"  k={k} t={t}: Mismatch={mismatch[k][t]:.2f}x  (P_traj={traj_p.get(k,{}).get(t,0):.4f}, P_train={p_train(int(k), float(t)):.4f})")

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

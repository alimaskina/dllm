#!/usr/bin/env python3
"""
Prompt generation eval: link mismatch / exposure → actual generation errors.

Setup: 25% gold prefix + generate() on remainder (same as free_decoding prompt).
Metrics: token accuracy, normalized char edit distance, exact-match rate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
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
from oracle_trajectory_random import aggregate_obs, compute_mismatch
from tokenization_audit import TOKENIZER_PRESETS
from trajectory_utils import observations_from_generate_trace, tokenize_sentence, word_positions_in_gen


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_edit(a: str, b: str) -> float:
    return levenshtein(a, b) / max(len(a), len(b), 1)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def filter_texts(texts: list[str], lang: str, tokenizer, *, min_tokens: int, max_tokens: int) -> list[str]:
    kept = []
    for text in texts:
        ids, _ = tokenize_sentence(text, lang, tokenizer)
        if min_tokens <= len(ids) <= max_tokens:
            kept.append(text)
    return kept


@torch.no_grad()
def run_prompt_eval(
    model,
    tokenizer,
    text: str,
    lang: str,
    *,
    prefix_frac: float,
    mask_id: int,
    steps: int,
    remasking: str,
) -> tuple[list, dict] | tuple[None, None]:
    ids, _ = tokenize_sentence(text, lang, tokenizer)
    if len(ids) < 12 or len(ids) > 128:
        return None, None
    prefix_len = max(1, int(len(ids) * prefix_frac))
    if prefix_len >= len(ids) - 4:
        return None, None
    gen_length = len(ids) - prefix_len
    block = gen_length
    while block > 1 and gen_length % block != 0:
        block -= 1
    gen_length = (gen_length // block) * block
    if gen_length < 8:
        return None, None
    prefix_len = len(ids) - gen_length
    gold_ids = ids[prefix_len : prefix_len + gen_length]

    prompt = torch.tensor([ids[:prefix_len]], dtype=torch.long, device=model.device)
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
    pred_ids = out[0, prefix_len : prefix_len + gen_length].tolist()
    tok_acc = sum(a == b for a, b in zip(pred_ids, gold_ids)) / len(gold_ids)
    gold_text = tokenizer.decode(gold_ids, skip_special_tokens=True)
    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

    word_pos_gen = word_positions_in_gen(text, lang, tokenizer, prefix_len, prefix_len + gen_length)
    if not word_pos_gen:
        return None, None
    obs = observations_from_generate_trace(trace[0], word_pos_gen, gen_length)
    metrics = {
        "tok_acc": tok_acc,
        "char_edit": normalized_edit(gold_text, pred_text),
        "exact_match": float(gold_text.strip() == pred_text.strip()),
    }
    return obs, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada")
    parser.add_argument("--langs", default="en,ru,de,fi")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--min-tokens", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefix-frac", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--t-grid", default="0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument(
        "--exposure-json",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "weighted_exposure_llada.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "prompt_gen_eval_llada.json",
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

    per_lang = []
    for lang in langs:
        texts = filter_texts(
            load_opus_texts(lang, max_samples=args.max_samples),
            lang,
            tokenizer,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
        rows: list[dict] = []
        all_obs = []
        for text in tqdm(texts, desc=lang):
            obs, metrics = run_prompt_eval(
                model,
                tokenizer,
                text,
                lang,
                prefix_frac=args.prefix_frac,
                mask_id=mask_id,
                steps=args.steps,
                remasking=args.remasking,
            )
            if obs is None or metrics is None:
                continue
            rows.append(metrics)
            all_obs.extend(obs)

        if not rows:
            continue
        traj_p = aggregate_obs(all_obs, t_bins)
        mismatch = compute_mismatch(traj_p, t_bins)
        summary = {
            "lang": lang,
            "n_sentences": len(rows),
            "mean_tok_acc": statistics.mean(r["tok_acc"] for r in rows),
            "median_tok_acc": statistics.median(r["tok_acc"] for r in rows),
            "mean_char_edit": statistics.mean(r["char_edit"] for r in rows),
            "exact_match_rate": statistics.mean(r["exact_match"] for r in rows),
            "traj_p": traj_p,
            "mismatch": mismatch,
        }
        per_lang.append(summary)
        print(
            f"\n{lang}: n={summary['n_sentences']} "
            f"tok_acc={summary['mean_tok_acc']:.3f} "
            f"char_edit={summary['mean_char_edit']:.3f} "
            f"exact={summary['exact_match_rate']:.3f}"
        )

    exposure_corr = None
    if args.exposure_json.exists() and per_lang:
        exp_data = json.loads(args.exposure_json.read_text())
        ratios = exp_data.get("exposure_ratio_pooled_ptraj_t03", {})
        xs, ys_ed, ys_acc = [], [], []
        for row in per_lang:
            lang = row["lang"]
            if lang in ratios:
                xs.append(ratios[lang].get("ratio_token_vs_en") or ratios[lang].get("ratio_corpus_vs_en"))
                ys_ed.append(row["mean_char_edit"])
                ys_acc.append(row["mean_tok_acc"])
        exposure_corr = {
            "exposure_vs_char_edit_r": pearson(xs, ys_ed),
            "exposure_vs_tok_acc_r": pearson(xs, ys_acc),
            "langs": [r["lang"] for r in per_lang],
        }

    payload = {
        "setup": "prompt",
        "prefix_frac": args.prefix_frac,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "policy": args.remasking,
        "per_lang": per_lang,
        "exposure_correlation": exposure_corr,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    if exposure_corr and exposure_corr["exposure_vs_char_edit_r"] is not None:
        print(
            f"Exposure vs char_edit r={exposure_corr['exposure_vs_char_edit_r']:.3f}  "
            f"vs tok_acc r={exposure_corr['exposure_vs_tok_acc_r']:.3f}"
        )


if __name__ == "__main__":
    main()

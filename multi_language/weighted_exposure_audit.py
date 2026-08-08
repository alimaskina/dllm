#!/usr/bin/env python3
"""
Weighted OOD exposure: fragmentation (f_k) × inference policy (P_traj).

Decomposes multilingual penalty without conflating:
  - how often words have k subtokens (corpus / tokenizer)
  - how often policy leaves whole k-words unresolved at mask ratio t

Counterfactuals (swap f or P_traj) isolate each factor.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_sources import load_opus_texts
from training_objective_audit import word_k_distribution
from transformers import AutoTokenizer

TOKENIZER = "GSAI-ML/LLaDA-8B-Base"
LANGS = ("en", "ru", "de", "fi")
T_BINS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
T_FOCUS = 0.3
K_MAX = 12  # cap tail; report sensitivity


def load_k_dist(lang: str, tokenizer, max_samples: int = 5000) -> dict[int, float]:
    texts = load_opus_texts(lang, max_samples=max_samples)
    ctr = word_k_distribution(texts, lang, tokenizer)
    n = sum(ctr.values())
    return {k: c / n for k, c in sorted(ctr.items())}


def pi_multi(k_dist: dict[int, float]) -> dict[int, float]:
    """π_k = P(k | k >= 2) over multi-token words."""
    multi = {k: p for k, p in k_dist.items() if k >= 2}
    s = sum(multi.values())
    return {k: p / s for k, p in multi.items()} if s > 0 else {}


def mean_k(k_dist: dict[int, float]) -> float:
    return sum(k * p for k, p in k_dist.items())


def get_ptraj(policy_data: dict, lang: str, policy: str, k: int, t: float) -> float | None:
    rows = policy_data["policies"][policy]
    row = next(r for r in rows if r["lang"] == lang)
    return row["traj_p"].get(str(k), {}).get(str(t))


def pool_ptraj(policy_data: dict, policy: str, k: int, t: float, langs: tuple[str, ...]) -> float | None:
    vals = []
    for lang in langs:
        v = get_ptraj(policy_data, lang, policy, k, t)
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def p_train(k: int, t: float) -> float:
    return t**k


def exposure_at_t(
    k_dist: dict[int, float],
    ptraj: dict[tuple[int, float], float],
    t: float,
    *,
    mode: str,
) -> dict[str, float]:
    """
    mode:
      corpus — Σ_{k≥2} f_k · P_traj(k,t)  (random word from full corpus)
      multi  — Σ_{k≥2} π_k · P_traj(k,t)  (random multi-token word only)
      token  — Σ_{k≥2} (k·f_k/E[k]) · P_traj(k,t)  (random subtoken)
    """
    pi = pi_multi(k_dist)
    mk = mean_k(k_dist)
    e_traj = 0.0
    e_train = 0.0
    for k, fk in k_dist.items():
        if k < 2:
            continue
        p_tr = ptraj.get((k, t))
        if p_tr is None:
            continue
        if mode == "corpus":
            w = fk
        elif mode == "multi":
            w = pi.get(k, 0.0)
        elif mode == "token":
            w = k * fk / mk
        else:
            raise ValueError(mode)
        e_traj += w * p_tr
        e_train += w * p_train(k, t)
    return {
        "exposure_traj": e_traj,
        "exposure_train": e_train,
        "excess": e_traj - e_train,
        "ratio_vs_train": e_traj / e_train if e_train > 0 else None,
    }


def exposure_word_level(
    k_dist: dict[int, float],
    ptraj: dict[tuple[int, float], float],
    *,
    t_bins: tuple[float, ...],
    use_token_weights: bool = False,
) -> dict[str, dict[str, float]]:
    """Backward-compatible wrapper."""
    mode = "token" if use_token_weights else "multi"
    return {str(t): exposure_at_t(k_dist, ptraj, t, mode=mode) for t in t_bins}


def integrated_exposure(
    k_dist: dict[int, float],
    ptraj: dict[tuple[int, float], float],
    t_bins: tuple[float, ...],
    mode: str,
) -> dict[str, float]:
    """Uniform average over t bins (PoC proxy for trajectory time)."""
    rows = [exposure_at_t(k_dist, ptraj, t, mode=mode) for t in t_bins]
    n = len(rows)
    return {
        "exposure_traj": sum(r["exposure_traj"] for r in rows) / n,
        "exposure_train": sum(r["exposure_train"] for r in rows) / n,
        "excess": sum(r["excess"] for r in rows) / n,
    }


def build_ptraj_table(
    policy_data: dict,
    policy: str,
    lang: str | None,
    langs: tuple[str, ...],
    t_bins: tuple[float, ...],
    k_max: int,
) -> dict[tuple[int, float], float]:
    table: dict[tuple[int, float], float] = {}
    for k in range(2, k_max + 1):
        for t in t_bins:
            if lang is not None:
                v = get_ptraj(policy_data, lang, policy, k, t)
            else:
                v = pool_ptraj(policy_data, policy, k, t, langs)
            if v is not None:
                table[(k, t)] = v
    return table


def counterfactual_matrix(
    k_dists: dict[str, dict[int, float]],
    ptraj_en: dict[tuple[int, float], float],
    ptraj_ru: dict[tuple[int, float], float],
    ptraj_pool: dict[tuple[int, float], float],
    t: float,
    mode: str,
) -> dict[str, float]:
    """2×2: f ∈ {EN,RU} × P_traj ∈ {EN,pool}."""

    def eval_exp(f_lang: str, pt: dict) -> float:
        return exposure_at_t(k_dists[f_lang], pt, t, mode=mode)["exposure_traj"]

    e_en_en = eval_exp("en", ptraj_en)
    e_ru_ru = eval_exp("ru", ptraj_ru)
    e_ru_pool = eval_exp("ru", ptraj_pool)
    e_en_pool = eval_exp("en", ptraj_pool)
    e_ru_en = eval_exp("ru", ptraj_en)

    return {
        "E_en_f_en_ptraj_en": e_en_en,
        "E_en_f_ru_ptraj_ru": e_ru_ru,
        "E_cf_f_ru_ptraj_en": e_ru_en,
        "E_cf_f_ru_ptraj_pool": e_ru_pool,
        "E_cf_f_en_ptraj_pool": e_en_pool,
        "fragmentation_factor_ru_vs_en": e_ru_pool / e_en_pool if e_en_pool > 0 else None,
        "ptraj_lang_gap_ru_vs_en_at_same_f": e_ru_ru / e_ru_en if e_ru_en > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-sweep",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "policy_sweep_llada.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "weighted_exposure_llada.json",
    )
    parser.add_argument("--policy", default="low_confidence")
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    policy_data = json.loads(args.policy_sweep.read_text())
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

    print("Loading k distributions from OPUS...")
    k_dists = {lang: load_k_dist(lang, tokenizer, args.max_samples) for lang in LANGS}

    ptraj_pool = build_ptraj_table(policy_data, args.policy, None, LANGS, T_BINS, K_MAX)
    per_lang = {}
    for lang in LANGS:
        pt_lang = build_ptraj_table(policy_data, args.policy, lang, LANGS, T_BINS, K_MAX)
        pt_pool = ptraj_pool
        p_multi = 1.0 - k_dists[lang].get(1, 0.0)
        per_lang[lang] = {
            "f_k": {str(k): v for k, v in k_dists[lang].items() if k <= K_MAX},
            "pi_k_given_multi": {str(k): v for k, v in pi_multi(k_dists[lang]).items()},
            "p_k_ge4": sum(p for k, p in k_dists[lang].items() if k >= 4),
            "p_multi": p_multi,
            "mean_k": mean_k(k_dists[lang]),
            "corpus_word_slot": {str(t): exposure_at_t(k_dists[lang], pt_pool, t, mode="corpus") for t in T_BINS},
            "multi_token_conditional": {str(t): exposure_at_t(k_dists[lang], pt_pool, t, mode="multi") for t in T_BINS},
            "token_weighted": {str(t): exposure_at_t(k_dists[lang], pt_pool, t, mode="token") for t in T_BINS},
            "per_lang_ptraj_corpus_t03": exposure_at_t(k_dists[lang], pt_lang, T_FOCUS, mode="corpus"),
            "integrated_corpus": integrated_exposure(k_dists[lang], pt_pool, T_BINS, mode="corpus"),
            "integrated_token": integrated_exposure(k_dists[lang], pt_pool, T_BINS, mode="token"),
        }

    t = T_FOCUS
    cf_corpus = counterfactual_matrix(
        k_dists,
        build_ptraj_table(policy_data, args.policy, "en", LANGS, T_BINS, K_MAX),
        build_ptraj_table(policy_data, args.policy, "ru", LANGS, T_BINS, K_MAX),
        ptraj_pool,
        t,
        mode="corpus",
    )
    cf_token = counterfactual_matrix(
        k_dists,
        build_ptraj_table(policy_data, args.policy, "en", LANGS, T_BINS, K_MAX),
        build_ptraj_table(policy_data, args.policy, "ru", LANGS, T_BINS, K_MAX),
        ptraj_pool,
        t,
        mode="token",
    )

    en_corpus = per_lang["en"]["corpus_word_slot"][str(t)]["exposure_traj"]
    en_token = per_lang["en"]["token_weighted"][str(t)]["exposure_traj"]
    ratios = {}
    for lang in LANGS:
        corp = per_lang[lang]["corpus_word_slot"][str(t)]["exposure_traj"]
        tok = per_lang[lang]["token_weighted"][str(t)]["exposure_traj"]
        ratios[lang] = {
            "corpus_exposure": corp,
            "ratio_corpus_vs_en": corp / en_corpus if en_corpus > 0 else None,
            "token_exposure": tok,
            "ratio_token_vs_en": tok / en_token if en_token > 0 else None,
        }

    payload = {
        "method": {
            "description": "Weighted OOD exposure = fragmentation f_k × policy P_traj(k,t)",
            "corpus_word_slot": "E_corpus(L,t) = Σ_{k≥2} f_k(L)·P_traj(k,t) — random word from corpus",
            "multi_token_conditional": "E_multi(L,t) = Σ_{k≥2} π_k(L)·P_traj — given word has k≥2",
            "token_weighted": "E_tok(L,t) = Σ_{k≥2} (k·f_k/E[k])·P_traj — random subtoken",
            "excess": "Excess = E_traj - E_train with same weights",
            "pooled_ptraj": "P_traj pooled over EN/RU/DE/FI — isolates fragmentation in cross-lang ratios",
            "counterfactual": "E(f_RU,P_pool)/E(f_EN,P_pool) with same metric mode",
            "integrated": "Uniform mean over t∈{0.1..0.9} as trajectory-time proxy",
        },
        "policy": args.policy,
        "t_focus": t,
        "per_lang": per_lang,
        "counterfactual_corpus_t03": cf_corpus,
        "counterfactual_token_t03": cf_token,
        "exposure_ratio_pooled_ptraj_t03": ratios,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== Corpus word-slot exposure @ t={t}, policy={args.policy} ===")
    print("E = Σ f_k · P_traj  (pooled policy; f_k from OPUS)\n")
    print(f"{'lang':<6} {'p_multi':>8} {'p_k≥4':>8} {'E_traj':>10} {'E_train':>10} {'excess':>10} {'vs EN':>8}")
    for lang in LANGS:
        row = per_lang[lang]["corpus_word_slot"][str(t)]
        r = ratios[lang]["ratio_corpus_vs_en"]
        pm = per_lang[lang]["p_multi"]
        pk4 = per_lang[lang]["p_k_ge4"]
        print(
            f"{lang:<6} {pm:8.1%} {pk4:8.1%} {row['exposure_traj']:10.4f} "
            f"{row['exposure_train']:10.4f} {row['excess']:10.4f} {r:8.2f}x"
        )

    print(f"\n=== Token-weighted exposure @ t={t} ===")
    print(f"{'lang':<6} {'E_traj':>10} {'E_train':>10} {'excess':>10} {'vs EN':>8}")
    for lang in LANGS:
        row = per_lang[lang]["token_weighted"][str(t)]
        r = ratios[lang]["ratio_token_vs_en"]
        print(
            f"{lang:<6} {row['exposure_traj']:10.4f} {row['exposure_train']:10.4f} "
            f"{row['excess']:10.4f} {r:8.2f}x"
        )

    print(f"\nCounterfactual corpus @ t={t}:")
    print(f"  E(EN f, EN ptraj)          = {cf_corpus['E_en_f_en_ptraj_en']:.4f}")
    print(f"  E(RU f, RU ptraj)          = {cf_corpus['E_en_f_ru_ptraj_ru']:.4f}")
    print(f"  E(RU f, EN ptraj) CF       = {cf_corpus['E_cf_f_ru_ptraj_en']:.4f}")
    print(f"  Fragmentation RU/EN (pool) = {cf_corpus['fragmentation_factor_ru_vs_en']:.2f}x")
    print(f"  P_traj lang gap @ same f_RU = {cf_corpus['ptraj_lang_gap_ru_vs_en_at_same_f']:.2f}x")

    print(f"\nCounterfactual token-weighted @ t={t}:")
    print(f"  Fragmentation RU/EN (pool) = {cf_token['fragmentation_factor_ru_vs_en']:.2f}x")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

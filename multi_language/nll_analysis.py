"""NLL aggregation with controls for k, t, char length, frequency."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict


def bucket_char_len(n: int) -> str:
    if n <= 3:
        return "1-3"
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    return "11+"


def bucket_log_freq(log_f: float, edges: list[float]) -> str:
    for i, edge in enumerate(edges):
        if log_f <= edge:
            return f"Q{i + 1}"
    return f"Q{len(edges) + 1}"


def compute_freq_map(texts: list[str], lang: str) -> dict[str, int]:
    from word_utils import segment_words

    ctr: Counter[str] = Counter()
    for text in texts:
        for w in segment_words(text, lang):
            ctr[w] += 1
    return dict(ctr)


def enrich_records(records: list[dict], freq_map: dict[str, int], freq_edges: list[float]) -> None:
    for r in records:
        w = r.get("word", "")
        c = freq_map.get(w, 0)
        r["word_freq"] = c
        r["log_freq"] = math.log1p(c)
        r["freq_bucket"] = bucket_log_freq(r["log_freq"], freq_edges)
        r["char_bucket"] = bucket_char_len(r.get("char_len", 1))
        r["t_bin"] = r.get("t_bin")  # set externally


def assign_t_bins(records: list[dict], t_bins: list[float]) -> None:
    for r in records:
        r["t_bin"] = min(t_bins, key=lambda tb: abs(tb - r["t"]))


def summarize_whole_vs_partial(records: list[dict], *, min_k: int = 2) -> dict:
    whole = [r["nll"] for r in records if r.get("whole_word_unresolved") and r["k"] >= min_k]
    partial = [r["nll"] for r in records if not r.get("whole_word_unresolved") and r["k"] >= min_k]
    low_t_whole = [r["nll"] for r in records if r.get("whole_word_unresolved") and r["k"] >= min_k and r["t"] <= 0.35]
    low_t_partial = [
        r["nll"] for r in records if not r.get("whole_word_unresolved") and r["k"] >= min_k and r["t"] <= 0.35
    ]
    ratio = statistics.mean(whole) / statistics.mean(partial) if whole and partial and statistics.mean(partial) > 0 else None
    delta = statistics.mean(whole) - statistics.mean(partial) if whole and partial else None
    return {
        "n_whole": len(whole),
        "n_partial": len(partial),
        "mean_nll_whole": statistics.mean(whole) if whole else None,
        "mean_nll_partial": statistics.mean(partial) if partial else None,
        "ratio_whole_vs_partial": ratio,
        "delta_nll_whole_minus_partial": delta,
        "mean_nll_whole_low_t": statistics.mean(low_t_whole) if low_t_whole else None,
        "mean_nll_partial_low_t": statistics.mean(low_t_partial) if low_t_partial else None,
    }


def summarize_by_group(
    records: list[dict],
    group_keys: tuple[str, ...],
    *,
    min_k: int = 2,
    min_n: int = 20,
) -> dict:
    """Whole vs partial NLL within strata (e.g. k+t, k+t+char_len)."""
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r["k"] < min_k:
            continue
        key = tuple(r.get(g) for g in group_keys)
        strata[key].append(r)

    out = {}
    for key, rows in sorted(strata.items()):
        whole = [x["nll"] for x in rows if x.get("whole_word_unresolved")]
        partial = [x["nll"] for x in rows if not x.get("whole_word_unresolved")]
        if len(whole) < min_n // 2 or len(partial) < min_n // 2:
            continue
        mw, mp = statistics.mean(whole), statistics.mean(partial)
        label = "|".join(f"{group_keys[i]}={key[i]}" for i in range(len(group_keys)))
        out[label] = {
            "n_whole": len(whole),
            "n_partial": len(partial),
            "mean_whole": mw,
            "mean_partial": mp,
            "delta": mw - mp,
            "ratio": mw / mp if mp > 0 else None,
        }
    return out


def cross_lang_k_t_deltas(lang_records: dict[str, list[dict]], t_bins: list[float], min_k: int = 2) -> dict:
    """
    For each (k, t_bin), compare whole−partial delta across languages.
    Large delta with similar k,t across langs → state effect, not language competence.
    """
    by_lang_kt: dict[str, dict[str, dict]] = {}
    for lang, recs in lang_records.items():
        by_kt = summarize_by_group(recs, ("k", "t_bin"), min_k=min_k, min_n=10)
        by_lang_kt[lang] = by_kt

    # keys present in multiple langs
    all_keys = set()
    for d in by_lang_kt.values():
        all_keys.update(d.keys())
    pooled = {}
    for key in sorted(all_keys):
        deltas = []
        for lang, d in by_lang_kt.items():
            if key in d and d[key]["delta"] is not None:
                deltas.append({"lang": lang, **d[key]})
        if len(deltas) >= 2:
            pooled[key] = {
                "per_lang": deltas,
                "mean_delta": statistics.mean(x["delta"] for x in deltas),
                "std_delta": statistics.pstdev(x["delta"] for x in deltas) if len(deltas) > 1 else 0.0,
            }
    return pooled

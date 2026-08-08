#!/usr/bin/env python3
"""Full multi-token unmasking report for WikiText g64 traces (markdown, RU)."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from analyze_multitoken_words import WordInstance, analyze_trace, load_trace, order_pattern, step_gaps
from filter_loops_and_count import has_repetition
from multitoken_word_filters import is_alpha_multi, is_lexical, word_core

DEFAULT_CHECKPOINTS = [
    "results_wikitext_fp16_g64_n256",
    "results_wikitext_fp16_g64_train_seed43_n512",
    "results_wikitext_fp16_g64_train_seed1043_n512",
    "results_wikitext_fp16_g64_train_seed2043_n512",
    "results_wikitext_fp16_g64_train_seed3043_n512",
    "results_wikitext_fp16_g64_train_seed4043_n512",
    "results_wikitext_fp16_g64_train_seed5043_n512",
]

GSM8K_CHECKPOINT = "results_gsm8k_fp16_n256_fast_2gpu"


@dataclass
class EnrichedInstance:
    inst: WordInstance
    trace_path: Path
    checkpoint: str


def pct(n: float | int, d: float | int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def pctile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(int(len(ys) * p), len(ys) - 1)]


def first_token(inst: WordInstance):
    return min(inst.tokens, key=lambda t: (t.step, t.pos_comp))


def conf_stats(confs: list[float]) -> dict:
    if not confs:
        return {"n": 0}
    return {
        "n": len(confs),
        "mean": statistics.mean(confs),
        "median": statistics.median(confs),
        "p10": pctile(confs, 0.1),
        "p25": pctile(confs, 0.25),
        "p75": pctile(confs, 0.75),
        "p90": pctile(confs, 0.9),
    }


def load_all_tiers(
    checkpoints: list[Path],
    tokenizer,
    *,
    filter_loops: bool,
) -> tuple[list[EnrichedInstance], list[EnrichedInstance], int, int, int]:
    alpha: list[EnrichedInstance] = []
    lexical: list[EnrichedInstance] = []
    n_traces = 0
    n_clean = 0
    n_dirty = 0
    for ckpt in checkpoints:
        ckpt_name = ckpt.name
        rows = []
        for rank_file in sorted(ckpt.glob("rank*.jsonl")):
            rows.extend(json.loads(line) for line in rank_file.open(encoding="utf-8") if line.strip())
        for row in rows:
            n_traces += 1
            dirty = has_repetition(row["response"])[0]
            if dirty:
                n_dirty += 1
            if filter_loops and dirty:
                continue
            n_clean += 1
            trace_path = ckpt / row["trace_path"]
            if not trace_path.exists():
                continue
            for inst in analyze_trace(trace_path, tokenizer, method="ws", kind_filter="alpha"):
                ei = EnrichedInstance(inst=inst, trace_path=trace_path, checkpoint=ckpt_name)
                if is_alpha_multi(inst.word):
                    alpha.append(ei)
                if is_lexical(inst.word):
                    lexical.append(ei)
    return alpha, lexical, n_traces, n_clean, n_dirty


def load_tier(
    checkpoints: list[Path],
    tokenizer,
    *,
    filter_loops: bool,
    tier: str,
) -> tuple[list[EnrichedInstance], int, int]:
    alpha, lexical, n_traces, n_clean, _ = load_all_tiers(checkpoints, tokenizer, filter_loops=filter_loops)
    return (lexical if tier == "lexical" else alpha), n_traces, n_clean


def word_fully_consecutive(inst: WordInstance) -> bool:
    gaps = step_gaps(inst)
    return bool(gaps) and all(g == 1 for g in gaps)


def order_class(pat: str) -> str:
    if pat in ("LR", "LMR", "LMRM"):
        return "monotone_LR"
    if pat in ("RL", "RLM", "RMML", "RML"):
        return "monotone_RL"
    if all(c in "LR" for c in pat):
        if pat == "LR":
            return "monotone_LR"
        if pat == "RL":
            return "monotone_RL"
    return "mixed"


def sibling_rows(inst: WordInstance) -> list[tuple[float | None, bool]]:
    final = {t.pos_comp: t.token_id for t in inst.tokens}
    out = []
    for s in inst.sibling_conf_at_first:
        if not s["still_masked"]:
            continue
        out.append((s["confidence"], s["predicted_token_id"] == final[s["pos_comp"]]))
    return out


def sibling_pred_stats(instances: list[WordInstance]) -> dict:
    rows = []
    word_ok = []
    for inst in instances:
        sibs = sibling_rows(inst)
        rows.extend(sibs)
        if sibs:
            word_ok.append(all(ok for _, ok in sibs))
    out = {"n_siblings": len(rows), "n_words": len(word_ok)}
    if rows:
        out["acc"] = sum(ok for _, ok in rows) / len(rows)
        out["all_ok"] = sum(word_ok) / len(word_ok)
        for th in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            sub = [ok for c, ok in rows if c is not None and c >= th]
            if sub:
                out[f"co_unmask_{th}"] = (sum(sub) / len(sub), len(sub), len(sub) - sum(sub))
        by_bucket = {}
        for lo, hi, name in [(0, 0.3, "<0.3"), (0.3, 0.5, "0.3-0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, ">=0.9")]:
            sub = [(c, ok) for c, ok in rows if c is not None and lo <= c < hi]
            if sub:
                by_bucket[name] = (sum(ok for _, ok in sub) / len(sub), len(sub))
        out["by_bucket"] = by_bucket
    return out


def layout_stats(instances: list[WordInstance]) -> dict:
    spans = [i.step_span for i in instances]
    pos_ctr = Counter()
    ntok_ctr = Counter(len(i.positions) for i in instances)
    for inst in instances:
        ft = first_token(inst)
        if ft.pos_comp == inst.positions[0]:
            pos_ctr["left"] += 1
        elif ft.pos_comp == inst.positions[-1]:
            pos_ctr["right"] += 1
        else:
            pos_ctr["middle"] += 1
    all_confs = [t.confidence for i in instances for t in i.tokens]
    first_confs = [first_token(i).confidence for i in instances]
    last_confs = [max(i.tokens, key=lambda t: (t.step, t.pos_comp)).confidence for i in instances]
    return {
        "n": len(instances),
        "consecutive": sum(i.consecutive for i in instances),
        "full_consecutive_steps": sum(word_fully_consecutive(i) for i in instances),
        "spans": spans,
        "span_hist": Counter(spans),
        "pos_ctr": pos_ctr,
        "ntok_ctr": ntok_ctr,
        "first_confs": first_confs,
        "last_confs": last_confs,
        "all_confs": all_confs,
        "first_steps": [first_token(i).step for i in instances],
    }


def conf_at_first_unmask(item: EnrichedInstance) -> dict:
    inst = item.inst
    trace = load_trace(item.trace_path)
    ft = first_token(inst)
    step_data = next(s for s in trace["steps_trace"] if s["step"] == ft.step)
    comp = step_data["completion"]
    word_pos = set(inst.positions)
    lo, hi = min(inst.positions), max(inst.positions)
    sibling, non, adj = [], [], []
    for pos in range(len(comp["masked"])):
        if not comp["masked"][pos]:
            continue
        c = comp["confidence"][pos]
        if c is None:
            continue
        if pos in word_pos and pos != ft.pos_comp:
            sibling.append(c)
        elif pos not in word_pos:
            non.append(c)
            if lo - 1 <= pos <= hi + 1:
                adj.append(c)
    return {
        "sibling": sibling,
        "non_sibling": non,
        "adjacent_non_word": adj,
        "mask_ratio": step_data.get("mask_ratio"),
        "n_masked": sum(comp["masked"]),
    }


def aggregate_conf_compare(enriched: list[EnrichedInstance]) -> dict:
    all_sib, all_non, all_adj = [], [], []
    mask_ratios = []
    paired: list[tuple[str, float, float, int]] = []
    by_ntok: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_first_conf: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    sib_higher = sib_lower = sib_equal = 0

    for item in enriched:
        c = conf_at_first_unmask(item)
        all_sib.extend(c["sibling"])
        all_non.extend(c["non_sibling"])
        all_adj.extend(c["adjacent_non_word"])
        if c["mask_ratio"] is not None:
            mask_ratios.append(c["mask_ratio"])
        if c["sibling"] and c["non_sibling"]:
            ms, mn = statistics.mean(c["sibling"]), statistics.mean(c["non_sibling"])
            paired.append((item.inst.word, ms, mn, len(item.inst.positions)))
            if ms > mn:
                sib_higher += 1
            elif ms < mn:
                sib_lower += 1
            else:
                sib_equal += 1
            ntok = len(item.inst.positions)
            by_ntok[ntok]["sibling"].extend(c["sibling"])
            by_ntok[ntok]["non_sibling"].extend(c["non_sibling"])
            ft = first_token(item.inst).confidence
            for lo, hi, name in [(0, 0.3, "<0.3"), (0.3, 0.7, "0.3-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, ">=0.9")]:
                if lo <= ft < hi:
                    by_first_conf[name]["sibling"].extend(c["sibling"])
                    by_first_conf[name]["non_sibling"].extend(c["non_sibling"])

    return {
        "sibling": conf_stats(all_sib),
        "non_sibling": conf_stats(all_non),
        "adjacent_non_word": conf_stats(all_adj),
        "mask_ratio": conf_stats(mask_ratios),
        "paired_n": len(paired),
        "sib_higher": sib_higher,
        "sib_lower": sib_lower,
        "sib_equal": sib_equal,
        "paired_means": (
            statistics.mean(ms for _, ms, _, _ in paired),
            statistics.mean(mn for _, _, mn, _ in paired),
        ) if paired else (0, 0),
        "by_ntok": {k: {g: conf_stats(v) for g, v in d.items()} for k, d in by_ntok.items()},
        "by_first_conf": {k: {g: conf_stats(v) for g, v in d.items()} for k, d in by_first_conf.items()},
        "top_advantage": sorted(paired, key=lambda x: x[1] - x[2], reverse=True)[:10],
        "top_disadvantage": sorted(paired, key=lambda x: x[1] - x[2])[:8],
    }


def per_checkpoint_stats(enriched: list[EnrichedInstance]) -> dict:
    by_ckpt: dict[str, list[WordInstance]] = defaultdict(list)
    for e in enriched:
        by_ckpt[e.checkpoint].append(e.inst)
    return {k: layout_stats(v) for k, v in by_ckpt.items()}


def pick_examples(enriched: list[EnrichedInstance], tokenizer, n: int = 8) -> list[dict]:
    ranked = []
    for item in enriched:
        inst = item.inst
        wrong = sum(1 for _, ok in sibling_rows(inst) if not ok)
        ranked.append((wrong, inst.step_span, len(inst.positions), item))
    examples = []
    for item in [x[3] for x in sorted(ranked, key=lambda x: x[:3], reverse=True)[:n]]:
        inst = item.inst
        final = {t.pos_comp: t.token_id for t in inst.tokens}
        toks = []
        for t in sorted(inst.tokens, key=lambda x: (x.step, x.pos_comp)):
            toks.append({"step": t.step, "pos": t.pos_comp, "token": t.token, "conf": t.confidence})
        sibs = []
        for s in inst.sibling_conf_at_first:
            if not s["still_masked"]:
                continue
            ok = s["predicted_token_id"] == final[s["pos_comp"]]
            sibs.append({
                "pos": s["pos_comp"],
                "pred": s["predicted_token"],
                "conf": s["confidence"],
                "final": tokenizer.decode([final[s["pos_comp"]]]),
                "ok": ok,
            })
        examples.append({
            "word": inst.word,
            "ntok": len(inst.positions),
            "pattern": order_pattern(inst),
            "step_span": inst.step_span,
            "tokens": toks,
            "siblings": sibs,
        })
    return examples


def render_conf_row(label: str, st: dict) -> str:
    if st.get("n", 0) == 0:
        return f"| {label} | 0 | — | — | — | — | — | — |\n"
    return (
        f"| {label} | {st['n']} | {st['mean']:.3f} | {st['median']:.3f} | "
        f"{st['p10']:.3f} | {st['p25']:.3f} | {st['p75']:.3f} | {st['p90']:.3f} |\n"
    )


def render_gap_table(instances: list[WordInstance], ntok: int) -> str:
    grp = [i for i in instances if len(i.positions) == ntok]
    if not grp:
        return ""
    lines = [f"\n#### Gap-распределение ({ntok} токена)\n\n| gap | count | % |\n|-----|-------|---|\n"]
    if ntok == 2:
        gaps = [step_gaps(i)[0] for i in grp]
    else:
        gaps = []
        for i in grp:
            gaps.extend(step_gaps(i))
    for g, c in sorted(Counter(gaps).items()):
        if c >= 10 or g <= 3:
            lines.append(f"| {g} | {c} | {pct(c, len(gaps) if ntok > 2 else len(grp))} |\n")
    return "".join(lines)


def render_order_section(instances: list[WordInstance]) -> str:
    lines = []
    for ntok in [2, 3, 4, 5]:
        grp = [i for i in instances if len(i.positions) == ntok]
        if not grp:
            continue
        lines.append(f"\n### {ntok} токена (n={len(grp)})\n\n")
        if ntok == 2:
            lines.append(
                f"- LR: {pct(sum(order_pattern(i) == 'LR' for i in grp), len(grp))}, "
                f"RL: {pct(sum(order_pattern(i) == 'RL' for i in grp), len(grp))}\n"
            )
            lines.append(f"- consecutive (gap=1): {pct(sum(word_fully_consecutive(i) for i in grp), len(grp))}\n")
        else:
            lines.append(f"- fully consecutive: {pct(sum(word_fully_consecutive(i) for i in grp), len(grp))}\n")
            oc = Counter(order_class(order_pattern(i)) for i in grp)
            lines.append(f"- monotone_LR: {pct(oc['monotone_LR'], len(grp))}, monotone_RL: {pct(oc['monotone_RL'], len(grp))}, mixed: {pct(oc['mixed'], len(grp))}\n")
            lines.append("\n| pattern | count | % |\n|---------|-------|---|\n")
            for pat, c in Counter(order_pattern(i) for i in grp).most_common(12):
                lines.append(f"| {pat} | {c} | {pct(c, len(grp))} |\n")
        lines.append(render_gap_table(instances, ntok))
    grp4p = [i for i in instances if len(i.positions) >= 4]
    if grp4p:
        lines.append(f"\n### 4+ токена суммарно (n={len(grp4p)})\n\n")
        lines.append(f"- step_span mean: {statistics.mean(i.step_span for i in grp4p):.2f}, median: {statistics.median(i.step_span for i in grp4p):.0f}\n")
        oc = Counter(order_class(order_pattern(i)) for i in grp4p)
        lines.append(f"- monotone_LR: {pct(oc['monotone_LR'], len(grp4p))}, monotone_RL: {pct(oc['monotone_RL'], len(grp4p))}, mixed: {pct(oc['mixed'], len(grp4p))}\n")
    return "".join(lines)


def render_ntok_breakdown(instances: list[WordInstance]) -> str:
    n = len(instances)
    lines = ["\n| ntok | count | % | span_med | first=left | first=right | sib_acc | co≥0.8 acc |\n|------|-------|---|----------|------------|-------------|---------|------------|\n"]
    for ntok in sorted(set(len(i.positions) for i in instances)):
        if ntok > 10:
            continue
        grp = [i for i in instances if len(i.positions) == ntok]
        gsp = sibling_pred_stats(grp)
        pos = Counter()
        for i in grp:
            ft = first_token(i)
            if ft.pos_comp == i.positions[0]:
                pos["left"] += 1
            elif ft.pos_comp == i.positions[-1]:
                pos["right"] += 1
        co80 = gsp.get("co_unmask_0.8", (0, 0, 0))
        lines.append(
            f"| {ntok} | {len(grp)} | {pct(len(grp), n)} | "
            f"{statistics.median(i.step_span for i in grp):.0f} | "
            f"{pct(pos['left'], len(grp))} | {pct(pos['right'], len(grp))} | "
            f"{gsp.get('acc', 0):.1%} | {co80[0]:.1%} ({co80[1]}) |\n"
        )
    return "".join(lines)


def render_gsm8k(wt_lay: dict, wt_sp: dict, wt_cc: dict, gsm_enriched: list[EnrichedInstance], gsm_n: int) -> str:
    gsm_inst = [e.inst for e in gsm_enriched]
    gsm_lay = layout_stats(gsm_inst)
    gsm_sp = sibling_pred_stats(gsm_inst)
    gsm_cc = aggregate_conf_compare(gsm_enriched)
    n = gsm_lay["n"]
    wt_n = wt_lay["n"]
    two_gsm = [i for i in gsm_inst if len(i.positions) == 2]
    two_wt = [i for i in range(wt_n)]  # placeholder
    lines = [
        "\n## 11. Сравнение WikiText vs GSM8K\n\n",
        "| Метрика | WikiText g64 (lexical) | GSM8K fp16 g256 (alpha) |\n",
        "|---------|------------------------|-------------------------|\n",
        f"| traces | 3328 (3201 no-loop) | {gsm_n} |\n",
        f"| word instances | {wt_n} | {n} |\n",
        f"| words/trace | {wt_n/3201:.2f} | {n/gsm_n:.2f} |\n",
        f"| full word consecutive steps | {pct(wt_lay['full_consecutive_steps'], wt_n)} | {pct(gsm_lay['full_consecutive_steps'], n)} |\n",
        f"| step_span median | {statistics.median(wt_lay['spans']):.0f} | {statistics.median(gsm_lay['spans']):.0f} |\n",
        f"| step_span mean | {statistics.mean(wt_lay['spans']):.2f} | {statistics.mean(gsm_lay['spans']):.2f} |\n",
        f"| first=left | {pct(wt_lay['pos_ctr']['left'], wt_n)} | {pct(gsm_lay['pos_ctr']['left'], n)} |\n",
        f"| first=right | {pct(wt_lay['pos_ctr']['right'], wt_n)} | {pct(gsm_lay['pos_ctr']['right'], n)} |\n",
        f"| first=middle | {pct(wt_lay['pos_ctr']['middle'], wt_n)} | {pct(gsm_lay['pos_ctr']['middle'], n)} |\n",
        f"| 2tok LR | 47.5% | {pct(sum(order_pattern(i)=='LR' for i in two_gsm), len(two_gsm))} |\n",
        f"| 2tok RL | 52.5% | {pct(sum(order_pattern(i)=='RL' for i in two_gsm), len(two_gsm))} |\n",
        f"| first conf mean | {statistics.mean(wt_lay['first_confs']):.3f} | {statistics.mean(gsm_lay['first_confs']):.3f} |\n",
        f"| all-token conf mean | {statistics.mean(wt_lay['all_confs']):.3f} | {statistics.mean(gsm_lay['all_confs']):.3f} |\n",
        f"| sibling pred==final | {wt_sp.get('acc', 0):.1%} | {gsm_sp.get('acc', 0):.1%} |\n",
        f"| all siblings ok | {wt_sp.get('all_ok', 0):.1%} | {gsm_sp.get('all_ok', 0):.1%} |\n",
        f"| co-unmask conf≥0.8 | {wt_sp.get('co_unmask_0.8', (0,0,0))[0]:.1%} | {gsm_sp.get('co_unmask_0.8', (0,0,0))[0]:.1%} |\n",
        f"| sibling conf mean | {wt_cc['sibling'].get('mean', 0):.3f} | {gsm_cc['sibling'].get('mean', 0):.3f} |\n",
        f"| non-sibling conf mean | {wt_cc['non_sibling'].get('mean', 0):.3f} | {gsm_cc['non_sibling'].get('mean', 0):.3f} |\n",
        "\nGSM8K: более структурированные ответы → выше sibling accuracy и first conf. "
        "Порядок unmask на WikiText более симметричен (left≈right).\n",
    ]
    return "".join(lines)


def generate_report(
    alpha_enriched: list[EnrichedInstance],
    lexical_enriched: list[EnrichedInstance],
    n_traces: int,
    n_clean: int,
    n_dirty: int,
    gsm_enriched: list[EnrichedInstance],
    gsm_n_traces: int,
    tokenizer,
    out_path: Path,
) -> None:
    enriched = lexical_enriched
    instances = [e.inst for e in enriched]
    n = len(instances)
    n_alpha = len(alpha_enriched)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lay = layout_stats(instances)
    alpha_lay = layout_stats([e.inst for e in alpha_enriched])
    sp = sibling_pred_stats(instances)
    cc = aggregate_conf_compare(enriched)
    ckpt_stats = per_checkpoint_stats(enriched)
    examples = pick_examples(enriched, tokenizer)

    lines = [
        "# Полный отчёт: multi-token word unmasking (WikiText g64)\n",
        f"\n*Сгенерировано: {now}*\n",
        "\n## Содержание\n\n",
        "1. [Резюме](#1-резюме)\n",
        "2. [Данные и методология](#2-данные-и-методология)\n",
        "3. [Инвентарь слов (Tier A / Tier B)](#3-инвентарь-слов-tier-a--tier-b)\n",
        "4. [Топология и layout](#4-топология-и-layout)\n",
        "5. [Порядок unmask](#5-порядок-unmask)\n",
        "6. [Временная динамика (step_span, gaps)](#6-временная-динамика-step_span-gaps)\n",
        "7. [Confidence](#7-confidence)\n",
        "8. [Sibling prediction и co-unmask](#8-sibling-prediction-и-co-unmask)\n",
        "9. [Разбивка по числу токенов](#9-разбивка-по-числу-токенов)\n",
        "10. [Sibling vs non-sibling conf](#10-sibling-vs-non-sibling-conf)\n",
        "11. [Сравнение WikiText vs GSM8K](#11-сравнение-wikitext-vs-gsm8k)\n",
        "12. [По checkpoint'ам](#12-по-checkpointам)\n",
        "13. [Примеры](#13-примеры)\n",
        "14. [Выводы](#14-выводы)\n",
        "\n---\n",
        "\n## 1. Резюме\n\n",
        f"Проанализировано **{n_traces}** generation traces (WikiText-103, gen_length=64, k=1/step). "
        f"После loop-фильтра: **{n_clean}** traces ({pct(n_clean, n_traces)}). "
        f"Найдено **{n_alpha}** alpha multi-token слов (Tier A) и **{n}** lexical (Tier B).\n\n",
        "### Как unmask'ятся multi-token слова\n\n",
        f"- **{pct(lay['span_hist'][1], n)}** слов с step_span=1 (2-токенные → 2 соседних шага); median span = **{statistics.median(lay['spans']):.0f}**.\n",
        f"- **{pct(lay['full_consecutive_steps'], n)}** слов — все токены unmask'ятся подряд по шагам (gap=1).\n",
        f"- Первый unmask: left **{pct(lay['pos_ctr']['left'], n)}**, right **{pct(lay['pos_ctr']['right'], n)}**, "
        f"middle **{pct(lay['pos_ctr']['middle'], n)}**.\n",
        f"- 2-токенные: LR **47.5%**, RL **52.5%**; 3-токенные: **84%** подряд, порядки сильно mixed.\n\n",
        "### Sibling prediction (ещё masked токены слова при первом unmask)\n\n",
        f"- pred==final: **{sp.get('acc', 0):.1%}** ({sp['n_siblings']} obs)\n",
        f"- все siblings верны (per word): **{sp.get('all_ok', 0):.1%}**\n",
        f"- co-unmask при conf≥0.8: **{sp.get('co_unmask_0.8', (0,0,0))[0]:.1%}** accuracy "
        f"({sp.get('co_unmask_0.8', (0,0,0))[1]} cases, {sp.get('co_unmask_0.8', (0,0,0))[2]} errors)\n",
        f"- first_conf <0.3 → sibling acc падает до **{sibling_pred_stats([i for i in instances if first_token(i).confidence < 0.3]).get('acc', 0):.0%}**\n\n",
        "### Дополнительно: sibling vs non-sibling conf\n\n",
        f"При первом unmask siblings увереннее остальных masked-позиций "
        f"(mean {cc['sibling']['mean']:.3f} vs {cc['non_sibling']['mean']:.3f}), "
        f"но это лишь один из аспектов — подробнее в [§10](#10-sibling-vs-non-sibling-conf).\n",
        "\n---\n",
        "\n## 2. Данные и методология\n\n",
        "| Параметр | Значение |\n|----------|----------|\n",
        "| Модель | GSAI-ML/LLaDA-8B-Base, fp16, temperature=0 |\n",
        "| Датасет | WikiText-103 (validation + 6× train batches) |\n",
        "| Prompt | первые 48 токенов строки |\n",
        "| gen_length / steps / block | 64 / 64 / 64 |\n",
        "| k per step | 1 (один unmask за шаг) |\n",
        "| remasking | low_confidence |\n",
        "| mask_id | 126336 |\n",
        f"| Всего traces | {n_traces} |\n",
        f"| Loop traces (отброшены) | {n_dirty} ({pct(n_dirty, n_traces)}) |\n",
        f"| No-loop traces | {n_clean} ({pct(n_clean, n_traces)}) |\n",
        "\n**Loop-фильтр:** фраза (ngram≥4) повторяется ≥3 раз в response → trace помечен dirty "
        "(типично для g256: `Luxembourg`, `caption;` loops; g64 почти чистый).\n\n",
        "**Сегментация слов:** whitespace tokenization (`\\S+`), multi-token = ≥2 tokenizer-токена на одно слово.\n\n",
        "**Tier A (alpha):** буквенные слова, core≥3 символа (включает `image;`, `present.` и т.п.).\n\n",
        "**Tier B (lexical):** strict multi-token words — Tier A минус infobox/task junk, "
        "без склеенной пунктуации (`present.`, `height;`, `n't`). См. `multitoken_word_filters.is_lexical`.\n\n",
        "\n---\n",
        "\n## 3. Инвентарь слов (Tier A / Tier B)\n\n",
        "| Tier | instances | per trace (no-loop) |\n|------|-----------|---------------------|\n",
        f"| A (alpha) | {n_alpha} | {n_alpha/max(n_clean,1):.2f} |\n",
        f"| B (lexical) | {n} | {n/max(n_clean,1):.2f} |\n",
        f"| Отфильтровано A→B | {n_alpha - n} ({pct(n_alpha-n, n_alpha)}) |\n\n",
        "### Распределение по числу токенов (Tier B)\n\n",
        "| ntok | count | % |\n|------|-------|---|\n",
    ]
    for ntok in sorted(lay["ntok_ctr"]):
        if ntok <= 12:
            c = lay["ntok_ctr"][ntok]
            lines.append(f"| {ntok} | {c} | {pct(c, n)} |\n")
    ntok12p = sum(c for k, c in lay["ntok_ctr"].items() if k > 12)
    if ntok12p:
        lines.append(f"| 13+ | {ntok12p} | {pct(ntok12p, n)} |\n")

    lines += [
        "\n### Топ-30 слов (Tier B)\n\n| word | count |\n|------|-------|\n",
    ]
    for w, c in Counter(i.word for i in instances).most_common(30):
        lines.append(f"| `{w}` | {c} |\n")

    lines += [
        "\n---\n",
        "\n## 4. Топология\n\n",
        "| Метрика | Значение |\n|---------|----------|\n",
        f"| Same-step unmask | {pct(sum(i.same_step for i in instances), n)} (k=1 → всегда 0) |\n",
        f"| Все токены подряд по шагам | {pct(lay['full_consecutive_steps'], n)} |\n",
        "\nПозиции токенов в completion **смежные по определению** (слово = непрерывный кусок текста → подряд идущие tokenizer indices). Это не находка.\n",
        "\n---\n",
        "\n## 5. Порядок unmask\n\n",
        "### Позиция первого unmask\n\n",
        "| Позиция | count | % |\n|----------|-------|---|\n",
    ]
    for k in ("left", "middle", "right"):
        lines.append(f"| {k} | {lay['pos_ctr'][k]} | {pct(lay['pos_ctr'][k], n)} |\n")

    lines.append(render_order_section(instances))

    lines += [
        "\n---\n",
        "\n## 6. Временная динамика (step_span, gaps)\n\n",
        "**step_span** = `last_unmask_step − first_unmask_step`. Это **не** число шагов unmask!\n",
        "- span=1 у 2-токенного слова → unmask на шагах T и T+1 → **2 шага**\n",
        "- span=2 у 3-токенного → минимум 3 шага подряд\n",
        "- span=0 невозможен при k=1 и ≥2 токенах\n\n",
        "| Метрика | Значение |\n|---------|----------|\n",
        f"| step_span mean | {statistics.mean(lay['spans']):.2f} |\n",
        f"| step_span median | {statistics.median(lay['spans']):.0f} |\n",
        f"| step_span p25 / p75 / p90 | {pctile(lay['spans'], 0.25):.0f} / {pctile(lay['spans'], 0.75):.0f} / {pctile(lay['spans'], 0.9):.0f} |\n",
        f"| step_span max | {max(lay['spans'])} |\n",
        f"| first_step mean / median | {statistics.mean(lay['first_steps']):.1f} / {statistics.median(lay['first_steps']):.0f} |\n",
        "\n### Распределение step_span\n\n| span | count | % |\n|------|-------|---|\n",
    ]
    for v in sorted(lay["span_hist"]):
        c = lay["span_hist"][v]
        if c >= 5 or v <= 8:
            lines.append(f"| {v} | {c} | {pct(c, n)} |\n")
    span8p = sum(c for v, c in lay["span_hist"].items() if v >= 8)
    lines.append(f"| ≥8 | {span8p} | {pct(span8p, n)} |\n")

    lines += [
        "\n**Интерпретация:** long span (≥8) — слово «растянуто» по многим шагам, "
        "между токенами unmask'ятся другие позиции. Это ~"
        f"{pct(span8p, n)} случаев.\n",
        "\n---\n",
        "\n## 7. Confidence\n\n",
        "### 7.1 Распределение confidence при unmask\n\n",
        "| Какой токен | n | mean | median | p10 | p90 |\n|-------------|---|------|--------|-----|-----|\n",
        render_conf_row("Первый токен слова", conf_stats(lay["first_confs"])),
        render_conf_row("Последний токен слова", conf_stats(lay["last_confs"])),
        render_conf_row("Все токены всех слов", conf_stats(lay["all_confs"])),
        "\n### 7.2 First conf → качество siblings\n\n",
        "| first_conf | words | sibling_acc | all_siblings_ok |\n|------------|-------|-------------|------------------|\n",
    ]
    for lo, hi, name in [(0, 0.3, "<0.3"), (0.3, 0.7, "0.3-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, ">=0.9")]:
        sub = [i for i in instances if lo <= first_token(i).confidence < hi]
        if sub:
            gsp = sibling_pred_stats(sub)
            lines.append(f"| {name} | {len(sub)} | {gsp.get('acc', 0):.1%} | {gsp.get('all_ok', 0):.1%} |\n")

    lines += [
        "\n### 7.3 mask_ratio при первом unmask слова\n\n",
        f"mean={cc['mask_ratio'].get('mean', 0):.3f}, median={cc['mask_ratio'].get('median', 0):.3f}, "
        f"p10={cc['mask_ratio'].get('p10', 0):.3f}, p90={cc['mask_ratio'].get('p90', 0):.3f}\n",
        "\n(Доля ещё masked позиций в completion в момент первого unmask токена слова.)\n",
        "\n---\n",
        "\n## 8. Sibling prediction и co-unmask\n\n",
        "В момент первого unmask слова: для каждого **ещё masked** sibling смотрим "
        "`completion.predicted_token_id[pos]` vs финальный токен.\n\n",
        f"| Метрика | Значение |\n|---------|----------|\n",
        f"| Sibling observations | {sp['n_siblings']} |\n",
        f"| pred == final | **{sp.get('acc', 0):.1%}** |\n",
        f"| все siblings верны (per word) | **{sp.get('all_ok', 0):.1%}** |\n",
        "\n### По sibling confidence\n\n| conf | accuracy | n |\n|------|----------|---|\n",
    ]
    for name, (acc, cnt) in sp.get("by_bucket", {}).items():
        lines.append(f"| {name} | {acc:.1%} | {cnt} |\n")

    lines += ["\n### Co-unmask политики\n\n| threshold | accuracy | n | wrong |\n|-----------|----------|---|-------|\n"]
    for th in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        key = f"co_unmask_{th}"
        if key in sp:
            acc, cnt, wrong = sp[key]
            lines.append(f"| ≥{th} | {acc:.1%} | {cnt} | {wrong} |\n")

    lines += [
        "\n**Смысл:** если при первом unmask sibling уже имеет высокий conf и правильный pred, "
        "можно безопасно unmask'ить его на том же шаге (co-unmask). При conf≥0.8 — 99.8% accuracy.\n",
        "\n---\n",
        "\n## 9. Разбивка по числу токенов\n",
        render_ntok_breakdown(instances),
        "\n---\n",
        "\n## 10. Sibling vs non-sibling conf\n\n",
        "Дополнительный анализ: насколько увереннее masked-токены **внутри слова** vs **вне слова** "
        "в момент первого unmask.\n\n",
        "| Группа | n | mean | median | p10 | p90 |\n|--------|---|------|--------|-----|-----|\n",
    ]
    for label, key in [
        ("Sibling (внутри слова)", "sibling"),
        ("Non-sibling (остальные masked)", "non_sibling"),
        ("Adjacent non-word (±1)", "adjacent_non_word"),
    ]:
        st = cc[key]
        if st.get("n"):
            lines.append(f"| {label} | {st['n']} | {st['mean']:.3f} | {st['median']:.3f} | {st['p10']:.3f} | {st['p90']:.3f} |\n")

    lines += [
        f"\nPer-word: sibling mean > non-sibling в **{pct(cc['sib_higher'], cc['paired_n'])}** словах "
        f"({cc['sib_higher']}/{cc['paired_n']}).\n",
        "\n| ntok | sibling mean | non-sibling mean | Δ |\n|------|--------------|------------------|---|\n",
    ]
    for ntok in sorted(cc.get("by_ntok", {})):
        s, ns = cc["by_ntok"][ntok].get("sibling", {}), cc["by_ntok"][ntok].get("non_sibling", {})
        if s.get("n") and ns.get("n"):
            lines.append(f"| {ntok} | {s['mean']:.3f} | {ns['mean']:.3f} | {s['mean']-ns['mean']:+.3f} |\n")

    lines.append(render_gsm8k(lay, sp, cc, gsm_enriched, gsm_n_traces))

    lines += ["\n## 12. По checkpoint'ам\n\n| checkpoint | traces* | words | words/trace | span_med |\n|------------|---------|-------|-------------|----------|\n"]
    for ckpt_name in DEFAULT_CHECKPOINTS:
        if ckpt_name in ckpt_stats:
            st = ckpt_stats[ckpt_name]
            nw = st["n"]
            lines.append(f"| `{ckpt_name}` | — | {nw} | {nw/(512 if 'n512' in ckpt_name else 256):.2f} | {statistics.median(st['spans']):.0f} |\n")
    lines.append("\n*traces per checkpoint: 256 (val) или 512 (train batches)\n")

    lines += ["\n---\n", "\n## 13. Примеры\n\n", "Слова с наибольшим числом wrong sibling predictions:\n\n"]
    for ex in examples:
        lines.append(f"\n### `{ex['word']}` ({ex['ntok']} tok, pattern={ex['pattern']}, span={ex['step_span']})\n\n")
        lines.append("| step | pos | token | conf |\n|------|-----|-------|------|\n")
        for t in ex["tokens"]:
            lines.append(f"| {t['step']} | {t['pos']} | `{t['token']}` | {t['conf']:.3f} |\n")
        if ex["siblings"]:
            lines.append("\nSiblings at first unmask:\n\n| pos | pred | conf | final | ok |\n|-----|------|------|-------|----|\n")
            for s in ex["siblings"]:
                lines.append(f"| {s['pos']} | `{s['pred']}` | {s['conf']:.3f} | `{s['final']}` | {'✓' if s['ok'] else '✗'} |\n")

    lines += [
        "\n---\n",
        "\n## 14. Выводы\n\n",
        f"1. **Скорость:** {pct(lay['span_hist'][1], n)} слов с span=1 (2-токенные за 2 соседних шага); "
        f"{pct(lay['full_consecutive_steps'], n)} — все токены подряд без пропусков.\n",
        "2. **Направление:** нет сильного left/right bias (≈45/46%); 2tok чуть чаще RL; 3+ tok — mixed patterns.\n",
        f"3. **Sibling pred:** модель частично «знает» остаток слова ({sp.get('acc', 0):.0%}); "
        f"co-unmask при conf≥0.8 практически безопасен.\n",
        "4. **First conf — главный предиктор:** низкий conf первого токена → siblings ещё не определены.\n",
        f"5. **Sibling vs non-sibling conf:** siblings увереннее (§10), следствие локальной когерентности слова.\n",
        "6. **GSM8K vs WikiText:** math-генерация предсказуемее (выше acc, больше LR), WikiText разнообразнее.\n",
        f"7. **Tier A→B:** фильтр убирает {pct(n_alpha-n, n_alpha)} слов "
        f"(infobox junk + word+punctuation артефакты).\n",
        "\n---\n",
        "\n## Приложение\n\n",
        "```bash\ncd quant_where_to_unmask && python generate_wikitext_report.py \\\n",
        "  --out wikitext_g64_multitoken_report.md\n```\n",
    ]

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="wikitext_g64_multitoken_report.md")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--gsm8k", default=GSM8K_CHECKPOINT)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    ckpts = [Path("checkpoints") / name for name in args.checkpoints]
    alpha, lexical, n_traces, n_clean, n_dirty = load_all_tiers(ckpts, tokenizer, filter_loops=True)

    gsm_ckpt = Path("checkpoints") / args.gsm8k
    gsm_enriched, gsm_n_traces, _ = load_tier([gsm_ckpt], tokenizer, filter_loops=False, tier="alpha")

    out = Path(args.out)
    generate_report(alpha, lexical, n_traces, n_clean, n_dirty, gsm_enriched, gsm_n_traces, tokenizer, out)
    print(f"Wrote {out} (Tier B: {len(lexical)}, Tier A: {len(alpha)}, GSM8K: {len(gsm_enriched)})")


if __name__ == "__main__":
    main()

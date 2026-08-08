#!/usr/bin/env python3
"""Hidden similarity between masked positions at unmask moments."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from analyze_hidden_jump_at_unmask import (
    agg,
    cache_trace_hiddens,
    collect_trace_jobs,
    control_positions,
    cos_sim,
    fmt,
    load_trace,
    pct,
)

MASK_ID = 126336


def masked_positions(trace: dict, step_idx: int) -> list[int]:
    return [i for i, m in enumerate(trace["steps_trace"][step_idx]["completion"]["masked"]) if m]


def pair_mean_cos(hiddens: list[torch.Tensor], plen: int, positions: list[int], step: int) -> float | None:
    if len(positions) < 2:
        return None
    vals: list[float] = []
    for i, p in enumerate(positions):
        for q in positions[i + 1 :]:
            vals.append(cos_sim(hiddens[step][plen + p], hiddens[step][plen + q]))
    return statistics.mean(vals)


def cross_cos(hiddens: list[torch.Tensor], plen: int, pos: int, others: list[int], step: int) -> list[float]:
    return [cos_sim(hiddens[step][plen + pos], hiddens[step][plen + q]) for q in others]


def analyze_event(
    trace: dict,
    hiddens: list[torch.Tensor],
    plen: int,
    step_idx: int,
    u: dict,
    word_positions: dict[int, set[int]],
) -> dict:
    pos = u["pos_comp"]
    masked_before = [p for p in masked_positions(trace, step_idx) if p != pos]
    controls = control_positions(trace, step_idx, skip=pos)

    cross_before = cross_cos(hiddens, plen, pos, masked_before, step_idx)
    ctrl_pair_cos = pair_mean_cos(hiddens, plen, controls, step_idx)

    h_before = hiddens[step_idx][plen + pos]
    h_after = hiddens[step_idx + 1][plen + pos]

    masked_after = masked_positions(trace, step_idx + 1) if step_idx + 1 < len(trace["steps_trace"]) else []
    cross_after = [cos_sim(h_after, hiddens[step_idx + 1][plen + q]) for q in masked_after]

    # sibling masks in same multi-token word
    sib_before = [p for p in masked_before if p in word_positions and word_positions[p] & word_positions.get(pos, set())]
    sib_cross_before = cross_cos(hiddens, plen, pos, sib_before, step_idx)

    jump_cos = cos_sim(h_before, h_after)
    cross_mean_before = statistics.mean(cross_before) if cross_before else None
    cross_max_before = max(cross_before) if cross_before else None
    cross_mean_after = statistics.mean(cross_after) if cross_after else None
    sib_mean_before = statistics.mean(sib_cross_before) if sib_cross_before else None

    # drift of cross similarities step→step+1 (same other pos, if still masked)
    still = [p for p in masked_before if p in masked_after]
    cross_drift = []
    for p in still:
        cross_drift.append(
            cos_sim(hiddens[step_idx + 1][plen + pos], hiddens[step_idx + 1][plen + p])
            - cos_sim(h_before, hiddens[step_idx][plen + p])
        )

    return {
        "step": step_idx,
        "pos_comp": pos,
        "token": u["token"],
        "confidence": u["confidence"],
        "mask_ratio": trace["steps_trace"][step_idx].get("mask_ratio"),
        "in_multitoken_word": pos in word_positions,
        "n_other_masks": len(masked_before),
        "cross_mask_cos_mean": cross_mean_before,
        "cross_mask_cos_max": cross_max_before,
        "ctrl_pair_cos_mean": ctrl_pair_cos,
        "delta_cross_vs_ctrl": cross_mean_before - ctrl_pair_cos if cross_mean_before is not None and ctrl_pair_cos is not None else None,
        "sib_mask_cos_mean": sib_mean_before,
        "jump_cos": jump_cos,
        "cross_to_remaining_mean": cross_mean_after,
        "delta_cross_after_minus_before": cross_mean_after - cross_mean_before
        if cross_mean_after is not None and cross_mean_before is not None
        else None,
        "cross_drift_mean": statistics.mean(cross_drift) if cross_drift else None,
        "jump_minus_cross_before": jump_cos - cross_mean_before if cross_mean_before is not None else None,
    }


def render_report(events: list[dict], n_traces: int, ckpt: str, layer: int, out: Path) -> None:
    word_e = [e for e in events if e["in_multitoken_word"]]
    has_sib = [e for e in events if e["sib_mask_cos_mean"] is not None]

    lines = [
        "# Mask↔mask hidden similarity при unmask\n\n",
        f"Traces: **{n_traces}**, событий: **{len(events)}**, checkpoint: `{ckpt}`, layer: **{layer}**\n\n",
        "## Методология\n\n",
        "На step `s` перед unmask позиции `p`:\n",
        "- **cross_mask** = cos(h[p], h[q]) для других `[MASK]` в completion\n",
        "- **ctrl_pair** = средний cos между парами **других** still-masked (без p)\n",
        "- **jump** = cos(h[p] before → h[p] after unmask) — как в hidden jump\n",
        "- **sib_mask** = cross только к sibling `[MASK]` в том же multi-token слове\n\n",
        "Вопрос: насколько unmask-позиция похожа на **другие маски** до открытия, "
        "и меняется ли это после commit.\n\n",
        "---\n\n",
        "## 1. Похожесть на другие маски **до** unmask\n\n",
        "| Метрика | value |\n",
        "|---------|-------|\n",
        f"| mean cross_mask cos | {fmt(agg(events, 'cross_mask_cos_mean'))} |\n",
        f"| mean ctrl_pair cos | {fmt(agg(events, 'ctrl_pair_cos_mean'))} |\n",
        f"| Δ(cross − ctrl) | {fmt(agg(events, 'delta_cross_vs_ctrl'), signed=True)} |\n",
        f"| mean cross_mask cos max | {fmt(agg(events, 'cross_mask_cos_max'))} |\n",
        f"| mean sib_mask cos | {fmt(agg(has_sib, 'sib_mask_cos_mean'))} (n={len(has_sib)}) |\n\n",
    ]

    pct_higher = sum(
        1
        for e in events
        if e["delta_cross_vs_ctrl"] is not None and e["delta_cross_vs_ctrl"] > 0.01
    )
    n_delta = sum(1 for e in events if e["delta_cross_vs_ctrl"] is not None)
    lines.append(
        f"- **{pct(pct_higher, n_delta)}** событий: cross_mask **выше** ctrl_pair на >0.01\n\n"
    )

    lines += [
        "---\n\n",
        "## 2. Self jump vs cross similarity\n\n",
        "| Метрика | value |\n",
        "|---------|-------|\n",
        f"| mean jump cos (self) | {fmt(agg(events, 'jump_cos'))} |\n",
        f"| mean cross_mask cos (before) | {fmt(agg(events, 'cross_mask_cos_mean'))} |\n",
        f"| mean (jump − cross) | {fmt(agg(events, 'jump_minus_cross_before'), signed=True)} |\n\n",
        "Если jump ≪ cross ⇒ unmask меняет hidden **сильнее**, чем типичное сходство между масками.\n\n",
        "---\n\n",
        "## 3. После unmask: связь с оставшимися масками\n\n",
        "| Метрика | value |\n",
        "|---------|-------|\n",
        f"| mean cos к remaining masks (after) | {fmt(agg(events, 'cross_to_remaining_mean'))} |\n",
        f"| Δ(after − before) cross | {fmt(agg(events, 'delta_cross_after_minus_before'), signed=True)} |\n",
        f"| mean cross_drift (same q, s→s+1) | {fmt(agg(events, 'cross_drift_mean'), signed=True)} |\n\n",
        "---\n\n",
        "## 4. Multi-token слова vs остальное\n\n",
        "| subset | n | cross before | Δ cross−ctrl | sib cos | jump−cross |\n",
        "|--------|---|--------------|--------------|---------|------------|\n",
        f"| multi-token | {len(word_e)} | {fmt(agg(word_e, 'cross_mask_cos_mean'))} | "
        f"{fmt(agg(word_e, 'delta_cross_vs_ctrl'), signed=True)} | {fmt(agg([e for e in word_e if e['sib_mask_cos_mean'] is not None], 'sib_mask_cos_mean'))} | "
        f"{fmt(agg(word_e, 'jump_minus_cross_before'), signed=True)} |\n",
        f"| other | {len(events)-len(word_e)} | {fmt(agg([e for e in events if not e['in_multitoken_word']], 'cross_mask_cos_mean'))} | "
        f"{fmt(agg([e for e in events if not e['in_multitoken_word']], 'delta_cross_vs_ctrl'), signed=True)} | — | "
        f"{fmt(agg([e for e in events if not e['in_multitoken_word']], 'jump_minus_cross_before'), signed=True)} |\n\n",
        "---\n\n",
        "## 5. По mask_ratio\n\n",
        "| bucket | n | cross before | Δ cross−ctrl | jump−cross |\n",
        "|--------|---|--------------|--------------|------------|\n",
    ]
    buckets = [
        (">0.9", lambda e: e["mask_ratio"] is not None and e["mask_ratio"] > 0.9),
        ("0.5-0.9", lambda e: e["mask_ratio"] is not None and 0.5 <= e["mask_ratio"] <= 0.9),
        ("<0.5", lambda e: e["mask_ratio"] is not None and e["mask_ratio"] < 0.5),
    ]
    for name, fn in buckets:
        sub = [e for e in events if fn(e)]
        lines.append(
            f"| {name} | {len(sub)} | {fmt(agg(sub, 'cross_mask_cos_mean'))} | "
            f"{fmt(agg(sub, 'delta_cross_vs_ctrl'), signed=True)} | "
            f"{fmt(agg(sub, 'jump_minus_cross_before'), signed=True)} |\n"
        )

    lines += ["\n---\n\n", "## 6. Выводы\n\n"]
    d = agg(events, "delta_cross_vs_ctrl")
    jc = agg(events, "jump_minus_cross_before")
    if d is not None:
        if d > 0.01:
            lines.append(f"- Unmask-позиция **чуть ближе** к другим маскам, чем random mask pairs (Δ={d:+.3f}).\n")
        elif d < -0.01:
            lines.append(f"- Unmask-позиция **дальше** от других масок, чем типичная mask pair (Δ={d:+.3f}).\n")
        else:
            lines.append(f"- Cross_mask ≈ ctrl_pair (Δ={d:+.3f}) — маски **одинаково** похожи друг на друга.\n")
    if jc is not None:
        lines.append(
            f"- Self jump cos на **{fmt(jc, signed=True)}** ниже/выше cross_mask — "
            f"unmask меняет позицию {'сильнее' if jc < -0.05 else 'сопоставимо'} с «кластером масок».\n"
        )

    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--out", default="mask_similarity_at_unmask.md")
    parser.add_argument("--save-events", default="mask_similarity_events.json")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    jobs = collect_trace_jobs(Path(args.checkpoint), tokenizer, args.limit_traces, args.seed)
    print(f"traces={len(jobs)}")

    events: list[dict] = []
    for i, job in enumerate(jobs):
        trace = load_trace(job.trace_path)
        hiddens = cache_trace_hiddens(model, trace, tokenizer, args.device, args.layer)
        for step_idx, st in enumerate(trace["steps_trace"]):
            for u in st["unmasked"]:
                events.append(
                    analyze_event(trace, hiddens, job.prompt_len, step_idx, u, job.word_positions)
                )
        del hiddens
        torch.cuda.empty_cache()
        if (i + 1) % 8 == 0 or i + 1 == len(jobs):
            print(f"  {i + 1}/{len(jobs)} traces → {len(events)} events")

    if args.save_events:
        Path(args.save_events).write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(events, len(jobs), args.checkpoint, args.layer, Path(args.out))
    print(f"Wrote {args.out} ({len(events)} events)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""2-token word attention by phase, all layers."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from analyze_hidden_jump_at_unmask import completion_after_step, prompt_ids_for, rebuild_x
from analyze_multitoken_words import WordInstance, analyze_trace, load_trace
from analyze_word_attention_phases import (
    QUERY_GROUPS,
    WordPhaseSnapshot,
    analyze_snapshot,
    word_snapshots,
)
from llada_attn_capture import forward_attn, get_blocks
from multitoken_word_filters import is_lexical


@dataclass
class LayerAgg:
    n: int = 0
    out_other_masked_open: list[float] = field(default_factory=list)
    out_other_masked_masked: list[float] = field(default_factory=list)
    out_word_masked_open: list[float] = field(default_factory=list)
    out_word_masked_masked: list[float] = field(default_factory=list)
    out_word_open_open: list[float] = field(default_factory=list)
    out_word_open_masked: list[float] = field(default_factory=list)
    in_open_from_other_masked: list[float] = field(default_factory=list)
    in_masked_from_other_masked: list[float] = field(default_factory=list)


def share(open_v: float | None, masked_v: float | None) -> float | None:
    if open_v is None or masked_v is None:
        return None
    t = open_v + masked_v
    return open_v / t if t > 0 else None


def add_snap(agg: LayerAgg, snap: WordPhaseSnapshot) -> None:
    agg.n += 1
    om = snap.outgoing["other_masked"]
    wm = snap.outgoing["word_masked"]
    wo = snap.outgoing["word_open"]
    agg.out_other_masked_open.append(om["to_word_open"] or 0.0)
    agg.out_other_masked_masked.append(om["to_word_masked"] or 0.0)
    if wm["to_word_open"] is not None:
        agg.out_word_masked_open.append(wm["to_word_open"])
        agg.out_word_masked_masked.append(wm["to_word_masked"] or 0.0)
    if wo["to_word_open"] is not None:
        agg.out_word_open_open.append(wo["to_word_open"])
        agg.out_word_open_masked.append(wo["to_word_masked"] or 0.0)
    agg.in_open_from_other_masked.append(snap.incoming["word_open"]["from_other_masked"] or 0.0)
    agg.in_masked_from_other_masked.append(snap.incoming["word_masked"]["from_other_masked"] or 0.0)


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "—"


def render_table(by_layer: dict[int, LayerAgg], phase: str) -> list[str]:
    lines = [
        f"### Фаза `{phase}`\n\n",
        "| layer | n | other_m→open | other_m→masked | **other open%** | "
        "word_m→open | word_m→masked | **wm open%** | "
        "word_o→open | word_o→masked | **wo open%** | "
        "in_open←other_m | in_mask←other_m |\n",
        "|------:|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for layer in sorted(by_layer):
        a = by_layer[layer]
        if not a.n:
            continue
        o_sh = share(mean(a.out_other_masked_open), mean(a.out_other_masked_masked))
        wm_sh = share(mean(a.out_word_masked_open), mean(a.out_word_masked_masked)) if a.out_word_masked_open else None
        wo_sh = share(mean(a.out_word_open_open), mean(a.out_word_open_masked)) if a.out_word_open_open else None
        lines.append(
            f"| {layer} | {a.n} | "
            f"{fmt(mean(a.out_other_masked_open))} | {fmt(mean(a.out_other_masked_masked))} | **{fmt(o_sh)}** | "
            f"{fmt(mean(a.out_word_masked_open))} | {fmt(mean(a.out_word_masked_masked))} | **{fmt(wm_sh)}** | "
            f"{fmt(mean(a.out_word_open_open))} | {fmt(mean(a.out_word_open_masked))} | **{fmt(wo_sh)}** | "
            f"{fmt(mean(a.in_open_from_other_masked))} | {fmt(mean(a.in_masked_from_other_masked))} |\n"
        )
    lines.append("\n")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="word_attention_2tok_all_layers.md")
    parser.add_argument("--save-json", default="word_attention_2tok_all_layers.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    n_blocks = len(get_blocks(model))
    all_layers = set(range(n_blocks))

    ckpt = Path(args.checkpoint)
    rows = []
    for rf in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(l) for l in rf.open(encoding="utf-8") if l.strip())
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit_traces]

    # phase -> layer -> LayerAgg
    agg: dict[str, dict[int, LayerAgg]] = defaultdict(lambda: defaultdict(LayerAgg))
    records: list[dict] = []
    n_words = 0

    for ri, row in enumerate(rows):
        trace_path = ckpt / row["trace_path"]
        trace = load_trace(trace_path)
        plen = len(prompt_ids_for(trace, tokenizer))
        gen_length = trace["gen_length"]
        instances = [
            inst
            for inst in analyze_trace(trace_path, tokenizer, method="ws", kind_filter="alpha")
            if is_lexical(inst.word) and len(inst.positions) == 2
        ]
        if not instances:
            continue

        needed: set[int] = set()
        plans: list[tuple[WordInstance, list]] = []
        for inst in instances:
            snaps = word_snapshots(inst, trace, plen)
            plans.append((inst, snaps))
            for _, step_idx, _ in snaps:
                needed.add(step_idx)

        attn_cache: dict[int, dict[int, torch.Tensor]] = {}
        for step_idx in sorted(needed):
            if step_idx < len(trace["steps_trace"]):
                comp = trace["steps_trace"][step_idx]["completion_tokens"]
            else:
                comp = completion_after_step(trace, len(trace["steps_trace"]) - 1)
            x = rebuild_x(trace, comp, tokenizer, args.device)
            attn_cache[step_idx] = forward_attn(model, x, all_layers)

        for inst, plans_snaps in plans:
            word_pos = set(inst.positions)
            for phase, step_idx, comp in plans_snaps:
                x_ids = prompt_ids_for(trace, tokenizer) + comp
                for layer in range(n_blocks):
                    snap = analyze_snapshot(
                        attn_cache[step_idx][layer],
                        plen,
                        gen_length,
                        inst.word,
                        word_pos,
                        x_ids,
                        phase,
                        step_idx,
                    )
                    add_snap(agg[phase][layer], snap)
                    records.append(
                        {
                            "word": inst.word,
                            "phase": phase,
                            "step": step_idx,
                            "layer": layer,
                            "outgoing": snap.outgoing,
                            "incoming": snap.incoming,
                        }
                    )
            n_words += 1

        del attn_cache
        torch.cuda.empty_cache()
        if (ri + 1) % 8 == 0 or ri + 1 == len(rows):
            print(f"  {ri + 1}/{len(rows)} traces, {n_words} 2-tok words")

    lines = [
        "# Attention: только 2-токенные слова, все слои\n\n",
        f"Traces: **{len(rows)}**, 2-tok words: **{n_words}**, layers: **0..{n_blocks - 1}**\n\n",
        "## Колонки\n\n",
        "- **other open%** = other_masked → (open vs masked) внутри слова; >0.5 → больше на открытый subword\n",
        "- **wm open%** = word_masked → open vs masked sibling\n",
        "- **wo open%** = word_open → open vs masked sibling\n",
        "- **in_open←other_m** / **in_mask←other_m** = incoming на opened / masked subword от чужих масок\n\n",
        "---\n\n",
    ]
    for phase in ("before_first", "between", "all_open"):
        lines += render_table(agg[phase], phase)

    # between summary plot text
    lines += ["## Профиль `between`: word_m open% по слоям\n\n", "```\n"]
    between = agg["between"]
    for layer in range(n_blocks):
        a = between[layer]
        if not a.out_word_masked_open:
            continue
        sh = share(mean(a.out_word_masked_open), mean(a.out_word_masked_masked))
        bar = int((sh or 0) * 40)
        lines.append(f"L{layer:2d} {sh:.3f} {'█' * bar}\n")
    lines.append("```\n")

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out} ({n_words} words, {len(records)} records)")


if __name__ == "__main__":
    main()

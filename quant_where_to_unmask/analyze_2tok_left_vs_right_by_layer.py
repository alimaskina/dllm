#!/usr/bin/env python3
"""Per layer: do other tokens prefer left vs right subword (by position)?"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from analyze_hidden_jump_at_unmask import completion_after_step, prompt_ids_for, rebuild_x
from analyze_multitoken_words import analyze_trace, load_trace
from analyze_word_attention_phases import classify_queries, word_snapshots
from llada_attn_capture import attn_mass, forward_attn, get_blocks
from multitoken_word_filters import is_lexical


def mean_out(attn: torch.Tensor, queries: list[int], targets: list[int]) -> float:
    return statistics.mean(attn_mass(attn[q], targets) for q in queries)


def share(a: float, b: float) -> float:
    t = a + b
    return a / t if t > 0 else 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="word_attention_2tok_left_vs_right_by_layer.md")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    n_layers = len(get_blocks(model))
    layer_ids = set(range(n_layers))

    ckpt = Path(args.checkpoint)
    rows = []
    for rf in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(l) for l in rf.open(encoding="utf-8") if l.strip())
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit_traces]

    # layer -> phase -> list of left_share
    per: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

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
        plans: list[tuple] = []
        for inst in instances:
            for phase, step_idx, comp in word_snapshots(inst, trace, plen):
                plans.append((inst, phase, step_idx, comp))
                needed.add(step_idx)

        attn_cache: dict[int, dict[int, torch.Tensor]] = {}
        for step_idx in sorted(needed):
            if step_idx < len(trace["steps_trace"]):
                comp = trace["steps_trace"][step_idx]["completion_tokens"]
            else:
                comp = completion_after_step(trace, len(trace["steps_trace"]) - 1)
            x = rebuild_x(trace, comp, tokenizer, args.device)
            attn_cache[step_idx] = forward_attn(model, x, layer_ids)

        for inst, phase, step_idx, comp in plans:
            poss = sorted(inst.positions)
            t_left, t_right = plen + poss[0], plen + poss[1]
            x_ids = prompt_ids_for(trace, tokenizer) + comp
            groups = classify_queries(plen, gen_length, set(inst.positions), x_ids)
            others = groups["other_masked"] + groups["other_open"]
            if not others:
                continue
            for layer in layer_ids:
                attn = attn_cache[step_idx][layer]
                ml = mean_out(attn, others, [t_left])
                mr = mean_out(attn, others, [t_right])
                per[layer][phase].append(share(ml, mr))

        del attn_cache
        torch.cuda.empty_cache()
        if (ri + 1) % 16 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    def pct(xs: list[float], pred) -> float:
        return 100 * sum(1 for x in xs if pred(x)) / len(xs) if xs else 0.0

    lines = [
        "# Left vs right subword preference by layer\n\n",
        f"Traces: **{len(rows)}**, layers: **0..{n_layers - 1}**, 2-tok lexical words/phase: **109**\n\n",
        "**left_share** = attn(other→left) / (→left + →right) by **position**.\n",
        ">0.5 → prefer **left** half; <0.5 → prefer **right** half.\n\n",
    ]

    for phase in ("before_first", "between", "all_open"):
        lines.append(f"## Фаза `{phase}`\n\n")
        lines.append(
            "| layer | mean left_share | prefer left (>0.5) | prefer right (<0.5) | "
            "strong left (>0.6) | strong right (<0.4) | n |\n"
        )
        lines.append("|------:|----------------:|-------------------:|--------------------:|--------------------:|---------------------:|--:|\n")
        for layer in range(n_layers):
            shares = per[layer][phase]
            if not shares:
                continue
            lines.append(
                f"| {layer} | {statistics.mean(shares):.3f} | "
                f"{pct(shares, lambda s: s > 0.5):.1f} | {pct(shares, lambda s: s < 0.5):.1f} | "
                f"{pct(shares, lambda s: s > 0.6):.1f} | {pct(shares, lambda s: s < 0.4):.1f} | "
                f"{len(shares)} |\n"
            )
        lines.append("\n")
        # compact ascii profile of mean left_share
        lines.append("Профиль mean left_share (█ = left bias, · = right bias):\n\n```\n")
        for layer in range(n_layers):
            shares = per[layer][phase]
            if not shares:
                continue
            m = statistics.mean(shares)
            # map 0.35..0.65 to bar length
            bar_n = int(max(0, min(30, (m - 0.35) / 0.30 * 30)))
            tag = "LEFT" if m > 0.52 else ("RIGHT" if m < 0.48 else "~sym")
            lines.append(f"L{layer:2d} {m:.3f} {'█' * bar_n}{'·' * (30 - bar_n)} {tag}\n")
        lines.append("```\n\n")

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Do others prefer the first-unmasked or second-unmasked subword?"""

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


def first_second_positions(inst) -> tuple[int, int]:
    by_time = sorted(inst.tokens, key=lambda t: (t.step, t.pos_comp))
    return by_time[0].pos_comp, by_time[1].pos_comp


def pct_gt(xs: list[float], thr: float) -> float:
    return 100 * sum(x > thr for x in xs) / len(xs) if xs else 0.0


def pct_lt(xs: list[float], thr: float) -> float:
    return 100 * sum(x < thr for x in xs) / len(xs) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="word_attention_2tok_first_vs_second_opened.md")
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

    # layer -> phase -> first_share list
    per: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_diff: dict[int, list[float]] = defaultdict(list)  # all_open, different steps only
    same_step_n = 0
    diff_step_n = 0

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

        for inst in instances:
            if inst.same_step:
                same_step_n += 1
            else:
                diff_step_n += 1

        for inst, phase, step_idx, comp in plans:
            p_first, p_second = first_second_positions(inst)
            t_first, t_second = plen + p_first, plen + p_second
            x_ids = prompt_ids_for(trace, tokenizer) + comp
            groups = classify_queries(plen, gen_length, set(inst.positions), x_ids)
            others = groups["other_masked"] + groups["other_open"]
            if not others:
                continue

            for layer in layer_ids:
                attn = attn_cache[step_idx][layer]
                m_first = mean_out(attn, others, [t_first])
                m_second = mean_out(attn, others, [t_second])
                fs = share(m_first, m_second)
                per[layer][phase].append(fs)
                if phase == "all_open" and not inst.same_step:
                    per_diff[layer].append(fs)

        del attn_cache
        torch.cuda.empty_cache()
        if (ri + 1) % 16 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    lines = [
        "# Чужие токены: первый открытый vs второй открытый subword\n\n",
        f"Traces: **{len(rows)}**, layers: **0..{n_layers - 1}**, 2-tok words/phase: **109**\n\n",
        f"Слов с разными step unmask: **{diff_step_n}**, same-step co-unmask: **{same_step_n}**\n\n",
        "**first_share** = attn(other→1st unmasked) / (→1st + →2nd by unmask order).\n",
        ">0.5 → больше на **первый** открытый; <0.5 → на **второй**.\n\n",
    ]

    for phase in ("before_first", "between", "all_open"):
        lines.append(f"## Фаза `{phase}`\n\n")
        lines.append("| layer | mean first_share | prefer 1st (>0.5) | prefer 2nd (<0.5) | strong 1st (>0.6) | strong 2nd (<0.4) | n |\n")
        lines.append("|------:|-----------------:|------------------:|------------------:|------------------:|------------------:|--:|\n")
        for layer in range(n_layers):
            shares = per[layer][phase]
            if not shares:
                continue
            lines.append(
                f"| {layer} | {statistics.mean(shares):.3f} | "
                f"{pct_gt(shares, 0.5):.1f} | {pct_lt(shares, 0.5):.1f} | "
                f"{pct_gt(shares, 0.6):.1f} | {pct_lt(shares, 0.4):.1f} | {len(shares)} |\n"
            )
        lines.append("\n")
        if phase == "between":
            lines.append(
                "> В `between` первый unmasked = единственный REAL, второй = MASK → "
                "first_share ≈ open_share.\n\n"
            )

    lines.append("## Только слова с разным step unmask (`all_open`)\n\n")
    lines.append("| layer | mean first_share | prefer 1st | prefer 2nd | n |\n")
    lines.append("|------:|-----------------:|---------:|---------:|--:|\n")
    for layer in range(n_layers):
        shares = per_diff[layer]
        if not shares:
            continue
        lines.append(
            f"| {layer} | {statistics.mean(shares):.3f} | "
            f"{pct_gt(shares, 0.5):.1f} | {pct_lt(shares, 0.5):.1f} | {len(shares)} |\n"
        )
    lines.append("\n")

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

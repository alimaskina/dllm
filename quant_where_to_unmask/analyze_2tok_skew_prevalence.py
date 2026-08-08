#!/usr/bin/env python3
"""Per-word skew: do other tokens prefer one subword over the other?"""

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

MASK_ID = 126336


def mean_out(attn: torch.Tensor, queries: list[int], targets: list[int]) -> float:
    return statistics.mean(attn_mass(attn[q], targets) for q in queries)


def share(a: float, b: float) -> float:
    t = a + b
    return a / t if t > 0 else 0.5


def pct_gt(dev: list[float], thr: float) -> float:
    return 100 * sum(d > thr for d in dev) / len(dev) if dev else 0.0


def summarize(shares: list[float], label: str) -> list[str]:
    if not shares:
        return [f"- {label}: n=0\n"]
    dev = [abs(s - 0.5) for s in shares]
    lines = [
        f"### {label} (n={len(shares)})\n\n",
        f"- mean share: **{statistics.mean(shares):.3f}** (0.5 = symmetric)\n",
        f"- mean |share−0.5|: **{statistics.mean(dev):.3f}**\n",
        f"- median |share−0.5|: **{statistics.median(dev):.3f}**\n",
    ]
    for thr in (0.05, 0.10, 0.15, 0.20):
        frac = sum(d > thr for d in dev) / len(dev)
        lines.append(f"- |share−0.5| > {thr:.2f}: **{100 * frac:.1f}%** слов\n")
    lines.append("\n")
    return lines


def render_layer_tables(
    per_layer: dict[int, dict[str, list[float]]],
    n_layers: int,
) -> list[str]:
    lines = [
        "# Per-word skew по всем слоям (чужие → subword0 vs subword1)\n\n",
        "Колонки: mean |Δ|, % слов с |share−0.5| > 0.10 / > 0.15 / > 0.20\n\n",
    ]
    for phase in ("before_first", "between", "all_open"):
        lines.append(f"## Фаза `{phase}`\n\n")
        lines.append("| layer | mean share | mean |Δ| | >10% | >15% | >20% | n |\n")
        lines.append("|------:|-----------:|-------:|-----:|-----:|-----:|--:|\n")
        for layer in range(n_layers):
            shares = per_layer[layer].get(f"other_pos_{phase}", [])
            if not shares:
                continue
            dev = [abs(s - 0.5) for s in shares]
            lines.append(
                f"| {layer} | {statistics.mean(shares):.3f} | {statistics.mean(dev):.3f} | "
                f"{pct_gt(dev, 0.10):.1f} | {pct_gt(dev, 0.15):.1f} | {pct_gt(dev, 0.20):.1f} | {len(shares)} |\n"
            )
        lines.append("\n")
        if phase == "between":
            lines.append("### open vs masked (between only)\n\n")
            lines.append("| layer | mean open_share | mean |Δ| | >10% | >15% | >20% | n |\n")
            lines.append("|------:|----------------:|-------:|-----:|-----:|-----:|--:|\n")
            for layer in range(n_layers):
                shares = per_layer[layer].get("other_open_vs_mask_between", [])
                if not shares:
                    continue
                dev = [abs(s - 0.5) for s in shares]
                lines.append(
                    f"| {layer} | {statistics.mean(shares):.3f} | {statistics.mean(dev):.3f} | "
                    f"{pct_gt(dev, 0.10):.1f} | {pct_gt(dev, 0.15):.1f} | {pct_gt(dev, 0.20):.1f} | {len(shares)} |\n"
                )
            lines.append("\n")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="word_attention_2tok_skew_prevalence.md")
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
    if args.all_layers:
        layer_ids = set(range(n_layers))
    elif args.layer is not None:
        layer_ids = {args.layer}
    else:
        layer_ids = {16}

    ckpt = Path(args.checkpoint)
    rows = []
    for rf in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(l) for l in rf.open(encoding="utf-8") if l.strip())
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit_traces]

    # layer -> metric -> per-word shares
    per_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    examples_skewed: list[tuple[float, str, str, int]] = []

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
            word_pos = set(inst.positions)
            poss = sorted(inst.positions)
            t0, t1 = plen + poss[0], plen + poss[1]
            x_ids = prompt_ids_for(trace, tokenizer) + comp
            groups = classify_queries(plen, gen_length, word_pos, x_ids)
            others = groups["other_masked"] + groups["other_open"]
            if not others:
                continue

            open_t, mask_t = [], []
            for rel in poss:
                abs_p = plen + rel
                (open_t if x_ids[abs_p] != MASK_ID else mask_t).append(abs_p)

            for layer in layer_ids:
                attn = attn_cache[step_idx][layer]
                m0 = mean_out(attn, others, [t0])
                m1 = mean_out(attn, others, [t1])
                pos_share = share(m0, m1)
                per_layer[layer][f"other_pos_{phase}"].append(pos_share)

                if open_t and mask_t:
                    mo = mean_out(attn, others, open_t)
                    mm = mean_out(attn, others, mask_t)
                    open_share = share(mo, mm)
                    key = f"other_open_vs_mask_{phase}"
                    per_layer[layer][key].append(open_share)

                dev = abs(pos_share - 0.5)
                if dev > 0.15 and layer == 16:
                    examples_skewed.append((dev, inst.word, phase, layer))

        del attn_cache
        torch.cuda.empty_cache()
        if (ri + 1) % 16 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    examples_skewed.sort(reverse=True)

    if args.all_layers:
        lines = render_layer_tables(per_layer, n_layers)
        lines.insert(
            1,
            f"Traces: **{len(rows)}**, layers: **0..{n_layers - 1}**, 2-tok lexical words per phase: **109**\n\n",
        )
    else:
        layer = next(iter(layer_ids))
        per_word = per_layer[layer]
        lines = [
            "# Сколько 2-tok слов с перекосом attention от «чужих» токенов\n\n",
            f"Traces: **{len(rows)}**, layer: **{layer}**\n\n",
            "Per-word share = attn(other→subword0) / (attn→subword0 + attn→subword1).\n",
            "0.5 = симметрия; перекос = share заметно отличается от 0.5.\n\n",
        ]
        for phase in ("before_first", "between", "all_open"):
            lines.append(f"## Фаза `{phase}`\n\n")
            lines += summarize(per_word[f"other_pos_{phase}"], "По позиции (subword0 vs subword1)")
            if per_word.get(f"other_open_vs_mask_{phase}"):
                lines += summarize(
                    per_word[f"other_open_vs_mask_{phase}"],
                    "По состоянию (open vs masked subword)",
                )

        lines.append("## Примеры сильного перекоса (|pos_share−0.5| > 0.15, L16)\n\n")
        for dev, word, phase, _ in examples_skewed[:15]:
            lines.append(f"- `{word}` ({phase}): |Δ|={dev:.2f}\n")

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

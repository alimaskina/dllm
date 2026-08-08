#!/usr/bin/env python3
"""Do other completion tokens prefer left vs right subword of 2-tok words?"""

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
from llada_attn_capture import attn_mass, forward_attn
from multitoken_word_filters import is_lexical

MASK_ID = 126336


def share(a: float, b: float) -> float | None:
    t = a + b
    return a / t if t > 0 else None


def mean_out(attn: torch.Tensor, queries: list[int], target: list[int]) -> float | None:
    if not queries or not target:
        return None
    return statistics.mean(attn_mass(attn[q], target) for q in queries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="16,31")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="word_attention_2tok_other_prefers_role.md")
    args = parser.parse_args()

    layer_ids = {int(x.strip()) for x in args.layers.split(",")}
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    ckpt = Path(args.checkpoint)
    rows = []
    for rf in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(l) for l in rf.open(encoding="utf-8") if l.strip())
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit_traces]

    # phase -> layer -> qgroup -> metric -> list
    agg: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
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
        plans = []
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
            attn_cache[step_idx] = forward_attn(model, x, layer_ids)

        for inst, snaps in plans:
            word_pos = set(inst.positions)
            poss = sorted(inst.positions)
            left_t, right_t = [plen + poss[0]], [plen + poss[1]]
            for phase, step_idx, comp in snaps:
                x_ids = prompt_ids_for(trace, tokenizer) + comp
                groups = classify_queries(plen, gen_length, word_pos, x_ids)
                for layer in layer_ids:
                    attn = attn_cache[step_idx][layer]
                    for qg in ("other_masked", "other_open", "word_open", "word_masked"):
                        queries = groups[qg]
                        if not queries:
                            continue
                        tl = mean_out(attn, queries, left_t)
                        tr = mean_out(attn, queries, right_t)
                        if tl is None or tr is None:
                            continue
                        bucket = agg[phase][layer][qg]
                        bucket["to_left"].append(tl)
                        bucket["to_right"].append(tr)
                        bucket["left_share"].append(share(tl, tr) or 0.0)
            n_words += 1

        del attn_cache
        torch.cuda.empty_cache()
        if (ri + 1) % 16 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    def m(xs: list[float]) -> float:
        return statistics.mean(xs)

    lines = [
        "# Prefer left vs right subword: кто смотрит?\n\n",
        f"Traces: **{len(rows)}**, 2-tok lexical words: **{n_words}**\n\n",
        "Для каждой query-группы: mean attn(queries → left) vs → right.\n",
        "**left_share** = left / (left + right); >0.5 → предпочитают **левый** subword.\n\n",
    ]

    for phase in ("before_first", "between", "all_open"):
        lines.append(f"## Фаза `{phase}`\n\n")
        for layer in sorted(layer_ids):
            lines.append(f"### Layer {layer}\n\n")
            lines.append("| query group | → left | → right | **left_share** | n |\n")
            lines.append("|-------------|-------:|--------:|---------------:|--:|\n")
            for qg in ("other_masked", "other_open", "word_open", "word_masked"):
                b = agg[phase][layer].get(qg)
                if not b or not b["to_left"]:
                    continue
                lines.append(
                    f"| {qg} | {m(b['to_left']):.4f} | {m(b['to_right']):.4f} | "
                    f"**{m(b['left_share']):.4f}** | {len(b['to_left'])} |\n"
                )
            lines.append("\n")

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

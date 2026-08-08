#!/usr/bin/env python3
"""Per-token (left/right) within-word attention for fully opened 2-tok words."""

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
from analyze_word_attention_phases import word_snapshots
from llada_attn_capture import forward_attn, get_blocks
from multitoken_word_filters import is_lexical


def share(sib: float, self_: float) -> float | None:
    t = sib + self_
    return sib / t if t > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="16,31")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="word_attention_2tok_all_open_per_role.md")
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

    # layer -> role -> metric -> list
    agg: dict[int, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    n_words = 0

    for ri, row in enumerate(rows):
        trace_path = ckpt / row["trace_path"]
        trace = load_trace(trace_path)
        plen = len(prompt_ids_for(trace, tokenizer))
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
            for phase, step_idx, _ in snaps:
                if phase == "all_open":
                    needed.add(step_idx)

        if not needed:
            continue

        attn_cache: dict[int, dict[int, torch.Tensor]] = {}
        for step_idx in sorted(needed):
            if step_idx < len(trace["steps_trace"]):
                comp = trace["steps_trace"][step_idx]["completion_tokens"]
            else:
                comp = completion_after_step(trace, len(trace["steps_trace"]) - 1)
            x = rebuild_x(trace, comp, tokenizer, args.device)
            attn_cache[step_idx] = forward_attn(model, x, layer_ids)

        for inst, snaps in plans:
            poss = sorted(inst.positions)
            left, right = plen + poss[0], plen + poss[1]
            for phase, step_idx, _ in snaps:
                if phase != "all_open":
                    continue
                for layer in layer_ids:
                    attn = attn_cache[step_idx][layer]
                    ls = float(attn[left, left])
                    lr = float(attn[left, right])
                    rs = float(attn[right, right])
                    rl = float(attn[right, left])
                    a = agg[layer]["left"]
                    b = agg[layer]["right"]
                    a["self"].append(ls)
                    a["sibling"].append(lr)
                    a["sib_share"].append(share(lr, ls) or 0.0)
                    b["self"].append(rs)
                    b["sibling"].append(rl)
                    b["sib_share"].append(share(rl, rs) or 0.0)
            n_words += 1

        del attn_cache
        torch.cuda.empty_cache()
        if (ri + 1) % 16 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    def m(xs: list[float]) -> float:
        return statistics.mean(xs)

    lines = [
        "# 2-tok words, фаза `all_open`: per-token attention внутри слова\n\n",
        f"Traces: **{len(rows)}**, words: **{n_words}**, layers: **{sorted(layer_ids)}**\n\n",
        "Для каждого токена слова (left / right по позиции):\n",
        "- **self** = attn[q→q]\n",
        "- **sibling** = attn[q→other subword]\n",
        "- **sib%** = sibling / (sibling + self)\n\n",
        "> 0.5 → больше mass на sibling, чем на self\n\n",
    ]
    for layer in sorted(layer_ids):
        lines.append(f"## Layer {layer}\n\n")
        lines.append("| role | self | sibling | **sib%** | n |\n")
        lines.append("|------|-----:|--------:|---------:|--:|\n")
        for role in ("left", "right"):
            d = agg[layer][role]
            lines.append(
                f"| {role} | {m(d['self']):.4f} | {m(d['sibling']):.4f} | "
                f"**{m(d['sib_share']):.4f}** | {len(d['self'])} |\n"
            )
        left_s = m(agg[layer]["left"]["sib_share"])
        right_s = m(agg[layer]["right"]["sib_share"])
        lines.append(f"\nΔ sib% (right − left) = **{right_s - left_s:+.4f}**\n\n")

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} ({n_words} words)")


if __name__ == "__main__":
    main()

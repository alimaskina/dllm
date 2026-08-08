#!/usr/bin/env python3
"""Do already-open token hiddens keep drifting step-to-step?"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from analyze_hidden_jump_at_unmask import (
    cache_trace_hiddens,
    cos_sim,
    l2_dist,
    prompt_ids_for,
)

MASK_ID = 126336


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="hidden_drift_after_open.md")
    args = parser.parse_args()

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

    jump_cos, jump_l2 = [], []
    mask_cos, mask_l2 = [], []
    open_cos, open_l2 = [], []
    # also: first step after a position opens (open@s+1 was mask@s) already in jump
    # open_later: open for >=1 step already before this transition
    open_fresh_cos, open_fresh_l2 = [], []  # just opened previous step: REAL@s and REAL@s+1, was MASK@s-1
    open_old_cos, open_old_l2 = [], []

    for ri, row in enumerate(rows):
        import gzip

        with gzip.open(ckpt / row["trace_path"], "rt", encoding="utf-8") as f:
            trace = json.load(f)
        plen = len(prompt_ids_for(trace, tokenizer))
        steps = trace["steps_trace"]
        hiddens = cache_trace_hiddens(model, trace, tokenizer, args.device, layer=args.layer)
        # hiddens[t] aligns with steps[t] input; hiddens[len] = after last

        for s in range(len(steps) - 1):
            c0 = steps[s]["completion_tokens"]
            c1 = steps[s + 1]["completion_tokens"]
            for p in range(len(c0)):
                abs_p = plen + p
                a = hiddens[s][abs_p]
                b = hiddens[s + 1][abs_p]
                cs, ls = cos_sim(a, b), l2_dist(a, b)
                was_mask = c0[p] == MASK_ID
                now_open = c1[p] != MASK_ID
                still_mask = c0[p] == MASK_ID and c1[p] == MASK_ID
                already_open = c0[p] != MASK_ID and c1[p] != MASK_ID

                if was_mask and now_open:
                    jump_cos.append(cs)
                    jump_l2.append(ls)
                elif still_mask:
                    mask_cos.append(cs)
                    mask_l2.append(ls)
                elif already_open:
                    open_cos.append(cs)
                    open_l2.append(ls)
                    # fresh vs old: was this position opened at s-1?
                    if s == 0:
                        open_fresh_cos.append(cs)
                        open_fresh_l2.append(ls)
                    else:
                        c_prev = steps[s - 1]["completion_tokens"]
                        if c_prev[p] == MASK_ID:
                            open_fresh_cos.append(cs)
                            open_fresh_l2.append(ls)
                        else:
                            open_old_cos.append(cs)
                            open_old_l2.append(ls)

        del hiddens
        torch.cuda.empty_cache()
        if (ri + 1) % 8 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    def m(xs):
        return statistics.mean(xs) if xs else float("nan")

    lines = [
        "# Hidden drift: jump vs still-MASK vs already-OPEN\n\n",
        f"Traces: **{len(rows)}**, layer: **{args.layer}**\n\n",
        "На каждом переходе step `s → s+1` для каждой completion-позиции:\n",
        "- **jump** — MASK→REAL на этом step\n",
        "- **still MASK** — осталась MASK\n",
        "- **already OPEN** — уже REAL на `s` и на `s+1`\n",
        "- **fresh OPEN** — открыли на `s-1`, сейчас второй step как REAL\n",
        "- **old OPEN** — REAL уже ≥2 steps\n\n",
        "| group | mean cos | mean L2 | n |\n",
        "|-------|---------:|--------:|--:|\n",
        f"| jump (MASK→REAL) | {m(jump_cos):.3f} | {m(jump_l2):.1f} | {len(jump_cos)} |\n",
        f"| still MASK | {m(mask_cos):.3f} | {m(mask_l2):.1f} | {len(mask_cos)} |\n",
        f"| already OPEN | {m(open_cos):.3f} | {m(open_l2):.1f} | {len(open_cos)} |\n",
        f"| fresh OPEN (1 step after unmask) | {m(open_fresh_cos):.3f} | {m(open_fresh_l2):.1f} | {len(open_fresh_cos)} |\n",
        f"| old OPEN (≥2 steps) | {m(open_old_cos):.3f} | {m(open_old_l2):.1f} | {len(open_old_cos)} |\n\n",
        f"ratio L2 jump / already OPEN = **{m(jump_l2) / m(open_l2):.2f}×**\n\n",
        f"ratio L2 already OPEN / still MASK = **{m(open_l2) / m(mask_l2):.2f}×**\n\n",
    ]
    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print("".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

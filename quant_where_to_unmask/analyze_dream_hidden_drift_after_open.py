#!/usr/bin/env python3
"""Dream: hidden drift jump vs still-MASK vs already-OPEN, all layers."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from analyze_hidden_jump_at_unmask import cos_sim, l2_dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_dream_wikitext_fp16_g64_n64")
    parser.add_argument("--model", default="Dream-org/Dream-v0-Base-7B")
    parser.add_argument("--limit-traces", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="dream_hidden_drift_after_open_all_layers.md")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mask_id = tokenizer.mask_token_id
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    n_hs = len(model.model.layers) + 1  # embed + blocks

    ckpt = Path(args.checkpoint)
    rows = [json.loads(l) for l in open(ckpt / "rank0.jsonl") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit_traces]

    agg: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for ri, row in enumerate(rows):
        with gzip.open(ckpt / row["trace_path"], "rt", encoding="utf-8") as f:
            trace = json.load(f)
        plen = len(tokenizer(trace["prompt_text"], add_special_tokens=False)["input_ids"])
        steps = trace["steps_trace"]
        comps = [st["completion_tokens"] for st in steps]
        final = comps[-1][:]
        for u in steps[-1]["unmasked"]:
            final[u["pos_comp"]] = u["token_id"]
        comps.append(final)

        hiddens = []
        with torch.inference_mode():
            for comp in comps:
                x = torch.tensor(
                    [tokenizer(trace["prompt_text"], add_special_tokens=False)["input_ids"] + comp],
                    device=args.device,
                )
                hs = model(x, attention_mask=None, output_hidden_states=True).hidden_states
                hiddens.append([h[0].float().cpu() for h in hs])

        for s in range(len(steps) - 1):
            c0, c1 = steps[s]["completion_tokens"], steps[s + 1]["completion_tokens"]
            for p in range(len(c0)):
                was_mask = c0[p] == mask_id
                now_real = c1[p] != mask_id
                still_mask = was_mask and c1[p] == mask_id
                already_open = (not was_mask) and now_real
                if was_mask and now_real:
                    group = "jump"
                elif still_mask:
                    group = "mask"
                elif already_open:
                    group = (
                        "fresh"
                        if s > 0 and steps[s - 1]["completion_tokens"][p] == mask_id
                        else "old"
                    )
                else:
                    continue
                abs_p = plen + p
                for layer in range(n_hs):
                    a, b = hiddens[s][layer][abs_p], hiddens[s + 1][layer][abs_p]
                    agg[layer][f"{group}_cos"].append(cos_sim(a, b))
                    agg[layer][f"{group}_l2"].append(l2_dist(a, b))

        del hiddens
        torch.cuda.empty_cache()
        if (ri + 1) % 4 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    def m(xs):
        return statistics.mean(xs) if xs else None

    def fmt(v):
        return f"{v:.3f}" if v is not None else "—"

    lines = [
        "# Dream: hidden drift after open (all layers)\n\n",
        f"Traces: **{len(rows)}**, layers: **0..{n_hs-1}**\n\n",
        "| layer | jump cos | mask cos | fresh cos | old cos | jump L2 | mask L2 | fresh L2 | old L2 | jump/old | old/mask |\n",
        "|------:|---------:|---------:|----------:|--------:|--------:|--------:|---------:|-------:|---------:|---------:|\n",
    ]
    for layer in range(n_hs):
        a = agg[layer]
        j, o, mk = m(a["jump_l2"]), m(a["old_l2"]), m(a["mask_l2"])
        lines.append(
            f"| {layer} | {fmt(m(a['jump_cos']))} | {fmt(m(a['mask_cos']))} | "
            f"{fmt(m(a['fresh_cos']))} | {fmt(m(a['old_cos']))} | "
            f"{fmt(j)} | {fmt(mk)} | {fmt(m(a['fresh_l2']))} | {fmt(o)} | "
            f"**{fmt(j/o if j and o else None)}×** | {fmt(o/mk if o and mk else None)}× |\n"
        )
    lines.append("\n## jump/old L2 profile\n\n```\n")
    for layer in range(n_hs):
        j, o = m(agg[layer]["jump_l2"]), m(agg[layer]["old_l2"])
        if not j or not o:
            continue
        r = j / o
        lines.append(f"L{layer:2d} {r:5.2f}x {'█' * int(min(40, r * 4))}\n")
    lines.append("```\n")
    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hidden drift after open: jump vs MASK vs OPEN, all layers."""

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

from analyze_hidden_jump_at_unmask import cos_sim, l2_dist, prompt_ids_for, rebuild_x
from llada_attn_capture import get_blocks

MASK_ID = 126336


def cache_all_layer_hiddens(model, trace, tokenizer, device) -> list[list[torch.Tensor]]:
    """Return hiddens_by_step[step][layer] as [T, D] CPU float tensors.
    layers: 0=embed ... n_blocks=after last block.
    """
    steps = trace["steps_trace"]
    n_layers = len(get_blocks(model)) + 1  # embed + blocks
    # step inputs + final after last
    comps = [st["completion_tokens"] for st in steps]
    # final
    final = comps[-1][:]
    for u in steps[-1]["unmasked"]:
        final[u["pos_comp"]] = u["token_id"]
    comps.append(final)

    out: list[list[torch.Tensor]] = []
    with torch.inference_mode():
        for comp in comps:
            x = rebuild_x(trace, comp, tokenizer, device)
            hs = model(x, attention_mask=torch.ones_like(x), output_hidden_states=True).hidden_states
            # hs: tuple len = n_blocks+1
            out.append([h[0].float().cpu() for h in hs])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="hidden_drift_after_open_all_layers.md")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    n_hs = len(get_blocks(model)) + 1  # 0..32

    ckpt = Path(args.checkpoint)
    rows = []
    for rf in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(l) for l in rf.open(encoding="utf-8") if l.strip())
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit_traces]

    # layer -> group -> list
    agg: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for ri, row in enumerate(rows):
        with gzip.open(ckpt / row["trace_path"], "rt", encoding="utf-8") as f:
            trace = json.load(f)
        plen = len(prompt_ids_for(trace, tokenizer))
        steps = trace["steps_trace"]
        hiddens = cache_all_layer_hiddens(model, trace, tokenizer, args.device)

        for s in range(len(steps) - 1):
            c0 = steps[s]["completion_tokens"]
            c1 = steps[s + 1]["completion_tokens"]
            for p in range(len(c0)):
                was_mask = c0[p] == MASK_ID
                now_real = c1[p] != MASK_ID
                still_mask = was_mask and c1[p] == MASK_ID
                already_open = (not was_mask) and now_real

                if was_mask and now_real:
                    group = "jump"
                elif still_mask:
                    group = "mask"
                elif already_open:
                    if s > 0 and steps[s - 1]["completion_tokens"][p] == MASK_ID:
                        group = "fresh"
                    else:
                        group = "old"
                else:
                    continue

                abs_p = plen + p
                for layer in range(n_hs):
                    a = hiddens[s][layer][abs_p]
                    b = hiddens[s + 1][layer][abs_p]
                    agg[layer][f"{group}_cos"].append(cos_sim(a, b))
                    agg[layer][f"{group}_l2"].append(l2_dist(a, b))

        del hiddens
        torch.cuda.empty_cache()
        if (ri + 1) % 4 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    def m(xs: list[float]) -> float | None:
        return statistics.mean(xs) if xs else None

    def fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "—"

    lines = [
        "# Hidden drift after open: all layers\n\n",
        f"Traces: **{len(rows)}**, layers: **0..{n_hs - 1}** (0=embed, {n_hs - 1}=last)\n\n",
        "- **jump** MASK→REAL\n",
        "- **mask** still MASK\n",
        "- **fresh** OPEN 1 step after unmask\n",
        "- **old** OPEN ≥2 steps\n\n",
        "| layer | jump cos | mask cos | fresh cos | old cos | jump L2 | mask L2 | fresh L2 | old L2 | jump/old L2 | old/mask L2 |\n",
        "|------:|---------:|---------:|----------:|--------:|--------:|--------:|---------:|-------:|------------:|------------:|\n",
    ]
    for layer in range(n_hs):
        a = agg[layer]
        j_l2, o_l2, m_l2 = m(a["jump_l2"]), m(a["old_l2"]), m(a["mask_l2"])
        ratio_jo = (j_l2 / o_l2) if j_l2 and o_l2 else None
        ratio_om = (o_l2 / m_l2) if o_l2 and m_l2 else None
        lines.append(
            f"| {layer} | {fmt(m(a['jump_cos']))} | {fmt(m(a['mask_cos']))} | "
            f"{fmt(m(a['fresh_cos']))} | {fmt(m(a['old_cos']))} | "
            f"{fmt(j_l2)} | {fmt(m_l2)} | {fmt(m(a['fresh_l2']))} | {fmt(o_l2)} | "
            f"**{fmt(ratio_jo)}×** | {fmt(ratio_om)}× |\n"
        )

    lines += [
        "\n## Профиль jump/old L2\n\n```\n",
    ]
    for layer in range(n_hs):
        j_l2, o_l2 = m(agg[layer]["jump_l2"]), m(agg[layer]["old_l2"])
        if not j_l2 or not o_l2:
            continue
        r = j_l2 / o_l2
        bar = int(min(40, r * 4))
        lines.append(f"L{layer:2d} {r:5.2f}x {'█' * bar}\n")
    lines.append("```\n")

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

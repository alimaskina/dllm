#!/usr/bin/env python3
"""Unified 2-tok suite for LLaDA or Dream: unmask + left/right attn + logit lens."""

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

from analyze_multitoken_words import analyze_trace, order_pattern
from analyze_word_attention_phases import word_snapshots
from multitoken_word_filters import is_lexical


def load_trace(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def prompt_ids(trace, tokenizer):
    return tokenizer(trace["prompt_text"], add_special_tokens=False)["input_ids"]


def rebuild_x(trace, comp, tokenizer, device):
    return torch.tensor([prompt_ids(trace, tokenizer) + comp], dtype=torch.long, device=device)


def mean_out(attn, queries, targets):
    if not queries or not targets:
        return None
    t = torch.tensor(targets, dtype=torch.long)
    return statistics.mean(float(attn[q][t].sum()) for q in queries)


def share(a, b):
    t = a + b
    return a / t if t > 0 else 0.5


def pct(xs, pred):
    return 100 * sum(1 for x in xs if pred(x)) / len(xs) if xs else 0.0


def classify(plen, gen_length, word_positions, x_ids, mask_id):
    groups = {g: [] for g in ("other_masked", "other_open", "word_masked", "word_open")}
    for rel in range(gen_length):
        abs_p = plen + rel
        tid = int(x_ids[abs_p])
        in_word = rel in word_positions
        if tid == mask_id:
            key = "word_masked" if in_word else "other_masked"
        else:
            key = "word_open" if in_word else "other_open"
        groups[key].append(abs_p)
    return groups


def get_lm_head(model, family: str):
    if family == "dream":
        return model.lm_head
    # LLaDA
    return model.model.transformer.ff_out


def n_transformer_layers(model, family: str) -> int:
    if family == "dream":
        return len(model.model.layers)
    return len(model.model.transformer.blocks)


def forward_pack(model, x, family: str, layer_ids: set[int], device: str):
    """Return attn[layer]=[T,T], hid[layer]=[T,D] after block `layer`."""
    with torch.inference_mode():
        if family == "dream":
            out = model(x, attention_mask=None, output_attentions=True, output_hidden_states=True)
            n = n_transformer_layers(model, family)
            attn = {
                lid: out.attentions[lid][0].float().mean(0).cpu()
                for lid in layer_ids
                if lid < n
            }
            hid = {
                lid: out.hidden_states[lid + 1][0].float().cpu()
                for lid in layer_ids
                if lid < n
            }
            return attn, hid

        # LLaDA: attentions via hook + hidden_states
        from llada_attn_capture import forward_attn

        attn = forward_attn(model, x, layer_ids)
        out = model(x, attention_mask=torch.ones_like(x), output_hidden_states=True)
        # LLaDA hidden_states: 0=embed, 1..32 after blocks
        hid = {
            lid: out.hidden_states[lid + 1][0].float().cpu()
            for lid in layer_ids
            if lid + 1 < len(out.hidden_states)
        }
        return attn, hid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["dream", "llada"], required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--layers", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.family == "dream":
        args.model = args.model or "Dream-org/Dream-v0-Base-7B"
        args.checkpoint = args.checkpoint or "checkpoints/results_dream_wikitext_fp16_g64_n64"
        args.layers = args.layers or "0,7,14,21,27"
        args.out = args.out or "dream_wikitext_multitoken_suite.md"
        dtype = torch.bfloat16
        load_kw = {"attn_implementation": "eager"}
    else:
        args.model = args.model or "GSAI-ML/LLaDA-8B-Base"
        args.checkpoint = args.checkpoint or "checkpoints/results_wikitext_fp16_g64_n256"
        args.layers = args.layers or "0,7,14,21,31"
        args.out = args.out or "llada_wikitext_multitoken_suite.md"
        dtype = torch.float16
        load_kw = {}

    layer_ids = {int(x) for x in args.layers.split(",")}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.family == "llada":
        mask_id = tokenizer.mask_token_id or 126336
    else:
        mask_id = tokenizer.mask_token_id
    assert mask_id is not None
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device,
        **load_kw,
    )
    model.eval()
    lm_head = get_lm_head(model, args.family)
    head_dtype = next(lm_head.parameters()).dtype

    ckpt = Path(args.checkpoint)
    rows = []
    for rf in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(l) for l in rf.open(encoding="utf-8") if l.strip())
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit_traces]

    all_inst = []
    for row in rows:
        all_inst.extend(
            i
            for i in analyze_trace(ckpt / row["trace_path"], tokenizer, method="ws", kind_filter="alpha")
            if is_lexical(i.word)
        )
    two = [i for i in all_inst if len(i.positions) == 2]

    left_shares = defaultdict(lambda: defaultdict(list))
    first_shares = defaultdict(lambda: defaultdict(list))
    within = defaultdict(lambda: defaultdict(list))
    probe = defaultdict(lambda: defaultdict(list))

    for ri, row in enumerate(rows):
        trace = load_trace(ckpt / row["trace_path"])
        plen = len(prompt_ids(trace, tokenizer))
        gen_length = trace["gen_length"]
        instances = [
            i
            for i in analyze_trace(ckpt / row["trace_path"], tokenizer, method="ws", kind_filter="alpha")
            if is_lexical(i.word) and len(i.positions) == 2
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

        cache_attn: dict = {}
        cache_hid: dict = {}
        for step_idx in sorted(needed):
            if step_idx < len(trace["steps_trace"]):
                comp = trace["steps_trace"][step_idx]["completion_tokens"]
            else:
                comp = trace["steps_trace"][-1]["completion_tokens"][:]
                for u in trace["steps_trace"][-1]["unmasked"]:
                    comp[u["pos_comp"]] = u["token_id"]
            x = rebuild_x(trace, comp, tokenizer, args.device)
            cache_attn[step_idx], cache_hid[step_idx] = forward_pack(
                model, x, args.family, layer_ids, args.device
            )
            torch.cuda.empty_cache()

        for inst, snaps in plans:
            poss = sorted(inst.positions)
            by_time = sorted(inst.tokens, key=lambda t: (t.step, t.pos_comp))
            p_first, p_second = by_time[0].pos_comp, by_time[1].pos_comp
            for phase, step_idx, comp in snaps:
                x_ids = prompt_ids(trace, tokenizer) + comp
                groups = classify(plen, gen_length, set(inst.positions), x_ids, mask_id)
                others = groups["other_masked"] + groups["other_open"]
                tL, tR = plen + poss[0], plen + poss[1]
                tF, tS = plen + p_first, plen + p_second

                for layer in layer_ids:
                    if layer not in cache_attn[step_idx]:
                        continue
                    attn = cache_attn[step_idx][layer]
                    if others:
                        left_shares[layer][phase].append(
                            share(mean_out(attn, others, [tL]), mean_out(attn, others, [tR]))
                        )
                        first_shares[layer][phase].append(
                            share(mean_out(attn, others, [tF]), mean_out(attn, others, [tS]))
                        )
                    if phase == "all_open":
                        within[layer]["L_self"].append(float(attn[tL, tL]))
                        within[layer]["L_sib"].append(float(attn[tL, tR]))
                        within[layer]["R_self"].append(float(attn[tR, tR]))
                        within[layer]["R_sib"].append(float(attn[tR, tL]))
                        h = cache_hid[step_idx][layer]
                        with torch.inference_mode():
                            hL = h[tL].to(device=args.device, dtype=head_dtype)
                            hR = h[tR].to(device=args.device, dtype=head_dtype)
                            logits_L = lm_head(hL.unsqueeze(0)).float().cpu()[0]
                            logits_R = lm_head(hR.unsqueeze(0)).float().cpu()[0]
                        id_L, id_R = x_ids[tL], x_ids[tR]
                        probe[layer]["L_self"].append(int((logits_L > logits_L[id_L]).sum()))
                        probe[layer]["R_self"].append(int((logits_R > logits_R[id_R]).sum()))
                        probe[layer]["L_reads_R"].append(int((logits_L > logits_L[id_R]).sum()))
                        probe[layer]["R_reads_L"].append(int((logits_R > logits_R[id_L]).sum()))

        if (ri + 1) % 8 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    ok = tot = 0
    for i in all_inst:
        final = {t.pos_comp: t.token_id for t in i.tokens}
        for s in i.sibling_conf_at_first:
            if s.get("still_masked") and s["predicted_token_id"] is not None:
                tot += 1
                ok += s["predicted_token_id"] == final[s["pos_comp"]]

    lines = [
        f"# {args.family.upper()} multitoken suite (WikiText)\n\n",
        f"Model: **{args.model}**\n",
        f"Checkpoint: `{args.checkpoint}` n={len(rows)}\n",
        f"MASK={mask_id}, layers={sorted(layer_ids)}\n\n",
        "## 1. Unmask order\n\n",
        f"- lexical multi-token: **{len(all_inst)}**, 2-tok: **{len(two)}**\n",
        f"- consecutive: **{100*sum(i.consecutive for i in all_inst)/max(1,len(all_inst)):.1f}%**\n",
        f"- same-step: **{100*sum(i.same_step for i in all_inst)/max(1,len(all_inst)):.1f}%**\n",
        f"- 2-tok gap=1: **{100*sum(i.step_span==1 for i in two)/max(1,len(two)):.1f}%**\n",
        f"- LR / RL: **{100*sum(order_pattern(i)=='LR' for i in two)/max(1,len(two)):.1f}%** / "
        f"**{100*sum(order_pattern(i)=='RL' for i in two)/max(1,len(two)):.1f}%**\n",
        f"- sibling pred==final: **{ok}/{tot} ({100*ok/tot if tot else 0:.1f}%)**\n\n",
        "## 2. Other → left vs right\n\n",
        "| layer | phase | mean left_share | prefer left% | prefer right% | n |\n",
        "|------:|-------|----------------:|-------------:|--------------:|--:|\n",
    ]
    for layer in sorted(layer_ids):
        for phase in ("before_first", "between", "all_open"):
            xs = left_shares[layer][phase]
            if not xs:
                continue
            lines.append(
                f"| {layer} | {phase} | {statistics.mean(xs):.3f} | "
                f"{pct(xs, lambda s: s>0.5):.1f} | {pct(xs, lambda s: s<0.5):.1f} | {len(xs)} |\n"
            )

    lines += [
        "\n## 3. Other → first vs second unmasked\n\n",
        "| layer | phase | mean first_share | prefer 1st% | prefer 2nd% | n |\n",
        "|------:|-------|-----------------:|------------:|------------:|--:|\n",
    ]
    for layer in sorted(layer_ids):
        for phase in ("before_first", "between", "all_open"):
            xs = first_shares[layer][phase]
            if not xs:
                continue
            lines.append(
                f"| {layer} | {phase} | {statistics.mean(xs):.3f} | "
                f"{pct(xs, lambda s: s>0.5):.1f} | {pct(xs, lambda s: s<0.5):.1f} | {len(xs)} |\n"
            )

    lines += [
        "\n## 4. Within-word attn `all_open`\n\n",
        "| layer | L self | L→R | L sib% | R self | R→L | R sib% |\n",
        "|------:|-------:|----:|-------:|-------:|----:|-------:|\n",
    ]
    for layer in sorted(layer_ids):
        w = within[layer]
        if not w["L_self"]:
            continue
        ls, lr = statistics.mean(w["L_self"]), statistics.mean(w["L_sib"])
        rs, rl = statistics.mean(w["R_self"]), statistics.mean(w["R_sib"])
        lines.append(
            f"| {layer} | {ls:.4f} | {lr:.4f} | {share(lr,ls):.3f} | "
            f"{rs:.4f} | {rl:.4f} | {share(rl,rs):.3f} |\n"
        )

    lines += [
        "\n## 5. Logit lens sibling ranks `all_open` (0=top1)\n\n",
        "| layer | L self | R self | L→R sib | R→L sib |\n",
        "|------:|-------:|-------:|--------:|--------:|\n",
    ]
    for layer in sorted(layer_ids):
        p = probe[layer]
        if not p["L_self"]:
            continue
        lines.append(
            f"| {layer} | {statistics.mean(p['L_self']):.1f} | {statistics.mean(p['R_self']):.1f} | "
            f"**{statistics.mean(p['L_reads_R']):.1f}** | **{statistics.mean(p['R_reads_L']):.1f}** |\n"
        )

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

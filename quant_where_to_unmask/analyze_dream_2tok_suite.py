#!/usr/bin/env python3
"""Dream multitoken suite: unmask order + left/right attn + sibling logit lens."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
from collections import Counter, defaultdict
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
    ids = prompt_ids(trace, tokenizer) + comp
    return torch.tensor([ids], dtype=torch.long, device=device)


def mean_out(attn, queries, targets):
    if not queries or not targets:
        return None
    vals = [float(attn[q, targets].sum()) if len(targets) > 1 else float(attn[q, targets[0]]) for q in queries]
    # fix: targets as list
    out = []
    for q in queries:
        out.append(float(attn[q][torch.tensor(targets, dtype=torch.long)].sum()))
    return statistics.mean(out)


def share(a, b):
    t = a + b
    return a / t if t > 0 else 0.5


def pct(xs, pred):
    return 100 * sum(1 for x in xs if pred(x)) / len(xs) if xs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_dream_wikitext_fp16_g64_n64")
    parser.add_argument("--model", default="Dream-org/Dream-v0-Base-7B")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--layers", default="0,7,14,21,27")
    parser.add_argument("--out", default="dream_wikitext_multitoken_suite.md")
    args = parser.parse_args()

    layer_ids = {int(x) for x in args.layers.split(",")}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mask_id = tokenizer.mask_token_id
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()
    n_layers = len(model.model.layers)
    lm_head = model.lm_head

    ckpt = Path(args.checkpoint)
    rows = [json.loads(l) for l in open(ckpt / "rank0.jsonl") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit_traces]

    # --- unmask stats ---
    all_inst = []
    for row in rows:
        insts = [
            i
            for i in analyze_trace(ckpt / row["trace_path"], tokenizer, method="ws", kind_filter="alpha")
            if is_lexical(i.word)
        ]
        all_inst.extend(insts)
    two = [i for i in all_inst if len(i.positions) == 2]

    # --- attention + probe ---
    # layer -> phase -> left_share list
    left_shares = defaultdict(lambda: defaultdict(list))
    # layer -> phase -> first_share (unmask order)
    first_shares = defaultdict(lambda: defaultdict(list))
    # layer -> {left_reads_right, right_reads_left, left_self, right_self} at all_open
    within = defaultdict(lambda: defaultdict(list))
    # probe: from left/right hidden, rank of sibling token id (all_open)
    probe = defaultdict(lambda: defaultdict(list))  # layer -> role -> list of ranks (0=top)

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

        needed = set()
        plans = []
        for inst in instances:
            snaps = word_snapshots(inst, trace, plen)
            plans.append((inst, snaps))
            for _, step_idx, _ in snaps:
                needed.add(step_idx)

        cache_attn = {}
        cache_hid = {}
        for step_idx in sorted(needed):
            if step_idx < len(trace["steps_trace"]):
                comp = trace["steps_trace"][step_idx]["completion_tokens"]
            else:
                # after last
                comp = trace["steps_trace"][-1]["completion_tokens"][:]
                for u in trace["steps_trace"][-1]["unmasked"]:
                    comp[u["pos_comp"]] = u["token_id"]
            x = rebuild_x(trace, comp, tokenizer, args.device)
            with torch.inference_mode():
                out = model(x, attention_mask=None, output_attentions=True, output_hidden_states=True)
            # attentions: tuple of [B,H,T,T]
            cache_attn[step_idx] = {
                lid: out.attentions[lid][0].float().mean(0).cpu() for lid in layer_ids if lid < n_layers
            }
            # hidden_states: 0=embed ... n_layers=after last
            # map layer L residual after block L -> hidden_states[L+1]
            cache_hid[step_idx] = {
                lid: out.hidden_states[lid + 1][0].float().cpu() for lid in layer_ids if lid < n_layers
            }
            del out
            torch.cuda.empty_cache()

        for inst, snaps in plans:
            poss = sorted(inst.positions)
            by_time = sorted(inst.tokens, key=lambda t: (t.step, t.pos_comp))
            p_first, p_second = by_time[0].pos_comp, by_time[1].pos_comp
            for phase, step_idx, comp in snaps:
                x_ids = prompt_ids(trace, tokenizer) + comp
                # patch MASK_ID in classify by rewriting: classify uses 126336 hardcoded!
                # monkey: temporarily replace MASK checks via local classify
                groups = _classify(plen, gen_length, set(inst.positions), x_ids, mask_id)
                others = groups["other_masked"] + groups["other_open"]
                tL, tR = plen + poss[0], plen + poss[1]
                tF, tS = plen + p_first, plen + p_second

                for layer in layer_ids:
                    if layer not in cache_attn[step_idx]:
                        continue
                    attn = cache_attn[step_idx][layer]
                    if others:
                        ml = mean_out(attn, others, [tL])
                        mr = mean_out(attn, others, [tR])
                        left_shares[layer][phase].append(share(ml, mr))
                        mf = mean_out(attn, others, [tF])
                        ms = mean_out(attn, others, [tS])
                        first_shares[layer][phase].append(share(mf, ms))

                    if phase == "all_open":
                        within[layer]["L_self"].append(float(attn[tL, tL]))
                        within[layer]["L_sib"].append(float(attn[tL, tR]))
                        within[layer]["R_self"].append(float(attn[tR, tR]))
                        within[layer]["R_sib"].append(float(attn[tR, tL]))

                        # logit lens: decode sibling from this position's hidden
                        h = cache_hid[step_idx][layer]
                        with torch.inference_mode():
                            hL = h[tL].to(device=args.device, dtype=next(lm_head.parameters()).dtype)
                            hR = h[tR].to(device=args.device, dtype=next(lm_head.parameters()).dtype)
                            logits_L = lm_head(hL.unsqueeze(0)).float().cpu()[0]
                            logits_R = lm_head(hR.unsqueeze(0)).float().cpu()[0]
                        id_L = x_ids[tL]
                        id_R = x_ids[tR]
                        # rank of sibling (0 = top1)
                        rank_L_for_R = int((logits_L > logits_L[id_R]).sum().item())
                        rank_R_for_L = int((logits_R > logits_R[id_L]).sum().item())
                        rank_L_self = int((logits_L > logits_L[id_L]).sum().item())
                        rank_R_self = int((logits_R > logits_R[id_R]).sum().item())
                        probe[layer]["L_reads_R"].append(rank_L_for_R)
                        probe[layer]["R_reads_L"].append(rank_R_for_L)
                        probe[layer]["L_self"].append(rank_L_self)
                        probe[layer]["R_self"].append(rank_R_self)

        if (ri + 1) % 8 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    # --- report ---
    lines = [
        "# Dream-v0-Base-7B × WikiText: multitoken suite\n\n",
        f"Model: **{args.model}** (AR→diffusion, Qwen-based, bidirectional attn, AR-shifted logits)\n",
        f"Checkpoint: `{args.checkpoint}` — n={len(rows)}, g={64}, steps=64, k≈1 maskgit_plus\n",
        f"Same prompts as LLaDA WikiText run. MASK id={mask_id}\n\n",
        "## 1. Unmask order (lexical multi-token)\n\n",
        f"- multi-token lexical words: **{len(all_inst)}**\n",
        f"- 2-tok: **{len(two)}**\n",
        f"- consecutive positions: **{100*sum(i.consecutive for i in all_inst)/max(1,len(all_inst)):.1f}%**\n",
        f"- same-step co-unmask: **{100*sum(i.same_step for i in all_inst)/max(1,len(all_inst)):.1f}%**\n",
        f"- 2-tok gap=1: **{100*sum(i.step_span==1 for i in two)/max(1,len(two)):.1f}%**\n",
        f"- 2-tok order LR: **{100*sum(order_pattern(i)=='LR' for i in two)/max(1,len(two)):.1f}%**, "
        f"RL: **{100*sum(order_pattern(i)=='RL' for i in two)/max(1,len(two)):.1f}%**\n\n",
    ]

    # sibling pred
    ok = tot = 0
    for i in all_inst:
        final = {t.pos_comp: t.token_id for t in i.tokens}
        for s in i.sibling_conf_at_first:
            if s.get("still_masked") and s["predicted_token_id"] is not None:
                tot += 1
                ok += s["predicted_token_id"] == final[s["pos_comp"]]
    lines.append(f"- sibling pred==final at first unmask: **{ok}/{tot} ({100*ok/tot if tot else 0:.1f}%)**\n\n")

    lines.append("## 2. Other tokens → left vs right (by layer)\n\n")
    lines.append("| layer | phase | mean left_share | prefer left% | prefer right% | n |\n")
    lines.append("|------:|-------|----------------:|-------------:|--------------:|--:|\n")
    for layer in sorted(layer_ids):
        for phase in ("before_first", "between", "all_open"):
            xs = left_shares[layer][phase]
            if not xs:
                continue
            lines.append(
                f"| {layer} | {phase} | {statistics.mean(xs):.3f} | "
                f"{pct(xs, lambda s: s>0.5):.1f} | {pct(xs, lambda s: s<0.5):.1f} | {len(xs)} |\n"
            )
    lines.append("\n")

    lines.append("## 3. Other tokens → first vs second unmasked\n\n")
    lines.append("| layer | phase | mean first_share | prefer 1st% | prefer 2nd% | n |\n")
    lines.append("|------:|-------|-----------------:|------------:|------------:|--:|\n")
    for layer in sorted(layer_ids):
        for phase in ("before_first", "between", "all_open"):
            xs = first_shares[layer][phase]
            if not xs:
                continue
            lines.append(
                f"| {layer} | {phase} | {statistics.mean(xs):.3f} | "
                f"{pct(xs, lambda s: s>0.5):.1f} | {pct(xs, lambda s: s<0.5):.1f} | {len(xs)} |\n"
            )
    lines.append("\n")

    lines.append("## 4. Within-word attn at `all_open` (self vs sibling)\n\n")
    lines.append("| layer | L self | L→R | L sib% | R self | R→L | R sib% |\n")
    lines.append("|------:|-------:|----:|-------:|-------:|----:|-------:|\n")
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
    lines.append("\n")

    lines.append("## 5. Logit lens: where is sibling readable? (`all_open`)\n\n")
    lines.append("Mean rank of target token (0 = top-1). Lower = more readable from that position.\n\n")
    lines.append("| layer | L reads self | R reads self | **L reads R (sib)** | **R reads L (sib)** |\n")
    lines.append("|------:|-------------:|-------------:|--------------------:|--------------------:|\n")
    for layer in sorted(layer_ids):
        p = probe[layer]
        if not p["L_self"]:
            continue
        lines.append(
            f"| {layer} | {statistics.mean(p['L_self']):.1f} | {statistics.mean(p['R_self']):.1f} | "
            f"**{statistics.mean(p['L_reads_R']):.1f}** | **{statistics.mean(p['R_reads_L']):.1f}** |\n"
        )
    lines.append(
        "\nIf AR last-token storage holds: R should read L (and word) better than L reads R "
        "at late layers.\n"
    )

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


def _classify(plen, gen_length, word_positions, x_ids, mask_id):
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


if __name__ == "__main__":
    main()

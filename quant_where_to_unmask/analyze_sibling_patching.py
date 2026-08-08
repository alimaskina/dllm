#!/usr/bin/env python3
"""Sibling activation patching at `between` phase (2-tok words).

Clean: real opened subword.
Corrupt: replace opened token with MASK (or random).
Patch clean residual at opened position into corrupt run at layer L;
measure Δ logit of correct sibling token at masked sibling position.
"""

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

from analyze_multitoken_words import analyze_trace
from analyze_word_attention_phases import word_snapshots
from multitoken_word_filters import is_lexical


def load_trace(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def prompt_ids(trace, tokenizer):
    return tokenizer(trace["prompt_text"], add_special_tokens=False)["input_ids"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["dream", "llada"], default="dream")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit-traces", type=int, default=48)
    parser.add_argument("--limit-words", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--layers", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.family == "dream":
        args.model = args.model or "Dream-org/Dream-v0-Base-7B"
        args.checkpoint = args.checkpoint or "checkpoints/results_dream_wikitext_fp16_g64_n64"
        args.layers = args.layers or "0,7,14,21,27"
        args.out = args.out or "dream_sibling_patching.md"
        dtype = torch.bfloat16
        load_kw = {}
    else:
        args.model = args.model or "GSAI-ML/LLaDA-8B-Base"
        args.checkpoint = args.checkpoint or "checkpoints/results_wikitext_fp16_g64_n256"
        args.layers = args.layers or "0,7,14,21,31"
        args.out = args.out or "llada_sibling_patching.md"
        dtype = torch.float16
        load_kw = {}

    layer_ids = [int(x) for x in args.layers.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # LLaDA tokenizer often has mask_token_id=None; fall back to known id
    if args.family == "llada":
        mask_id = tokenizer.mask_token_id or 126336
    else:
        mask_id = tokenizer.mask_token_id
    assert mask_id is not None, "mask_token_id required"
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype, device_map=args.device, **load_kw
    )
    model.eval()

    if args.family == "dream":
        blocks = model.model.layers
        lm_head = model.lm_head
    else:
        blocks = model.model.transformer.blocks
        lm_head = model.model.transformer.ff_out

    ckpt = Path(args.checkpoint)
    rows = []
    for rf in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(l) for l in rf.open() if l.strip())
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit_traces]

    # collect between snapshots
    jobs = []
    for row in rows:
        trace = load_trace(ckpt / row["trace_path"])
        plen = len(prompt_ids(trace, tokenizer))
        for inst in analyze_trace(ckpt / row["trace_path"], tokenizer, method="ws", kind_filter="alpha"):
            if not is_lexical(inst.word) or len(inst.positions) != 2:
                continue
            for phase, step_idx, comp in word_snapshots(inst, trace, plen):
                if phase != "between":
                    continue
                poss = sorted(inst.positions)
                open_rel = [p for p in poss if comp[p] != mask_id]
                mask_rel = [p for p in poss if comp[p] == mask_id]
                if len(open_rel) != 1 or len(mask_rel) != 1:
                    continue
                final_sib = next(t.token_id for t in inst.tokens if t.pos_comp == mask_rel[0])
                jobs.append(
                    {
                        "trace": trace,
                        "comp": comp,
                        "plen": plen,
                        "open_abs": plen + open_rel[0],
                        "mask_abs": plen + mask_rel[0],
                        "sib_id": final_sib,
                        "word": inst.word,
                    }
                )
                break  # one between snap per word
        if len(jobs) >= args.limit_words:
            break
    jobs = jobs[: args.limit_words]
    print(f"patching jobs: {len(jobs)}")

    # layer -> list of (clean_logit, corrupt_logit, patched_logit)
    results = defaultdict(list)

    for ji, job in enumerate(jobs):
        plen = job["plen"]
        clean_ids = prompt_ids(job["trace"], tokenizer) + job["comp"]
        corrupt_ids = clean_ids[:]
        corrupt_ids[job["open_abs"]] = mask_id

        clean_x = torch.tensor([clean_ids], device=args.device)
        corrupt_x = torch.tensor([corrupt_ids], device=args.device)

        # cache clean hiddens at patch layers
        clean_h_at = {}
        handles = []

        def make_hook(layer_i):
            def hook(_m, _inp, out):
                # out may be tensor or tuple
                h = out[0] if isinstance(out, tuple) else out
                clean_h_at[layer_i] = h.detach()

            return hook

        for li in layer_ids:
            handles.append(blocks[li].register_forward_hook(make_hook(li)))

        with torch.inference_mode():
            if args.family == "dream":
                clean_out = model(clean_x, attention_mask=None)
            else:
                clean_out = model(clean_x, attention_mask=torch.ones_like(clean_x))
        for h in handles:
            h.remove()

        # clean sibling logit from clean final logits (with Dream AR-shift if dream)
        logits_clean = clean_out.logits[0].float()
        if args.family == "dream":
            logits_clean = torch.cat([logits_clean[:1], logits_clean[:-1]], dim=0)
        clean_sib = float(logits_clean[job["mask_abs"], job["sib_id"]])

        # corrupt baseline
        with torch.inference_mode():
            if args.family == "dream":
                corrupt_out = model(corrupt_x, attention_mask=None)
            else:
                corrupt_out = model(corrupt_x, attention_mask=torch.ones_like(corrupt_x))
        logits_c = corrupt_out.logits[0].float()
        if args.family == "dream":
            logits_c = torch.cat([logits_c[:1], logits_c[:-1]], dim=0)
        corrupt_sib = float(logits_c[job["mask_abs"], job["sib_id"]])

        # patch each layer separately: corrupt forward, replace opened pos hidden at layer L
        for li in layer_ids:
            stored = clean_h_at[li]
            open_abs = job["open_abs"]

            def patch_hook(_m, _inp, out, _stored=stored, _pos=open_abs):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[:, _pos, :] = _stored[:, _pos, :]
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h

            handle = blocks[li].register_forward_hook(patch_hook)
            with torch.inference_mode():
                if args.family == "dream":
                    pout = model(corrupt_x, attention_mask=None)
                else:
                    pout = model(corrupt_x, attention_mask=torch.ones_like(corrupt_x))
            handle.remove()
            logits_p = pout.logits[0].float()
            if args.family == "dream":
                logits_p = torch.cat([logits_p[:1], logits_p[:-1]], dim=0)
            patched_sib = float(logits_p[job["mask_abs"], job["sib_id"]])
            results[li].append((clean_sib, corrupt_sib, patched_sib))

        if (ji + 1) % 10 == 0:
            print(f"  {ji + 1}/{len(jobs)}")

    lines = [
        f"# Sibling activation patching ({args.family})\n\n",
        f"Model: **{args.model}**, words: **{len(jobs)}**, phase=`between`\n\n",
        "Corrupt = opened subword → MASK. Patch = restore clean residual at opened pos at layer L.\n",
        "Metric: logit of **correct sibling token** at still-masked position.\n\n",
        "| layer | clean | corrupt | patched | Δ(patch−corrupt) | recovery% |\n",
        "|------:|------:|--------:|--------:|-----------------:|----------:|\n",
    ]
    for li in layer_ids:
        rows_l = results[li]
        if not rows_l:
            continue
        c = statistics.mean(r[0] for r in rows_l)
        k = statistics.mean(r[1] for r in rows_l)
        p = statistics.mean(r[2] for r in rows_l)
        gap = c - k
        rec = 100 * (p - k) / gap if abs(gap) > 1e-6 else 0.0
        lines.append(
            f"| {li} | {c:.2f} | {k:.2f} | {p:.2f} | **{p-k:+.2f}** | **{rec:.1f}%** |\n"
        )
    lines.append(
        "\nrecovery% = (patched−corrupt)/(clean−corrupt). ~100% ⇒ that layer's opened residual "
        "carries the sibling-relevant info.\n"
    )
    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test: does step-0 pairwise attention predict same multi-token word grouping?"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from analyze_hidden_jump_at_unmask import prompt_ids_for, rebuild_x
from analyze_multitoken_words import char_spans, find_word_instances_ws, load_trace
from llada_attn_capture import forward_attn
from multitoken_word_filters import is_lexical

MASK_ID = 126336


def pos_to_word(final_ids, tokenizer):
    spans, full = char_spans(final_ids, tokenizer)
    pos2w: dict[int, int] = {}
    words = []
    for wi, (word, _kind, positions) in enumerate(find_word_instances_ws(full, spans)):
        if not is_lexical(word) or len(positions) < 2:
            continue
        words.append((word, positions))
        for p in positions:
            pos2w[p] = wi
    return pos2w, words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--device", default="cuda:2")
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
    random.Random(42).shuffle(rows)
    rows = rows[: args.limit_traces]

    same_sym: list[float] = []
    diff_sym: list[float] = []
    same_conf: list[float] = []
    diff_conf: list[float] = []

    for row in rows:
        trace = load_trace(ckpt / row["trace_path"])
        plen = len(prompt_ids_for(trace, tokenizer))
        gen = trace["gen_length"]
        final_ids = trace["steps_trace"][-1]["completion_tokens"]
        pos2w, _ = pos_to_word(final_ids, tokenizer)
        if not pos2w:
            continue

        comp0 = trace["steps_trace"][0]["completion_tokens"]
        comp = trace["steps_trace"][0]["completion"]
        x = rebuild_x(trace, comp0, tokenizer, args.device)
        attn = forward_attn(model, x, {args.layer})[args.layer]

        for i in range(gen - 1):
            j = i + 1
            if int(comp0[i]) != MASK_ID or int(comp0[j]) != MASK_ID:
                continue
            ai, aj = plen + i, plen + j
            sym = float(attn[ai, aj] + attn[aj, ai])
            c = min(comp["confidence"][i], comp["confidence"][j])
            wi, wj = pos2w.get(i), pos2w.get(j)
            if wi is not None and wi == wj:
                same_sym.append(sym)
                same_conf.append(c)
            else:
                diff_sym.append(sym)
                diff_conf.append(c)

        del attn
        torch.cuda.empty_cache()

    print(f"Traces: {len(rows)}")
    print(f"Adjacent masked pairs: same_word={len(same_sym)}, diff_word={len(diff_sym)}")
    print(f"Layer {args.layer}, step 0\n")

    ms, md = statistics.mean(same_sym), statistics.mean(diff_sym)
    print(f"Symmetric attn  mean same={ms:.5f}  diff={md:.5f}  ratio={ms/md:.3f}")
    print(f"Symmetric attn  median same={statistics.median(same_sym):.5f}  diff={statistics.median(diff_sym):.5f}")

    mc_s, mc_d = statistics.mean(same_conf), statistics.mean(diff_conf)
    print(f"\nConfidence (min of pair)  same={mc_s:.4f}  diff={mc_d:.4f}  ratio={mc_s/mc_d:.2f}")

    print("\nThreshold: predict same_word if sym_attn >= t")
    for t in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        tp = sum(1 for x in same_sym if x >= t)
        fp = sum(1 for x in diff_sym if x >= t)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / len(same_sym) if same_sym else 0.0
        print(f"  t={t:.3f}: precision={prec:.3f} recall={rec:.3f}  (tp={tp} fp={fp})")

    # confidence threshold as baseline
    print("\nBaseline: predict same_word if min_conf >= t")
    for t in [0.05, 0.1, 0.15, 0.2, 0.3]:
        tp = sum(1 for x in same_conf if x >= t)
        fp = sum(1 for x in diff_conf if x >= t)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / len(same_conf) if same_conf else 0.0
        print(f"  t={t:.2f}: precision={prec:.3f} recall={rec:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Correlate unmask history (conf, step, flip, jump) with later attention to the token."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from analyze_attention_at_unmask import attn_profile, cache_trace_attn, region_indices
from analyze_hidden_jump_at_unmask import (
    collect_trace_jobs,
    completion_after_step,
    load_trace,
    prompt_ids_for,
)
from llada_attn_capture import attn_mass

MASK_ID = 126336


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        sv = sorted((x, i) for i, x in enumerate(v))
        r = [0.0] * len(v)
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1][0] == sv[i][0]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[sv[k][1]] = avg
            i = j + 1
        return r

    return pearson(ranks(xs), ranks(ys))


def fmt(v: float | None, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def had_flip(trace: dict, step_idx: int, pos: int) -> bool:
    preds: list[int] = []
    for si in range(step_idx):
        comp = trace["steps_trace"][si]["completion"]
        if comp["masked"][pos]:
            tid = comp["predicted_token_id"][pos]
            if tid is not None:
                preds.append(tid)
    return len(set(preds)) > 1 if preds else False


def n_pred_flips(trace: dict, step_idx: int, pos: int) -> int:
    preds: list[int] = []
    for si in range(step_idx):
        comp = trace["steps_trace"][si]["completion"]
        if comp["masked"][pos]:
            tid = comp["predicted_token_id"][pos]
            if tid is not None:
                preds.append(tid)
    if len(preds) < 2:
        return 0
    return sum(1 for i in range(1, len(preds)) if preds[i] != preds[i - 1])


def incoming_mass(
    attn: torch.Tensor,
    target_abs: int,
    plen: int,
    gen_length: int,
    x_ids: list[int],
    from_kind: str,
) -> float:
    x_row = torch.tensor(x_ids, dtype=torch.long)
    comp_start = plen
    from_idx = region_indices(plen, gen_length, comp_start, from_kind, x_row)
    if not from_idx:
        return 0.0
    col = attn[:, target_abs]
    return attn_mass(col, from_idx)


def analyze_late_attention(
    attn_cache: list[dict[int, torch.Tensor]],
    trace: dict,
    plen: int,
    step_idx: int,
    u: dict,
    layer: int,
    tokenizer,
) -> dict:
    pos = u["pos_comp"]
    abs_pos = plen + pos
    gen_length = trace["gen_length"]
    steps = trace["steps_trace"]
    n_steps = len(steps)
    final_idx = n_steps  # cache has len(steps)+1 with final completion

    def x_at(si: int) -> list[int]:
        if si < n_steps:
            return prompt_ids_for(trace, tokenizer) + steps[si]["completion_tokens"]
        return prompt_ids_for(trace, tokenizer) + completion_after_step(trace, n_steps - 1)

    x_final = x_at(final_idx)

    attn_final = attn_cache[final_idx][layer]
    in_final_comp = incoming_mass(attn_final, abs_pos, plen, gen_length, x_final, "completion")
    in_final_real = incoming_mass(attn_final, abs_pos, plen, gen_length, x_final, "comp_real")
    in_final_masked = incoming_mass(attn_final, abs_pos, plen, gen_length, x_final, "comp_masked")
    in_final_prompt = incoming_mass(attn_final, abs_pos, plen, gen_length, x_final, "prompt")

    # mean incoming from completion over steps after unmask (pos is REAL in cache[si] for si > step_idx)
    post_in_comp: list[float] = []
    post_in_real: list[float] = []
    post_out_from_unmasker: list[float] = []
    for si in range(step_idx + 1, n_steps):
        x_si = x_at(si)
        attn_si = attn_cache[si][layer]
        post_in_comp.append(incoming_mass(attn_si, abs_pos, plen, gen_length, x_si, "completion"))
        post_in_real.append(incoming_mass(attn_si, abs_pos, plen, gen_length, x_si, "comp_real"))
        for u2 in steps[si]["unmasked"]:
            q_abs = plen + u2["pos_comp"]
            post_out_from_unmasker.append(float(attn_si[q_abs, abs_pos].item()))

    prof_unmask = attn_profile(attn_cache[step_idx + 1][layer], abs_pos, plen, gen_length, x_at(step_idx + 1))

    return {
        "step": step_idx,
        "pos_comp": pos,
        "token": u["token"],
        "layer": layer,
        "confidence": u["confidence"],
        "mask_ratio": steps[step_idx].get("mask_ratio"),
        "steps_until_end": n_steps - 1 - step_idx,
        "had_flip": had_flip(trace, step_idx, pos),
        "n_pred_flips": n_pred_flips(trace, step_idx, pos),
        "after_in_comp_real": prof_unmask["in_comp_real"],
        "final_in_completion": in_final_comp,
        "final_in_comp_real": in_final_real,
        "final_in_comp_masked": in_final_masked,
        "final_in_prompt": in_final_prompt,
        "post_mean_in_completion": statistics.mean(post_in_comp) if post_in_comp else None,
        "post_mean_in_comp_real": statistics.mean(post_in_real) if post_in_real else None,
        "post_mean_attn_from_later_unmaskers": statistics.mean(post_out_from_unmasker)
        if post_out_from_unmasker
        else None,
    }


def render_report(events: list[dict], n_traces: int, ckpt: str, layers: list[int], out: Path) -> None:
    lines = [
        "# Attention позже unmask vs история открытия\n\n",
        f"Traces: **{n_traces}**, событий: **{len(events) // len(layers)}** × layers, "
        f"checkpoint: `{ckpt}`, layers: **{layers}**\n\n",
        "## Методология\n\n",
        "Для каждого unmask на step `s`, pos `p` фиксируем **историю открытия**: "
        "confidence, step, flip предсказания пока masked, steps до конца блока.\n\n",
        "Attention **потом**:\n",
        "- `final_in_*` — incoming на `p` на **финальном** forward (все 64 tok real)\n",
        "- `post_mean_in_*` — средний incoming на `p` на steps `s+1..63`\n",
        "- `post_mean_attn_from_later_unmaskers` — attn от query-позиций, которые unmask'ятся **позже** `p`, на key `p`\n\n",
        "Корреляции: Pearson / Spearman.\n\n",
        "---\n\n",
    ]

    metric_keys = [
        "final_in_completion",
        "final_in_comp_real",
        "final_in_prompt",
        "post_mean_in_completion",
        "post_mean_in_comp_real",
        "post_mean_attn_from_later_unmaskers",
        "after_in_comp_real",
    ]
    meta_keys = [
        ("confidence", "confidence"),
        ("step", "step unmask"),
        ("steps_until_end", "steps until block end"),
        ("n_pred_flips", "prediction flips while masked"),
        ("mask_ratio", "mask_ratio at unmask"),
    ]

    for layer in layers:
        le = [e for e in events if e["layer"] == layer]
        lines.append(f"## Layer {layer}\n\n")
        lines.append("### Корреляции\n\n| meta | metric | pearson | spearman |\n|------|--------|---------|----------|\n")
        for mk, mlabel in meta_keys:
            for yk in metric_keys:
                pairs = [(e[mk], e[yk]) for e in le if e.get(mk) is not None and e.get(yk) is not None]
                if len(pairs) < 20:
                    continue
                xs, ys = zip(*pairs)
                lines.append(
                    f"| {mlabel} | {yk} | {fmt(pearson(list(xs), list(ys)), signed=True)} | "
                    f"{fmt(spearman(list(xs), list(ys)), signed=True)} |\n"
                )

        lines.append("\n### Бuckets\n\n")
        lines.append(
            "| subset | n | final_in_real | post_in_real | attn from later unmaskers |\n"
            "|--------|---|---------------|--------------|---------------------------|\n"
        )

        def row(label: str, filt) -> None:
            sub = [e for e in le if filt(e)]
            if len(sub) < 15:
                return
            lines.append(
                f"| {label} | {len(sub)} | "
                f"{fmt(statistics.mean(e['final_in_comp_real'] for e in sub))} | "
                f"{fmt(statistics.mean(e['post_mean_in_comp_real'] for e in sub if e['post_mean_in_comp_real'] is not None))} | "
                f"{fmt(statistics.mean(e['post_mean_attn_from_later_unmaskers'] for e in sub if e['post_mean_attn_from_later_unmaskers'] is not None))} |\n"
            )

        row("conf < 0.3", lambda e: e["confidence"] < 0.3)
        row("conf 0.3–0.6", lambda e: 0.3 <= e["confidence"] < 0.6)
        row("conf ≥ 0.6", lambda e: e["confidence"] >= 0.6)
        row("early step < 8", lambda e: e["step"] < 8)
        row("mid 8–31", lambda e: 8 <= e["step"] < 32)
        row("late step ≥ 32", lambda e: e["step"] >= 32)
        row("no flip", lambda e: not e["had_flip"])
        row("had flip", lambda e: e["had_flip"])
        row("flips ≥ 3", lambda e: e["n_pred_flips"] >= 3)
        lines.append("\n")

    # cross-layer summary
    lines.append("## Выводы\n\n")
    for layer in layers:
        le = [e for e in events if e["layer"] == layer]
        r_conf = pearson(
            [e["confidence"] for e in le],
            [e["final_in_comp_real"] for e in le],
        )
        r_step = spearman(
            [e["step"] for e in le],
            [e["final_in_comp_real"] for e in le],
        )
        r_flip = pearson(
            [float(e["had_flip"]) for e in le],
            [e["post_mean_attn_from_later_unmaskers"] for e in le if e["post_mean_attn_from_later_unmaskers"]],
        )
        flip_sub = [e for e in le if e["post_mean_attn_from_later_unmaskers"] is not None]
        no_flip = [e for e in flip_sub if not e["had_flip"]]
        yes_flip = [e for e in flip_sub if e["had_flip"]]
        lines.append(f"- **Layer {layer}**: conf↔final_in_real r={fmt(r_conf, signed=True)}; ")
        lines.append(f"step↔final_in_real ρ={fmt(r_step, signed=True)}.\n")
        if no_flip and yes_flip:
            m0 = statistics.mean(e["post_mean_attn_from_later_unmaskers"] for e in no_flip)
            m1 = statistics.mean(e["post_mean_attn_from_later_unmaskers"] for e in yes_flip)
            lines.append(
                f"  attn от later unmaskers: no_flip={fmt(m0)}, had_flip={fmt(m1)} "
                f"(Δ={fmt(m1 - m0, signed=True)}).\n"
            )

    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="16,31")
    parser.add_argument("--out", default="attention_late_correlation.md")
    parser.add_argument("--save-events", default="attention_late_correlation_events.json")
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    from llada_attn_capture import get_blocks

    n_blocks = len(get_blocks(model))
    layer_ids = set(int(x.strip()) for x in args.layers.split(","))
    report_layers = sorted(layer_ids)

    jobs = collect_trace_jobs(Path(args.checkpoint), tokenizer, args.limit_traces, args.seed)
    print(f"traces={len(jobs)} layers={report_layers}")

    events: list[dict] = []
    for i, job in enumerate(jobs):
        trace = load_trace(job.trace_path)
        attn_cache = cache_trace_attn(model, trace, tokenizer, args.device, layer_ids)
        for step_idx, st in enumerate(trace["steps_trace"]):
            for u in st["unmasked"]:
                for layer in report_layers:
                    events.append(
                        analyze_late_attention(
                            attn_cache, trace, job.prompt_len, step_idx, u, layer, tokenizer
                        )
                    )
        del attn_cache
        torch.cuda.empty_cache()
        if (i + 1) % 4 == 0 or i + 1 == len(jobs):
            print(f"  {i + 1}/{len(jobs)} traces → {len(events)} events")

    if args.save_events:
        Path(args.save_events).write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(events, len(jobs), args.checkpoint, report_layers, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

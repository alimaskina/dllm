#!/usr/bin/env python3
"""Attention pattern changes at token unmask moments."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from analyze_hidden_jump_at_unmask import (
    collect_trace_jobs,
    completion_after_step,
    control_positions,
    fmt,
    load_trace,
    prompt_ids_for,
    rebuild_x,
)
from llada_attn_capture import attn_mass, forward_attn, get_blocks

MASK_ID = 126336


def region_indices(plen: int, gen_length: int, comp_start: int, kind: str, x_row: torch.Tensor) -> list[int]:
    seq_len = plen + gen_length
    if kind == "prompt":
        return list(range(plen))
    if kind == "comp_masked":
        return [comp_start + i for i in range(gen_length) if int(x_row[comp_start + i].item()) == MASK_ID]
    if kind == "comp_real":
        return [comp_start + i for i in range(gen_length) if int(x_row[comp_start + i].item()) != MASK_ID]
    if kind == "completion":
        return list(range(comp_start, seq_len))
    raise ValueError(kind)


def attn_profile(
    attn: torch.Tensor,
    abs_pos: int,
    plen: int,
    gen_length: int,
    x_ids: list[int],
) -> dict:
    x_row = torch.tensor(x_ids, dtype=torch.long)
    comp_start = plen
    row_out = attn[abs_pos]
    col_in = attn[:, abs_pos]

    prof = {}
    for kind in ("prompt", "comp_masked", "comp_real", "completion"):
        idx = region_indices(plen, gen_length, comp_start, kind, x_row)
        prof[f"out_{kind}"] = attn_mass(row_out, idx)
        prof[f"in_{kind}"] = attn_mass(col_in, idx)

    prof["out_self"] = float(row_out[abs_pos].item())
    prof["in_self"] = float(col_in[abs_pos].item())
    return prof


def delta_profiles(before: dict, after: dict) -> dict:
    out = {}
    for k in before:
        out[f"d_{k}"] = after[k] - before[k]
    return out


def analyze_event(
    attn_before: dict[int, torch.Tensor],
    attn_after: dict[int, torch.Tensor],
    trace: dict,
    plen: int,
    step_idx: int,
    u: dict,
    layer: int,
    word_positions: dict[int, set[int]],
    tokenizer,
) -> dict:
    pos = u["pos_comp"]
    abs_pos = plen + pos
    gen_length = trace["gen_length"]
    comp_start = plen

    x_before = prompt_ids_for(trace, tokenizer) + trace["steps_trace"][step_idx]["completion_tokens"]
    if step_idx + 1 < len(trace["steps_trace"]):
        x_after = prompt_ids_for(trace, tokenizer) + trace["steps_trace"][step_idx + 1]["completion_tokens"]
    else:
        x_after = prompt_ids_for(trace, tokenizer) + completion_after_step(trace, step_idx)

    prof_b = attn_profile(attn_before[layer], abs_pos, plen, gen_length, x_before)
    prof_a = attn_profile(attn_after[layer], abs_pos, plen, gen_length, x_after)
    delta = delta_profiles(prof_b, prof_a)

    controls = control_positions(trace, step_idx, skip=pos)
    ctrl_b = ctrl_a = None
    if controls:
        cp = controls[0]
        ctrl_abs = plen + cp
        ctrl_b = attn_profile(attn_before[layer], ctrl_abs, plen, gen_length, x_before)
        ctrl_a = attn_profile(attn_after[layer], ctrl_abs, plen, gen_length, x_after)

    sib_before = [
        plen + p
        for p in range(gen_length)
        if p != pos
        and int(x_before[plen + p]) == MASK_ID
        and p in word_positions
        and word_positions[p] & word_positions.get(pos, set())
    ]

    return {
        "step": step_idx,
        "pos_comp": pos,
        "token": u["token"],
        "layer": layer,
        "in_multitoken_word": pos in word_positions,
        **prof_b,
        **{f"after_{k}": v for k, v in prof_a.items()},
        **delta,
        "out_mass_sib_masks": attn_mass(attn_before[layer][abs_pos], sib_before),
        "after_out_mass_sib_masks": attn_mass(
            attn_after[layer][abs_pos],
            [i for i in sib_before if i < len(x_after) and int(x_after[i]) == MASK_ID],
        ),
        "ctrl_out_comp_masked_before": ctrl_b["out_comp_masked"] if ctrl_b else None,
        "ctrl_d_out_comp_masked": (ctrl_a["out_comp_masked"] - ctrl_b["out_comp_masked"])
        if ctrl_b and ctrl_a
        else None,
        "ctrl_d_in_comp_real": (ctrl_a["in_comp_real"] - ctrl_b["in_comp_real"])
        if ctrl_b and ctrl_a
        else None,
    }


def cache_trace_attn(model, trace: dict, tokenizer, device: str, layers: set[int]) -> list[dict[int, torch.Tensor]]:
    steps = trace["steps_trace"]
    cached: list[dict[int, torch.Tensor]] = []
    for step_idx in range(len(steps)):
        x = rebuild_x(trace, steps[step_idx]["completion_tokens"], tokenizer, device)
        cached.append(forward_attn(model, x, layers))
    x_final = rebuild_x(trace, completion_after_step(trace, len(steps) - 1), tokenizer, device)
    cached.append(forward_attn(model, x, layers))
    return cached


def agg(events: list[dict], key: str) -> float | None:
    vals = [e[key] for e in events if e.get(key) is not None]
    return statistics.mean(vals) if vals else None


def render_report(events: list[dict], n_traces: int, ckpt: str, layers: list[int], out: Path) -> None:
    lines = [
        "# Attention при unmask: куда смотрим и кто смотрит на нас\n\n",
        f"Traces: **{n_traces}**, событий: **{len(events)}**, checkpoint: `{ckpt}`, "
        f"layers: **{layers}**\n\n",
        "## Методология\n\n",
        "LLaDA не поддерживает `output_attentions` — веса считаем через hook на SDPA (bidirectional MDM).\n\n",
        "На step `s` (before) и `s+1` (after unmask pos `p`):\n",
        "- **out_X** = Σ attn[p→·] на регион X (prompt / comp_masked / comp_real)\n",
        "- **in_X** = Σ attn[·→p] из региона X\n",
        "- **d_*** = after − before; control = still-masked позиция на том же step\n\n",
        "---\n\n",
    ]

    for layer in layers:
        le = [e for e in events if e["layer"] == layer]
        lines += [
            f"## Layer {layer}\n\n",
            "### 1. Outgoing (куда смотрит unmask-позиция)\n\n",
            "| mass | before | after | Δ |\n",
            "|------|--------|-------|---|\n",
        ]
        for kind in ("prompt", "comp_masked", "comp_real", "self"):
            bk = f"out_{kind}"
            lines.append(
                f"| {kind} | {fmt(agg(le, bk))} | {fmt(agg(le, f'after_{bk}'))} | "
                f"{fmt(agg(le, f'd_{bk}'), signed=True)} |\n"
            )

        lines += [
            "\n### 2. Incoming (кто смотрит на unmask-позицию)\n\n",
            "| mass | before | after | Δ |\n",
            "|------|--------|-------|---|\n",
        ]
        for kind in ("prompt", "comp_masked", "comp_real", "self"):
            bk = f"in_{kind}"
            lines.append(
                f"| {kind} | {fmt(agg(le, bk))} | {fmt(agg(le, f'after_{bk}'))} | "
                f"{fmt(agg(le, f'd_{bk}'), signed=True)} |\n"
            )

        lines += [
            "\n### 3. vs control (still masked)\n\n",
            f"- mean Δ out→comp_masked (unmask): **{fmt(agg(le, 'd_out_comp_masked'), signed=True)}**\n",
            f"- mean Δ out→comp_masked (control): **{fmt(agg(le, 'ctrl_d_out_comp_masked'), signed=True)}**\n",
            f"- mean Δ in←comp_real (unmask): **{fmt(agg(le, 'd_in_comp_real'), signed=True)}**\n",
            f"- mean Δ in←comp_real (control): **{fmt(agg(le, 'ctrl_d_in_comp_real'), signed=True)}**\n\n",
            f"- mean out→sibling masks: before **{fmt(agg(le, 'out_mass_sib_masks'))}**, "
            f"after **{fmt(agg(le, 'after_out_mass_sib_masks'))}**\n\n",
            "---\n\n",
        ]

    lines += ["## Выводы\n\n"]
    le = events
    d_out_m = agg(le, "d_out_comp_masked")
    d_in_r = agg(le, "d_in_comp_real")
    if d_out_m is not None and d_out_m < -0.01:
        lines.append("- После unmask позиция **меньше** смотрит на другие маски.\n")
    elif d_out_m is not None and d_out_m > 0.01:
        lines.append("- После unmask позиция **больше** смотрит на другие маски.\n")
    if d_in_r is not None and d_in_r > 0.01:
        lines.append("- После unmask **больше** attention от уже открытых (real) completion токенов.\n")

    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="-1,16", help="comma-separated layer indices (-1=last)")
    parser.add_argument("--out", default="attention_at_unmask.md")
    parser.add_argument("--save-events", default="attention_at_unmask_events.json")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    n_blocks = len(get_blocks(model))
    raw_layers = [int(x.strip()) for x in args.layers.split(",")]
    layer_ids = {(n_blocks - 1 if l == -1 else l) for l in raw_layers}
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
                        analyze_event(
                            attn_cache[step_idx],
                            attn_cache[step_idx + 1],
                            trace,
                            job.prompt_len,
                            step_idx,
                            u,
                            layer,
                            job.word_positions,
                            tokenizer,
                        )
                    )
        del attn_cache
        torch.cuda.empty_cache()
        if (i + 1) % 4 == 0 or i + 1 == len(jobs):
            print(f"  {i + 1}/{len(jobs)} traces → {len(events)} events")

    if args.save_events:
        Path(args.save_events).write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(events, len(jobs), args.checkpoint, report_layers, Path(args.out))
    print(f"Wrote {args.out} ({len(events)} events)")


if __name__ == "__main__":
    main()

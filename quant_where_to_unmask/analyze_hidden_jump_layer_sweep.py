#!/usr/bin/env python3
"""Layer sweep: hidden jump at unmask across all transformer layers."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from analyze_hidden_jump_at_unmask import (
    collect_trace_jobs,
    completion_after_step,
    fmt,
    load_trace,
    pct,
    rebuild_x,
)
from transformers import AutoModel, AutoTokenizer


def forward_completion_hiddens_all_layers(
    model,
    x: torch.Tensor,
    comp_start: int,
    gen_length: int,
) -> list[torch.Tensor]:
    """Return list[layer][gen_length, dim] on CPU fp16."""
    with torch.inference_mode():
        out = model(x, attention_mask=torch.ones_like(x), output_hidden_states=True)
    return [
        h[0, comp_start : comp_start + gen_length].detach().cpu().to(torch.float16)
        for h in out.hidden_states
    ]


def cache_trace_hiddens_all_layers(
    model,
    trace: dict,
    tokenizer,
    device: str,
) -> tuple[list[list[torch.Tensor]], int, int]:
    """Return hiddens[step][layer][pos_comp, dim], prompt_len, gen_length."""
    steps = trace["steps_trace"]
    plen = len(tokenizer(trace["prompt_text"], add_special_tokens=False)["input_ids"])
    gen_length = trace["gen_length"]
    comp_start = plen

    cached: list[list[torch.Tensor]] = []
    for step_idx in range(len(steps)):
        x = rebuild_x(trace, steps[step_idx]["completion_tokens"], tokenizer, device)
        cached.append(forward_completion_hiddens_all_layers(model, x, comp_start, gen_length))

    x_final = rebuild_x(trace, completion_after_step(trace, len(steps) - 1), tokenizer, device)
    cached.append(forward_completion_hiddens_all_layers(model, x_final, comp_start, gen_length))
    return cached, plen, gen_length


def layer_hiddens_from_cache(cached: list[list[torch.Tensor]], layer: int) -> list[torch.Tensor]:
    """Convert all-layer cache to per-layer seq tensors indexed by abs completion pos."""
    return [step_layers[layer] for step_layers in cached]


def aggregate_layer(events: list[dict]) -> dict:
    def mean(key: str) -> float | None:
        vals = [e[key] for e in events if e.get(key) is not None]
        return statistics.mean(vals) if vals else None

    ratio_vals = [e["ratio_jump_l2"] for e in events if e.get("ratio_jump_l2") is not None]
    bigger = sum(1 for v in ratio_vals if v > 1.5)
    return {
        "n": len(events),
        "jump_cos": mean("jump_cos"),
        "ctrl_jump_cos": mean("ctrl_jump_cos_mean"),
        "delta_jump_cos": mean("delta_jump_cos"),
        "jump_l2": mean("jump_l2"),
        "ctrl_jump_l2": mean("ctrl_jump_l2_mean"),
        "ratio_jump_l2": mean("ratio_jump_l2"),
        "traj_cos": mean("traj_cos_mean"),
        "traj_l2": mean("traj_l2_mean"),
        "jump_vs_traj_l2": mean("jump_vs_traj_l2"),
        "jump_vs_traj_cos": mean("jump_vs_traj_cos"),
        "pct_l2_gt_1p5x_ctrl": pct(bigger, len(ratio_vals)),
    }


def _mean(vals) -> float | None:
    xs = [v for v in vals if v is not None]
    return statistics.mean(xs) if xs else None


def render_layer_sweep_report(
    by_layer: dict[int, dict],
    n_traces: int,
    n_events: int,
    n_layers: int,
    ckpt: str,
    out: Path,
) -> None:
    # hidden_states: 0=embed, 1..n_layers=blocks, index n_layers is last block output
    layer_ids = sorted(by_layer)
    ratios = {l: by_layer[l]["ratio_jump_l2"] for l in layer_ids if by_layer[l]["ratio_jump_l2"] is not None}
    deltas = {l: by_layer[l]["delta_jump_cos"] for l in layer_ids if by_layer[l]["delta_jump_cos"] is not None}
    best_ratio = max(ratios, key=ratios.get) if ratios else layer_ids[-1]
    best_delta = min(deltas, key=deltas.get) if deltas else layer_ids[-1]

    lines = [
        "# Hidden jump при unmask: sweep по слоям\n\n",
        f"Traces: **{n_traces}**, unmask-событий на слой: **{n_events}**, "
        f"checkpoint: `{ckpt}`, layers: **0..{n_layers}** ({n_layers + 1} hidden states incl. embed)\n\n",
        "## Методология\n\n",
        "Один forward на step → `output_hidden_states=True` → метрики jump/traj/control "
        "на **completion-регионе** (64 позиции) для каждого слоя.\n\n",
        "- **layer 0** — embedding output\n",
        f"- **layer 1..{n_layers - 1}** — промежуточные блоки\n",
        f"- **layer {n_layers}** — last block (то, что раньше брали как `layer=-1`)\n\n",
        "---\n\n",
        "## 1. Сводка по слоям\n\n",
        "| layer | jump cos | ctrl cos | Δ cos | jump L2 | ratio L2 | traj cos | jump/traj L2 | >1.5× ctrl |\n",
        "|-------|----------|----------|-------|---------|----------|----------|--------------|------------|\n",
    ]
    for layer in layer_ids:
        s = by_layer[layer]
        label = f"{layer}" if layer not in (0, n_layers) else (f"{layer} (embed)" if layer == 0 else f"{layer} (last)")
        lines.append(
            f"| {label} | {fmt(s['jump_cos'])} | {fmt(s['ctrl_jump_cos'])} | "
            f"{fmt(s['delta_jump_cos'], signed=True)} | {fmt(s['jump_l2'])} | "
            f"**{fmt(s['ratio_jump_l2'])}×** | {fmt(s['traj_cos'])} | "
            f"**{fmt(s['jump_vs_traj_l2'])}×** | {s['pct_l2_gt_1p5x_ctrl']} |\n"
        )

    lines += [
        "\n---\n\n",
        "## 2. Где эффект максимален\n\n",
        f"- **Max ratio L2 (jump/control):** layer **{best_ratio}** → "
        f"**{by_layer[best_ratio]['ratio_jump_l2']:.2f}×**\n",
        f"- **Max Δcos drop (jump − control):** layer **{best_delta}** → "
        f"**{by_layer[best_delta]['delta_jump_cos']:+.3f}**\n\n",
        "---\n\n",
        "## 3. Профиль ratio L2 по слоям\n\n",
        "```\n",
        "layer  ratio_L2\n",
    ]
    max_ratio = max((by_layer[l]["ratio_jump_l2"] or 0) for l in layer_ids)
    for layer in layer_ids:
        r = by_layer[layer]["ratio_jump_l2"] or 0
        bar = "█" * int(40 * r / max_ratio) if max_ratio > 0 else ""
        lines.append(f"{layer:3d}  {r:5.2f}x  {bar}\n")
    lines.append("```\n\n")

    lines += [
        "---\n\n",
        "## 4. Интерпретация\n\n",
    ]
    q = n_layers // 4
    early = _mean(by_layer[l]["ratio_jump_l2"] for l in layer_ids if l <= q)
    mid = _mean(by_layer[l]["ratio_jump_l2"] for l in layer_ids if q < l <= 3 * q)
    late = _mean(by_layer[l]["ratio_jump_l2"] for l in layer_ids if l > 3 * q)
    if early is not None:
        lines.append(f"- Early layers (0–{q}): mean ratio **{early:.2f}×**\n")
    if mid is not None:
        lines.append(f"- Mid layers ({q + 1}–{3 * q}): mean ratio **{mid:.2f}×**\n")
    if late is not None:
        lines.append(f"- Late layers ({3 * q + 1}–{n_layers}): mean ratio **{late:.2f}×**\n")

    if early is not None and late is not None:
        if late > early * 1.2:
            lines.append("- Скачок **усиливается** к верхним слоям.\n")
        elif early > late * 1.2:
            lines.append("- Скачок **сильнее в ранних** слоях, верхние сглаживают.\n")
        else:
            lines.append("- Профиль **относительно ровный** по глубине (нет одного доминирующего слоя).\n")

    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="hidden_jump_layer_sweep.md")
    parser.add_argument("--save-events", default="hidden_jump_layer_sweep.json")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    n_layers = int(getattr(model.config, "n_layers", getattr(model.config, "num_hidden_layers", 32)))

    jobs = collect_trace_jobs(Path(args.checkpoint), tokenizer, args.limit_traces, args.seed)
    print(f"traces={len(jobs)}, n_hidden_states={n_layers + 1}")

    events_by_layer: dict[int, list[dict]] = {layer: [] for layer in range(n_layers + 1)}

    for i, job in enumerate(jobs):
        trace = load_trace(job.trace_path)
        cached, plen, _ = cache_trace_hiddens_all_layers(model, trace, tokenizer, device)

        for layer in range(n_layers + 1):
            hiddens = layer_hiddens_from_cache(cached, layer)
            for step_idx, st in enumerate(trace["steps_trace"]):
                for u in st["unmasked"]:
                    events_by_layer[layer].append(
                        analyze_unmask_event_completion(
                            hiddens, trace, plen, step_idx, u, job.word_positions
                        )
                    )

        del cached
        torch.cuda.empty_cache()
        if (i + 1) % 8 == 0 or i + 1 == len(jobs):
            print(f"  {i + 1}/{len(jobs)} traces")

    by_layer = {layer: aggregate_layer(ev) for layer, ev in events_by_layer.items()}
    n_events = len(next(iter(events_by_layer.values())))

    if args.save_events:
        payload = {
            "by_layer_summary": {str(k): v for k, v in by_layer.items()},
        }
        Path(args.save_events).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    render_layer_sweep_report(by_layer, len(jobs), n_events, n_layers, args.checkpoint, Path(args.out))
    print(f"Wrote {args.out}")


def analyze_unmask_event_completion(
    hiddens: list[torch.Tensor],
    trace: dict,
    plen: int,
    step_idx: int,
    u: dict,
    word_positions: dict[int, set[int]],
) -> dict:
    """Same metrics as analyze_unmask_event but hiddens[step] is [gen_length, dim]."""
    from analyze_hidden_jump_at_unmask import (
        control_positions,
        cos_sim,
        l2_dist,
        masked_trajectory_stats,
    )

    pos = u["pos_comp"]
    h_before = hiddens[step_idx][pos]
    h_after = hiddens[step_idx + 1][pos]

    jump_cos = cos_sim(h_before, h_after)
    jump_l2 = l2_dist(h_before, h_after)

    controls = control_positions(trace, step_idx, skip=pos)
    ctrl_cos = [cos_sim(hiddens[step_idx][p], hiddens[step_idx + 1][p]) for p in controls]
    ctrl_l2 = [l2_dist(hiddens[step_idx][p], hiddens[step_idx + 1][p]) for p in controls]

    traj = masked_trajectory_stats_completion(hiddens, pos, step_idx)
    ctrl_traj_cos: list[float] = []
    ctrl_traj_l2: list[float] = []
    for p in controls[: min(8, len(controls))]:
        tstats = masked_trajectory_stats_completion(hiddens, p, step_idx)
        if tstats["traj_cos_mean"] is not None:
            ctrl_traj_cos.append(tstats["traj_cos_mean"])
            ctrl_traj_l2.append(tstats["traj_l2_mean"])

    ctrl_cos_mean = statistics.mean(ctrl_cos) if ctrl_cos else None
    ctrl_l2_mean = statistics.mean(ctrl_l2) if ctrl_l2 else None
    st = trace["steps_trace"][step_idx]

    return {
        "step": step_idx,
        "pos_comp": pos,
        "token": u["token"],
        "confidence": u["confidence"],
        "mask_ratio": st.get("mask_ratio"),
        "in_multitoken_word": pos in word_positions,
        "jump_cos": jump_cos,
        "jump_l2": jump_l2,
        "ctrl_jump_cos_mean": ctrl_cos_mean,
        "ctrl_jump_l2_mean": ctrl_l2_mean,
        "delta_jump_cos": jump_cos - ctrl_cos_mean if ctrl_cos_mean is not None else None,
        "ratio_jump_l2": jump_l2 / ctrl_l2_mean if ctrl_l2_mean and ctrl_l2_mean > 0 else None,
        "traj_cos_mean": traj["traj_cos_mean"],
        "traj_l2_mean": traj["traj_l2_mean"],
        "jump_vs_traj_cos": jump_cos - traj["traj_cos_mean"] if traj["traj_cos_mean"] is not None else None,
        "jump_vs_traj_l2": jump_l2 / traj["traj_l2_mean"] if traj["traj_l2_mean"] and traj["traj_l2_mean"] > 0 else None,
    }


def masked_trajectory_stats_completion(hiddens: list[torch.Tensor], pos: int, unmask_step: int) -> dict:
    from analyze_hidden_jump_at_unmask import cos_sim, l2_dist

    cos_vals: list[float] = []
    l2_vals: list[float] = []
    for t in range(unmask_step):
        cos_vals.append(cos_sim(hiddens[t][pos], hiddens[t + 1][pos]))
        l2_vals.append(l2_dist(hiddens[t][pos], hiddens[t + 1][pos]))
    return {
        "traj_cos_mean": statistics.mean(cos_vals) if cos_vals else None,
        "traj_l2_mean": statistics.mean(l2_vals) if l2_vals else None,
    }


if __name__ == "__main__":
    main()

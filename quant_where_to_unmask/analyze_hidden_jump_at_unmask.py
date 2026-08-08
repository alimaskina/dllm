#!/usr/bin/env python3
"""Hidden-state jump / trajectory analysis at token unmask moments."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from analyze_multitoken_words import WordInstance, analyze_trace, load_trace
from multitoken_word_filters import is_lexical

MASK_ID = 126336


@dataclass
class TraceJob:
    trace_path: Path
    prompt_len: int
    word_positions: dict[int, set[int]]  # pos_comp -> set of word instance ids


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)).item()
    )


def l2_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm().item())


def prompt_ids_for(trace: dict, tokenizer) -> list[int]:
    return tokenizer(trace["prompt_text"], add_special_tokens=False)["input_ids"]


def completion_after_step(trace: dict, step_idx: int) -> list[int]:
    steps = trace["steps_trace"]
    if step_idx + 1 < len(steps):
        return steps[step_idx + 1]["completion_tokens"]
    comp = steps[step_idx]["completion_tokens"][:]
    for u in steps[step_idx]["unmasked"]:
        comp[u["pos_comp"]] = u["token_id"]
    return comp


def rebuild_x(trace: dict, completion: list[int], tokenizer, device) -> torch.Tensor:
    ids = prompt_ids_for(trace, tokenizer) + completion
    return torch.tensor([ids], dtype=torch.long, device=device)


def forward_hidden(model, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
    with torch.inference_mode():
        out = model(x, attention_mask=torch.ones_like(x), output_hidden_states=True)
    return out.hidden_states[layer][0].detach()


def cache_trace_hiddens(model, trace: dict, tokenizer, device, layer: int) -> list[torch.Tensor]:
    steps = trace["steps_trace"]
    hiddens: list[torch.Tensor] = []
    for step_idx in range(len(steps)):
        x = rebuild_x(trace, steps[step_idx]["completion_tokens"], tokenizer, device)
        hiddens.append(forward_hidden(model, x, layer))
    x_final = rebuild_x(trace, completion_after_step(trace, len(steps) - 1), tokenizer, device)
    hiddens.append(forward_hidden(model, x_final, layer))
    return hiddens


def control_positions(trace: dict, step_idx: int, skip: int) -> list[int]:
    prev = trace["steps_trace"][step_idx]["completion"]["masked"]
    nxt = trace["steps_trace"][step_idx + 1]["completion"]["masked"] if step_idx + 1 < len(trace["steps_trace"]) else [False] * len(prev)
    if step_idx + 1 >= len(trace["steps_trace"]):
        nxt = [False] * len(prev)
    out = []
    for rel, (m0, m1) in enumerate(zip(prev, nxt)):
        if rel == skip:
            continue
        if m0 and m1:
            out.append(rel)
    return out


def masked_trajectory_stats(hiddens: list[torch.Tensor], plen: int, pos: int, unmask_step: int) -> dict:
    """Step-to-step cos/L2 for the same position while input token stays [MASK]."""
    abs_pos = plen + pos
    cos_vals: list[float] = []
    l2_vals: list[float] = []
    for t in range(unmask_step):
        cos_vals.append(cos_sim(hiddens[t][abs_pos], hiddens[t + 1][abs_pos]))
        l2_vals.append(l2_dist(hiddens[t][abs_pos], hiddens[t + 1][abs_pos]))
    return {
        "traj_steps": len(cos_vals),
        "traj_cos_mean": statistics.mean(cos_vals) if cos_vals else None,
        "traj_cos_min": min(cos_vals) if cos_vals else None,
        "traj_l2_mean": statistics.mean(l2_vals) if l2_vals else None,
        "traj_l2_max": max(l2_vals) if l2_vals else None,
    }


def analyze_unmask_event(
    trace: dict,
    hiddens: list[torch.Tensor],
    plen: int,
    step_idx: int,
    u: dict,
    word_positions: dict[int, set[int]],
) -> dict:
    pos = u["pos_comp"]
    abs_pos = plen + pos
    h_before = hiddens[step_idx][abs_pos]
    h_after = hiddens[step_idx + 1][abs_pos]

    jump_cos = cos_sim(h_before, h_after)
    jump_l2 = l2_dist(h_before, h_after)

    controls = control_positions(trace, step_idx, skip=pos)
    ctrl_cos = [cos_sim(hiddens[step_idx][plen + p], hiddens[step_idx + 1][plen + p]) for p in controls]
    ctrl_l2 = [l2_dist(hiddens[step_idx][plen + p], hiddens[step_idx + 1][plen + p]) for p in controls]

    traj = masked_trajectory_stats(hiddens, plen, pos, step_idx)
    ctrl_traj_cos: list[float] = []
    ctrl_traj_l2: list[float] = []
    for p in controls[: min(8, len(controls))]:
        tstats = masked_trajectory_stats(hiddens, plen, p, step_idx)
        if tstats["traj_cos_mean"] is not None:
            ctrl_traj_cos.append(tstats["traj_cos_mean"])
            ctrl_traj_l2.append(tstats["traj_l2_mean"])

    ctrl_cos_mean = statistics.mean(ctrl_cos) if ctrl_cos else None
    ctrl_l2_mean = statistics.mean(ctrl_l2) if ctrl_l2 else None
    ctrl_traj_cos_mean = statistics.mean(ctrl_traj_cos) if ctrl_traj_cos else None
    ctrl_traj_l2_mean = statistics.mean(ctrl_traj_l2) if ctrl_traj_l2 else None

    st = trace["steps_trace"][step_idx]
    return {
        "step": step_idx,
        "pos_comp": pos,
        "token": u["token"],
        "confidence": u["confidence"],
        "mask_ratio": st.get("mask_ratio"),
        "n_masked_before": st.get("n_masked_before"),
        "in_multitoken_word": pos in word_positions,
        "jump_cos": jump_cos,
        "jump_l2": jump_l2,
        "ctrl_n": len(controls),
        "ctrl_jump_cos_mean": ctrl_cos_mean,
        "ctrl_jump_l2_mean": ctrl_l2_mean,
        "delta_jump_cos": jump_cos - ctrl_cos_mean if ctrl_cos_mean is not None else None,
        "ratio_jump_l2": jump_l2 / ctrl_l2_mean if ctrl_l2_mean and ctrl_l2_mean > 0 else None,
        "traj_steps": traj["traj_steps"],
        "traj_cos_mean": traj["traj_cos_mean"],
        "traj_l2_mean": traj["traj_l2_mean"],
        "ctrl_traj_cos_mean": ctrl_traj_cos_mean,
        "ctrl_traj_l2_mean": ctrl_traj_l2_mean,
        "jump_vs_traj_cos": jump_cos - traj["traj_cos_mean"] if traj["traj_cos_mean"] is not None else None,
        "jump_vs_traj_l2": jump_l2 / traj["traj_l2_mean"] if traj["traj_l2_mean"] and traj["traj_l2_mean"] > 0 else None,
    }


def collect_trace_jobs(ckpt: Path, tokenizer, limit_traces: int, seed: int) -> list[TraceJob]:
    rows = []
    for rank_file in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(line) for line in rank_file.open(encoding="utf-8") if line.strip())
    rng = random.Random(seed)
    rng.shuffle(rows)

    jobs: list[TraceJob] = []
    for row in rows[:limit_traces]:
        trace_path = ckpt / row["trace_path"]
        if not trace_path.exists():
            continue
        trace = load_trace(trace_path)
        plen = len(prompt_ids_for(trace, tokenizer))
        word_positions: dict[int, set[int]] = defaultdict(set)
        for wi, inst in enumerate(analyze_trace(trace_path, tokenizer, method="ws", kind_filter="alpha")):
            if is_lexical(inst.word) and len(inst.positions) >= 2:
                for p in inst.positions:
                    word_positions[p].add(wi)
        jobs.append(TraceJob(trace_path=trace_path, prompt_len=plen, word_positions=word_positions))
    return jobs


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def fmt(v, spec=".3f", signed: bool = False) -> str:
    if v is None:
        return "—"
    if signed:
        return f"{v:+.3f}"
    return format(v, spec)


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(q * (len(s) - 1))))
    return s[idx]


def agg(events: list[dict], key: str) -> float | None:
    vals = [e[key] for e in events if e.get(key) is not None]
    return statistics.mean(vals) if vals else None


def render_report(events: list[dict], n_traces: int, ckpt: str, layer: int, out: Path) -> None:
    all_e = events
    word_e = [e for e in events if e["in_multitoken_word"]]
    early = [e for e in events if e["step"] == 0]
    late = [e for e in events if e["step"] >= 32]

    jump_cos = [e["jump_cos"] for e in all_e]
    jump_l2 = [e["jump_l2"] for e in all_e]
    delta_cos = [e["delta_jump_cos"] for e in all_e if e["delta_jump_cos"] is not None]
    ratio_l2 = [e["ratio_jump_l2"] for e in all_e if e["ratio_jump_l2"] is not None]
    jump_vs_traj_cos = [e["jump_vs_traj_cos"] for e in all_e if e["jump_vs_traj_cos"] is not None]
    jump_vs_traj_l2 = [e["jump_vs_traj_l2"] for e in all_e if e["jump_vs_traj_l2"] is not None]

    bigger_jump = sum(1 for e in all_e if e["ratio_jump_l2"] is not None and e["ratio_jump_l2"] > 1.5)
    n_ratio = sum(1 for e in all_e if e["ratio_jump_l2"] is not None)
    smaller_cos = sum(1 for e in all_e if e["jump_vs_traj_cos"] is not None and e["jump_vs_traj_cos"] < -0.05)

    lines = [
        "# Hidden jump при unmask: траектория и скачок MASK→REAL\n\n",
        f"Traces: **{n_traces}**, unmask-событий: **{len(all_e)}**, checkpoint: `{ckpt}`, layer: **{layer}**\n\n",
        "## Методология\n\n",
        "Trace фиксирует состояние **до** commit unmask (см. `generate.py`).\n",
        "Для unmask на step `s`, pos `p`:\n",
        "- **h_before** = forward на sequence trace step `s` (позиция ещё `[MASK]`)\n",
        "- **h_after** = forward на sequence trace step `s+1` (позиция уже REAL)\n",
        "- **jump** = переход `h[s] → h[s+1]` на той же позиции\n",
        "- **control** = те же метрики для позиций, оставшихся `[MASK]` и на `s`, и на `s+1`\n",
        "- **traj** = средний step-to-step сдвиг hidden на этой позиции **пока она masked** (steps `0..s-1`)\n\n",
        "---\n\n",
        "## 1. Главный результат: есть ли скачок при unmask?\n\n",
        "| Метрика | unmask jump | control (still masked) | Δ / ratio |\n",
        "|---------|-------------|------------------------|-----------|\n",
        f"| mean cos(step→step+1) | {fmt(agg(all_e, 'jump_cos'))} | {fmt(agg(all_e, 'ctrl_jump_cos_mean'))} | "
        f"{fmt(agg(all_e, 'delta_jump_cos'), signed=True)} |\n",
        f"| mean L2(step→step+1) | {fmt(agg(all_e, 'jump_l2'))} | {fmt(agg(all_e, 'ctrl_jump_l2_mean'))} | "
        f"ratio **{fmt(agg(all_e, 'ratio_jump_l2'))}×** |\n",
        f"| median cos | {fmt(statistics.median(jump_cos))} | {fmt(agg(all_e, 'ctrl_jump_cos_mean'))} | — |\n",
        f"| median L2 | {fmt(statistics.median(jump_l2))} | — | — |\n",
        f"| p10 cos | {fmt(quantile(jump_cos, 0.1))} | — | — |\n",
        f"| p90 cos | {fmt(quantile(jump_cos, 0.9))} | — | — |\n\n",
        f"- **{pct(bigger_jump, n_ratio)}** событий: L2 jump > **1.5×** control (n={n_ratio})\n",
        f"- **{pct(smaller_cos, len(jump_vs_traj_cos))}** событий: cos jump **ниже** средней masked-траектории на >0.05\n\n",
        "**Интерпретация:** если unmask = обычный шаг diffusion, jump ≈ control и ≈ traj. "
        "Большой ratio или падение cos vs traj ⇒ сдвиг траектории в момент commit.\n\n",
        "---\n\n",
        "## 2. Jump vs masked-траектория (одна позиция до открытия)\n\n",
        "| Метрика | while masked (traj) | at unmask (jump) | jump / traj |\n",
        "|---------|---------------------|------------------|-------------|\n",
        f"| mean cos | {fmt(agg(all_e, 'traj_cos_mean'))} | {fmt(agg(all_e, 'jump_cos'))} | — |\n",
        f"| mean L2 | {fmt(agg(all_e, 'traj_l2_mean'))} | {fmt(agg(all_e, 'jump_l2'))} | "
        f"**{fmt(agg(all_e, 'jump_vs_traj_l2'))}×** |\n",
        f"| mean Δcos (jump − traj) | — | {fmt(agg(all_e, 'jump_vs_traj_cos'), signed=True)} | — |\n\n",
        f"Средняя длина masked-траектории: **{fmt(agg(all_e, 'traj_steps'), spec='.1f')}** steps "
        f"(0 для step=0 unmask).\n\n",
        "---\n\n",
        "## 3. Разбивка по step unmask\n\n",
        "| step bucket | n | jump cos | ctrl cos | Δ cos | jump L2 | ratio L2 | jump/traj L2 |\n",
        "|-------------|---|----------|----------|-------|---------|----------|--------------|\n",
    ]

    buckets = [
        ("step=0", lambda e: e["step"] == 0),
        ("step 1-7", lambda e: 1 <= e["step"] <= 7),
        ("step 8-31", lambda e: 8 <= e["step"] <= 31),
        ("step 32-63", lambda e: e["step"] >= 32),
    ]
    for name, fn in buckets:
        sub = [e for e in all_e if fn(e)]
        lines.append(
            f"| {name} | {len(sub)} | {fmt(agg(sub, 'jump_cos'))} | {fmt(agg(sub, 'ctrl_jump_cos_mean'))} | "
            f"{fmt(agg(sub, 'delta_jump_cos'), signed=True)} | {fmt(agg(sub, 'jump_l2'))} | "
            f"{fmt(agg(sub, 'ratio_jump_l2'))} | {fmt(agg(sub, 'jump_vs_traj_l2'))} |\n"
        )

    lines += [
        "\n---\n\n",
        "## 4. Multi-token слова vs остальные позиции\n\n",
        "| subset | n | jump cos | ctrl cos | Δ cos | ratio L2 | jump/traj L2 |\n",
        "|--------|---|----------|----------|-------|----------|--------------|\n",
        f"| multi-token word | {len(word_e)} | {fmt(agg(word_e, 'jump_cos'))} | {fmt(agg(word_e, 'ctrl_jump_cos_mean'))} | "
        f"{fmt(agg(word_e, 'delta_jump_cos'), signed=True)} | {fmt(agg(word_e, 'ratio_jump_l2'))} | "
        f"{fmt(agg(word_e, 'jump_vs_traj_l2'))} |\n",
        f"| other positions | {len(all_e) - len(word_e)} | {fmt(agg([e for e in all_e if not e['in_multitoken_word']], 'jump_cos'))} | "
        f"{fmt(agg([e for e in all_e if not e['in_multitoken_word']], 'ctrl_jump_cos_mean'))} | "
        f"{fmt(agg([e for e in all_e if not e['in_multitoken_word']], 'delta_jump_cos'), signed=True)} | "
        f"{fmt(agg([e for e in all_e if not e['in_multitoken_word']], 'ratio_jump_l2'))} | "
        f"{fmt(agg([e for e in all_e if not e['in_multitoken_word']], 'jump_vs_traj_l2'))} |\n\n",
        "---\n\n",
        "## 5. По mask_ratio в момент unmask\n\n",
        "| mask_ratio bucket | n | jump cos | ratio L2 | jump/traj L2 |\n",
        "|-------------------|---|----------|----------|--------------|\n",
    ]
    ratio_buckets = [
        (">0.9", lambda e: e["mask_ratio"] is not None and e["mask_ratio"] > 0.9),
        ("0.5-0.9", lambda e: e["mask_ratio"] is not None and 0.5 <= e["mask_ratio"] <= 0.9),
        ("<0.5", lambda e: e["mask_ratio"] is not None and e["mask_ratio"] < 0.5),
    ]
    for name, fn in ratio_buckets:
        sub = [e for e in all_e if fn(e)]
        lines.append(
            f"| {name} | {len(sub)} | {fmt(agg(sub, 'jump_cos'))} | {fmt(agg(sub, 'ratio_jump_l2'))} | "
            f"{fmt(agg(sub, 'jump_vs_traj_l2'))} |\n"
        )

    lines += ["\n---\n\n", "## 6. Примеры: largest / smallest jump\n\n"]
    top_l2 = sorted(all_e, key=lambda e: e["jump_l2"], reverse=True)[:8]
    low_cos = sorted(all_e, key=lambda e: e["jump_cos"])[:8]
    lines.append("### Top L2 jump\n\n")
    for e in top_l2:
        lines.append(
            f"- step={e['step']} `{e['token']}` conf={e['confidence']:.2f}: "
            f"cos={e['jump_cos']:.3f}, L2={e['jump_l2']:.1f}, ratio={fmt(e['ratio_jump_l2'])}×, "
            f"traj_L2={fmt(e['traj_l2_mean'])}\n"
        )
    lines.append("\n### Lowest cos jump (strongest direction change)\n\n")
    for e in low_cos:
        lines.append(
            f"- step={e['step']} `{e['token']}`: cos={e['jump_cos']:.3f}, L2={e['jump_l2']:.1f}, "
            f"Δcos_vs_ctrl={fmt(e['delta_jump_cos'], signed=True)}\n"
        )

    lines += [
        "\n---\n\n",
        "## 7. Выводы\n\n",
    ]
    mean_ratio = agg(all_e, "ratio_jump_l2")
    mean_jvt = agg(all_e, "jump_vs_traj_l2")
    mean_dcos = agg(all_e, "delta_jump_cos")
    mean_jvtc = agg(all_e, "jump_vs_traj_cos")
    if mean_ratio is not None and mean_jvt is not None:
        if mean_ratio > 1.2 or (mean_jvt and mean_jvt > 1.2):
            lines.append(
                f"- Unmask-переход **крупнее** типичного masked step: ratio vs control ≈ **{mean_ratio:.2f}×**, "
                f"vs masked traj ≈ **{mean_jvt:.2f}×**.\n"
            )
        else:
            lines.append(
                f"- Unmask-переход **сопоставим** с обычным masked step: ratio vs control ≈ **{mean_ratio:.2f}×**.\n"
            )
    if mean_dcos is not None:
        lines.append(f"- Средний Δcos(unmask − control) = **{mean_dcos:+.3f}**.\n")
    if mean_jvtc is not None:
        lines.append(f"- Средний Δcos(jump − masked traj) = **{mean_jvtc:+.3f}**.\n")
    lines.append(
        f"- Ранние unmask (step=0): n={len(early)}, ratio L2={fmt(agg(early, 'ratio_jump_l2'))}×; "
        f"поздние (step≥32): n={len(late)}, ratio L2={fmt(agg(late, 'ratio_jump_l2'))}×.\n"
    )

    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--out", default="hidden_jump_at_unmask.md")
    parser.add_argument("--save-events", default="hidden_jump_events.json")
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

    jobs = collect_trace_jobs(Path(args.checkpoint), tokenizer, args.limit_traces, args.seed)
    print(f"traces={len(jobs)}")

    events: list[dict] = []
    for i, job in enumerate(jobs):
        trace = load_trace(job.trace_path)
        hiddens = cache_trace_hiddens(model, trace, tokenizer, device, args.layer)
        for step_idx, st in enumerate(trace["steps_trace"]):
            for u in st["unmasked"]:
                events.append(
                    analyze_unmask_event(trace, hiddens, job.prompt_len, step_idx, u, job.word_positions)
                )
        del hiddens
        torch.cuda.empty_cache()
        if (i + 1) % 8 == 0 or i + 1 == len(jobs):
            print(f"  {i + 1}/{len(jobs)} traces → {len(events)} events")

    if args.save_events:
        Path(args.save_events).write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

    render_report(events, len(jobs), args.checkpoint, args.layer, Path(args.out))
    print(f"Wrote {args.out} ({len(events)} events)")


if __name__ == "__main__":
    main()

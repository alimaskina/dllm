#!/usr/bin/env python3
"""
Attention to/from multi-token word positions, split by decoding phase.

Phases (per lexical word instance):
  before_first  — snapshot at first unmask step, all word tokens still [MASK]
  between       — after >=1 word token REAL, before all REAL (each intermediate snapshot)
  all_open      — all word tokens REAL

Query groups (outgoing attn mass FROM queries TO word targets):
  other_masked  — completion [MASK], not in this word
  other_open    — completion REAL, not in this word
  word_masked   — word positions still [MASK]
  word_open     — word positions REAL

Word targets split into: word_open / word_masked (within the word).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from analyze_hidden_jump_at_unmask import completion_after_step, prompt_ids_for, rebuild_x
from analyze_multitoken_words import WordInstance, analyze_trace, load_trace
from llada_attn_capture import attn_mass, forward_attn, get_blocks
from multitoken_word_filters import is_lexical

MASK_ID = 126336


@dataclass
class WordPhaseSnapshot:
    word: str
    ntok: int
    phase: str  # before_first | between | all_open
    step: int
    n_word_open: int
    n_word_masked: int
    # outgoing: query_group -> {to_word_open, to_word_masked, to_word_total}
    outgoing: dict[str, dict[str, float]] = field(default_factory=dict)
    # incoming: target_group (word_open/word_masked) <- query_group mass (mean over targets in group)
    incoming: dict[str, dict[str, float]] = field(default_factory=dict)
    # per-role outgoing from other_masked -> left/center/right word positions (by pos order)
    role_out: dict[str, float] = field(default_factory=dict)


QUERY_GROUPS = ("other_masked", "other_open", "word_masked", "word_open")
TARGET_KEYS = ("to_word_open", "to_word_masked")


def classify_queries(
    plen: int,
    gen_length: int,
    word_positions: set[int],
    x_ids: list[int],
) -> dict[str, list[int]]:
    comp_start = plen
    groups: dict[str, list[int]] = {g: [] for g in QUERY_GROUPS}
    for rel in range(gen_length):
        abs_p = comp_start + rel
        tid = int(x_ids[abs_p])
        in_word = rel in word_positions
        if tid == MASK_ID:
            key = "word_masked" if in_word else "other_masked"
        else:
            key = "word_open" if in_word else "other_open"
        groups[key].append(abs_p)
    return groups


def word_target_sets(word_positions: set[int], x_ids: list[int], plen: int) -> tuple[list[int], list[int]]:
    open_abs: list[int] = []
    masked_abs: list[int] = []
    for rel in sorted(word_positions):
        abs_p = plen + rel
        if int(x_ids[abs_p]) == MASK_ID:
            masked_abs.append(abs_p)
        else:
            open_abs.append(abs_p)
    return open_abs, masked_abs


def mean_outgoing(attn: torch.Tensor, queries: list[int], targets: list[int]) -> float | None:
    if not queries or not targets:
        return None
    vals = [attn_mass(attn[q], targets) for q in queries]
    return statistics.mean(vals)


def mean_incoming(attn: torch.Tensor, queries: list[int], targets: list[int]) -> float | None:
    if not queries or not targets:
        return None
    vals = [attn_mass(attn[:, t], queries) for t in targets]
    return statistics.mean(vals)


def role_targets(word_positions: set[int], plen: int) -> dict[str, list[int]]:
    poss = sorted(word_positions)
    if len(poss) == 1:
        return {"left": [plen + poss[0]], "mid": [], "right": [plen + poss[0]]}
    left = plen + poss[0]
    right = plen + poss[-1]
    mid = [plen + p for p in poss[1:-1]]
    return {"left": [left], "mid": mid, "right": [right]}


def analyze_snapshot(
    attn: torch.Tensor,
    plen: int,
    gen_length: int,
    word: str,
    word_positions: set[int],
    x_ids: list[int],
    phase: str,
    step: int,
) -> WordPhaseSnapshot:
    groups = classify_queries(plen, gen_length, word_positions, x_ids)
    t_open, t_masked = word_target_sets(word_positions, x_ids, plen)

    snap = WordPhaseSnapshot(
        word=word,
        ntok=len(word_positions),
        phase=phase,
        step=step,
        n_word_open=len(t_open),
        n_word_masked=len(t_masked),
    )

    for qg in QUERY_GROUPS:
        queries = groups[qg]
        m_open = mean_outgoing(attn, queries, t_open)
        m_masked = mean_outgoing(attn, queries, t_masked)
        total = (m_open or 0.0) + (m_masked or 0.0) if (m_open is not None or m_masked is not None) else None
        snap.outgoing[qg] = {
            "to_word_open": m_open,
            "to_word_masked": m_masked,
            "to_word_total": total,
        }

    # incoming TO word from other groups
    for tg, targets in (("word_open", t_open), ("word_masked", t_masked)):
        snap.incoming[tg] = {}
        for qg in ("other_masked", "other_open", "word_masked", "word_open"):
            snap.incoming[tg][f"from_{qg}"] = mean_incoming(attn, groups[qg], targets)

    # role breakdown: other_masked -> left/mid/right (absolute positions in word)
    roles = role_targets(word_positions, plen)
    om = groups["other_masked"]
    for role, tidx in roles.items():
        snap.role_out[f"other_masked_to_{role}"] = mean_outgoing(attn, om, tidx)

    return snap


def word_snapshots(inst: WordInstance, trace: dict, plen: int) -> list[tuple[str, int, list[int]]]:
    """Return list of (phase, step_idx, completion_tokens)."""
    word_pos = set(inst.positions)
    by_time = sorted(inst.tokens, key=lambda t: (t.step, t.pos_comp))
    first_step = by_time[0].step
    last_step = by_time[-1].step
    out: list[tuple[str, int, list[int]]] = []

    # before first unmask
    out.append(("before_first", first_step, trace["steps_trace"][first_step]["completion_tokens"]))

    # between: before each unmask after the first
    for tok in by_time[1:]:
        out.append(("between", tok.step, trace["steps_trace"][tok.step]["completion_tokens"]))

    # all open: after last word token committed
    steps = trace["steps_trace"]
    if last_step + 1 < len(steps):
        comp = steps[last_step + 1]["completion_tokens"]
    else:
        comp = completion_after_step(trace, last_step)
    out.append(("all_open", last_step + 1, comp))
    return out


def agg_snaps(snaps: list[WordPhaseSnapshot]) -> dict:
    out: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for s in snaps:
        for qg in QUERY_GROUPS:
            for tk in TARGET_KEYS + ("to_word_total",):
                v = s.outgoing.get(qg, {}).get(tk)
                if v is not None:
                    out[f"out_{qg}_{tk}"][s.phase].append(v)
        for tg in ("word_open", "word_masked"):
            for qg in QUERY_GROUPS:
                v = s.incoming.get(tg, {}).get(f"from_{qg}")
                if v is not None:
                    out[f"in_{tg}_from_{qg}"][s.phase].append(v)
        for rk in s.role_out:
            v = s.role_out[rk]
            if v is not None:
                out[rk][s.phase].append(v)
    return out


def mean_or_none(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def fmt(v, signed=False) -> str:
    if v is None:
        return "—"
    return f"{v:+.4f}" if signed else f"{v:.4f}"


def render_report(agg: dict, n_traces: int, n_snaps: int, layers: list[int], out: Path) -> None:
    phases = ["before_first", "between", "all_open"]
    lines = [
        "# Attention и multi-token слова по фазам unmask\n\n",
        f"Traces: **{n_traces}**, snapshots: **{n_snaps}**, layers: **{layers}**\n\n",
        "## Фазы\n\n",
        "| Фаза | Описание |\n|------|----------|\n",
        "| `before_first` | До первого unmask слова — все его токены `[MASK]` |\n",
        "| `between` | После ≥1 REAL, до полного слова — каждый промежуточный step |\n",
        "| `all_open` | Все токены слова REAL |\n\n",
        "## Query groups\n\n",
        "- **other_masked** — другие `[MASK]` в completion (не это слово)\n",
        "- **other_open** — другие REAL в completion\n",
        "- **word_masked** — `[MASK]` внутри слова\n",
        "- **word_open** — REAL внутри слова\n\n",
        "Метрика: mean Σ attn(query→targets) усреднён по queries в группе.\n\n",
        "---\n\n",
    ]

    for phase in phases:
        lines.append(f"## Фаза: {phase}\n\n")
        lines.append("### Outgoing: куда смотрят → на слово (open vs masked)\n\n")
        lines.append("| query group | → word_open | → word_masked | open/(open+masked) | n |\n")
        lines.append("|-------------|-------------|---------------|---------------------|---|\n")
        for qg in QUERY_GROUPS:
            o = agg.get(f"out_{qg}_to_word_open", {}).get(phase, [])
            m = agg.get(f"out_{qg}_to_word_masked", {}).get(phase, [])
            if not o and not m:
                continue
            mo, mm = mean_or_none(o), mean_or_none(m)
            ratio = mo / (mo + mm) if mo is not None and mm is not None and (mo + mm) > 0 else None
            lines.append(
                f"| {qg} | {fmt(mo)} | {fmt(mm)} | {fmt(ratio)} | {max(len(o), len(m))} |\n"
            )

        lines.append("\n### Incoming: кто смотрит на word_open / word_masked\n\n")
        for tg, label in (("word_open", "REAL токены слова"), ("word_masked", "MASK токены слова")):
            lines.append(f"**Target: {label}**\n\n")
            lines.append("| from | mean incoming mass | n |\n|------|-------------------|---|\n")
            for qg in QUERY_GROUPS:
                key = f"in_{tg}_from_{qg}"
                xs = agg.get(key, {}).get(phase, [])
                lines.append(f"| {qg} | {fmt(mean_or_none(xs))} | {len(xs)} |\n")
            lines.append("\n")

        lines.append("### other_masked → позиция в слове (left / mid / right)\n\n")
        lines.append("| target role | mean mass | n |\n|-------------|-----------|---|\n")
        for role in ("left", "mid", "right"):
            xs = agg.get(f"other_masked_to_{role}", {}).get(phase, [])
            lines.append(f"| {role} | {fmt(mean_or_none(xs))} | {len(xs)} |\n")
        lines.append("\n---\n\n")

    # Summary comparison
    lines.append("## Сводка: other_masked предпочитает open или masked siblings?\n\n")
    lines.append("| phase | → open | → masked | open share |\n|-------|--------|----------|------------|\n")
    for phase in phases:
        o = agg.get("out_other_masked_to_word_open", {}).get(phase, [])
        m = agg.get("out_other_masked_to_word_masked", {}).get(phase, [])
        mo, mm = mean_or_none(o), mean_or_none(m)
        ratio = mo / (mo + mm) if mo is not None and mm is not None and (mo + mm) > 0 else None
        lines.append(f"| {phase} | {fmt(mo)} | {fmt(mm)} | {fmt(ratio)} |\n")

    lines.append("\n## Сводка: other_open → word\n\n")
    lines.append("| phase | → word_open | → word_masked | open share |\n")
    lines.append("|-------|-------------|---------------|------------|\n")
    for phase in phases:
        o = agg.get("out_other_open_to_word_open", {}).get(phase, [])
        m = agg.get("out_other_open_to_word_masked", {}).get(phase, [])
        mo, mm = mean_or_none(o), mean_or_none(m)
        ratio = mo / (mo + mm) if mo is not None and mm is not None and (mo + mm) > 0 else None
        lines.append(f"| {phase} | {fmt(mo)} | {fmt(mm)} | {fmt(ratio)} |\n")

    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="16,31")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="word_attention_phases.md")
    parser.add_argument("--save-json", default="word_attention_phases.json")
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
    layer_ids = {int(x.strip()) for x in args.layers.split(",")}
    report_layers = sorted(layer_ids)

    ckpt = Path(args.checkpoint)
    rows = []
    for rf in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(l) for l in rf.open(encoding="utf-8") if l.strip())
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit_traces]

    all_snaps: list[dict] = []
    n_words = 0

    for ri, row in enumerate(rows):
        trace_path = ckpt / row["trace_path"]
        trace = load_trace(trace_path)
        plen = len(prompt_ids_for(trace, tokenizer))
        gen_length = trace["gen_length"]
        instances = [
            inst
            for inst in analyze_trace(trace_path, tokenizer, method="ws", kind_filter="alpha")
            if is_lexical(inst.word) and len(inst.positions) >= 2
        ]
        if not instances:
            continue

        # cache attn per step needed
        needed_steps: set[int] = set()
        word_plans: list[tuple[WordInstance, list]] = []
        for inst in instances:
            snaps = word_snapshots(inst, trace, plen)
            word_plans.append((inst, snaps))
            for _, step_idx, _ in snaps:
                needed_steps.add(step_idx)

        attn_cache: dict[int, dict[int, torch.Tensor]] = {}
        for step_idx in sorted(needed_steps):
            if step_idx < len(trace["steps_trace"]):
                comp = trace["steps_trace"][step_idx]["completion_tokens"]
            else:
                comp = completion_after_step(trace, len(trace["steps_trace"]) - 1)
            x = rebuild_x(trace, comp, tokenizer, args.device)
            attn_cache[step_idx] = forward_attn(model, x, layer_ids)

        for inst, plans in word_plans:
            word_pos = set(inst.positions)
            for phase, step_idx, comp in plans:
                x_ids = prompt_ids_for(trace, tokenizer) + comp
                for layer in report_layers:
                    attn = attn_cache[step_idx][layer]
                    snap = analyze_snapshot(
                        attn, plen, gen_length, inst.word, word_pos, x_ids, phase, step_idx
                    )
                    rec = {
                        "word": snap.word,
                        "ntok": snap.ntok,
                        "phase": snap.phase,
                        "step": snap.step,
                        "layer": layer,
                        "n_word_open": snap.n_word_open,
                        "n_word_masked": snap.n_word_masked,
                        "outgoing": snap.outgoing,
                        "incoming": snap.incoming,
                        "role_out": snap.role_out,
                    }
                    all_snaps.append(rec)
            n_words += 1

        del attn_cache
        torch.cuda.empty_cache()
        if (ri + 1) % 8 == 0 or ri + 1 == len(rows):
            print(f"  {ri + 1}/{len(rows)} traces, {len(all_snaps)} snapshots")

    # aggregate per layer
    for layer in report_layers:
        layer_snaps = [WordPhaseSnapshot(
            word=r["word"], ntok=r["ntok"], phase=r["phase"], step=r["step"],
            n_word_open=r["n_word_open"], n_word_masked=r["n_word_masked"],
            outgoing=r["outgoing"], incoming=r["incoming"], role_out=r["role_out"],
        ) for r in all_snaps if r["layer"] == layer]
        agg = agg_snaps(layer_snaps)
        out_path = Path(args.out.replace(".md", f"_L{layer}.md"))
        render_report(agg, len(rows), len(layer_snaps), [layer], out_path)
        print(f"Wrote {out_path}")

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(all_snaps, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {n_words} words, {len(all_snaps)} snapshots")


if __name__ == "__main__":
    main()

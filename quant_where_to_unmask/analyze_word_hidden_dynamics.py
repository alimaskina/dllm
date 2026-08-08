#!/usr/bin/env python3
"""Hidden dynamics at each token-unmask moment within multi-token words."""

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
class Sample:
    inst: WordInstance
    trace_path: Path
    prompt_len: int


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)).item())


def rebuild_x(trace: dict, step: int, tokenizer, device) -> torch.Tensor:
    st = trace["steps_trace"][step]
    prompt_ids = tokenizer(trace["prompt_text"], add_special_tokens=False)["input_ids"]
    return torch.tensor([prompt_ids + st["completion_tokens"]], dtype=torch.long, device=device)


def logit_lens_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    w = model.model.transformer.ff_out.weight
    return F.linear(hidden.float(), w.float())


def tokens_by_unmask_order(inst: WordInstance) -> list:
    return sorted(inst.tokens, key=lambda t: (t.step, t.pos_comp))


def collect_samples(ckpt: Path, tokenizer, limit_traces: int, limit_words: int, seed: int) -> list[Sample]:
    rows = []
    for rank_file in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(line) for line in rank_file.open(encoding="utf-8") if line.strip())
    rng = random.Random(seed)
    rng.shuffle(rows)
    out: list[Sample] = []
    for row in rows[:limit_traces]:
        trace_path = ckpt / row["trace_path"]
        if not trace_path.exists():
            continue
        trace = load_trace(trace_path)
        plen = len(tokenizer(trace["prompt_text"], add_special_tokens=False)["input_ids"])
        for inst in analyze_trace(trace_path, tokenizer, method="ws", kind_filter="alpha"):
            if is_lexical(inst.word) and len(inst.positions) >= 2:
                out.append(Sample(inst=inst, trace_path=trace_path, prompt_len=plen))
                if len(out) >= limit_words:
                    return out
    return out


def masked_siblings_at_step(trace: dict, step: int, inst: WordInstance, opened_pos: int) -> list[int]:
    comp = trace["steps_trace"][step]["completion"]
    out = []
    for pos in inst.positions:
        if pos == opened_pos:
            continue
        if comp["masked"][pos]:
            out.append(pos)
    return out


def analyze_opening_moment(
    model,
    tokenizer,
    trace: dict,
    inst: WordInstance,
    plen: int,
    opened_tok,
    open_idx: int,
) -> dict:
    """open_idx: 1 = first token opened, 2 = second, etc."""
    step = opened_tok.step
    x = rebuild_x(trace, step, tokenizer, model.device)
    h = model(x, attention_mask=torch.ones_like(x), output_hidden_states=True).hidden_states[-1][0]

    word_abs = {p: plen + p for p in inst.positions}
    opened_abs = word_abs[opened_tok.pos_comp]
    final = {t.pos_comp: t.token_id for t in inst.tokens}

    # all pairwise within word at THIS step
    abs_positions = list(word_abs.values())
    within_pairs = []
    for i, pi in enumerate(abs_positions):
        for pj in abs_positions[i + 1 :]:
            within_pairs.append(cos_sim(h[pi], h[pj]))

    # opened (REAL token) vs still-masked siblings in word
    sib_masked = masked_siblings_at_step(trace, step, inst, opened_tok.pos_comp)
    opened_vs_masked = [cos_sim(h[opened_abs], h[word_abs[p]]) for p in sib_masked]

    # opened vs already-unmasked siblings in word (REAL vs REAL)
    comp = trace["steps_trace"][step]["completion"]
    already_open = [p for p in inst.positions if p != opened_tok.pos_comp and not comp["masked"][p]]
    opened_vs_open = [cos_sim(h[opened_abs], h[word_abs[p]]) for p in already_open]

    # fair control at SAME step: masked sibling (MASK) vs other masked (MASK)
    other_masked_abs = [
        plen + rel
        for rel, m in enumerate(comp["masked"])
        if m and rel not in inst.positions
    ]
    sib_vs_other = []
    paired_delta = []
    for p in sib_masked:
        sib_abs = word_abs[p]
        if not other_masked_abs:
            continue
        vs_other = [cos_sim(h[sib_abs], h[o]) for o in other_masked_abs]
        mean_other = statistics.mean(vs_other)
        sib_vs_other.append(mean_other)
        opened_sib = cos_sim(h[opened_abs], h[sib_abs])
        paired_delta.append(opened_sib - mean_other)

    # logit lens from just-opened hidden -> still-masked siblings
    lens = logit_lens_logits(model, h[opened_abs])
    lens_top1 = int(lens.argmax().item())
    lens_top10 = set(lens.topk(10).indices.tolist())
    sib_hits = []
    for pos in sib_masked:
        tid = final[pos]
        sib_hits.append({
            "final_tok": tokenizer.decode([tid]),
            "top1": tid == lens_top1,
            "top10": tid in lens_top10,
            "rank": int((lens >= lens[tid]).sum().item()),
        })

    # logit lens at each masked sibling position (native)
    native_top1 = []
    for pos in sib_masked:
        ll = logit_lens_logits(model, h[word_abs[pos]])
        native_top1.append(int(ll.argmax().item()) == final[pos])

    n_masked_left = len(sib_masked)
    mask_ratio = trace["steps_trace"][step].get("mask_ratio")

    return {
        "word": inst.word,
        "ntok": len(inst.positions),
        "open_idx": open_idx,  # 1st, 2nd, 3rd token opened
        "opened_tok": opened_tok.token,
        "opened_conf": opened_tok.confidence,
        "step": step,
        "n_masked_siblings_left": n_masked_left,
        "mask_ratio": mask_ratio,
        "within_pairs_mean": statistics.mean(within_pairs) if within_pairs else None,
        "within_pairs_min": min(within_pairs) if within_pairs else None,
        "opened_vs_masked_mean": statistics.mean(opened_vs_masked) if opened_vs_masked else None,
        "opened_vs_open_mean": statistics.mean(opened_vs_open) if opened_vs_open else None,
        "sib_masked_vs_other_mean": statistics.mean(sib_vs_other) if sib_vs_other else None,
        "paired_delta_mean": statistics.mean(paired_delta) if paired_delta else None,
        "lens_top1_tok": tokenizer.decode([lens_top1]),
        "sib_top1": sum(x["top1"] for x in sib_hits) / len(sib_hits) if sib_hits else None,
        "sib_top10": sum(x["top10"] for x in sib_hits) / len(sib_hits) if sib_hits else None,
        "sib_rank_med": statistics.median([x["rank"] for x in sib_hits]) if sib_hits else None,
        "native_top1": sum(native_top1) / len(native_top1) if native_top1 else None,
        "sib_hits": sib_hits,
    }


def analyze_final_word_state(model, tokenizer, trace: dict, inst: WordInstance, plen: int) -> dict | None:
    """All word tokens REAL at end of last unmask step."""
    ordered = tokens_by_unmask_order(inst)
    last = ordered[-1]
    x = rebuild_x(trace, last.step, tokenizer, model.device)
    h = model(x, attention_mask=torch.ones_like(x), output_hidden_states=True).hidden_states[-1][0]
    word_abs = [plen + p for p in inst.positions]
    pairs = []
    for i, pi in enumerate(word_abs):
        for pj in word_abs[i + 1 :]:
            pairs.append(cos_sim(h[pi], h[pj]))
    if not pairs:
        return None
    final = {t.pos_comp: t.token_id for t in inst.tokens}
    # logit lens from each position -> other positions' final tokens
    cross_pos_top1 = []
    for src in inst.positions:
        ll = logit_lens_logits(model, h[plen + src])
        for dst in inst.positions:
            if dst == src:
                continue
            cross_pos_top1.append(int(ll.argmax().item()) == final[dst])
    return {
        "word": inst.word,
        "ntok": len(inst.positions),
        "final_within_mean": statistics.mean(pairs),
        "final_within_min": min(pairs),
        "cross_pos_top1": sum(cross_pos_top1) / len(cross_pos_top1) if cross_pos_top1 else None,
    }


def analyze_word(model, tokenizer, sample: Sample) -> list[dict]:
    trace = load_trace(sample.trace_path)
    ordered = tokens_by_unmask_order(sample.inst)
    return [
        analyze_opening_moment(model, tokenizer, trace, sample.inst, sample.prompt_len, tok, i + 1)
        for i, tok in enumerate(ordered)
    ]


def agg(rows: list[dict], key) -> float | None:
    vals = [key(r) for r in rows if key(r) is not None]
    return statistics.mean(vals) if vals else None


def fmt(v, spec=".3f", signed=False) -> str:
    if v is None:
        return "—"
    if signed:
        return f"{v:+.3f}"
    return format(v, spec)


def render_report(all_events: list[dict], n_words: int, ckpt: str, out: Path, *, final_states: list[dict] | None = None) -> None:
    by_open: dict[int, list[dict]] = defaultdict(list)
    for e in all_events:
        by_open[e["open_idx"]].append(e)

    lines = [
        "# Hidden dynamics: момент открытия каждого токена слова\n\n",
        f"Слов: **{n_words}**, событий открытия: **{len(all_events)}**, checkpoint: `{ckpt}`\n",
        "**Фильтр:** strict lexical (`is_lexical`, без word+punctuation)\n\n",
        "## Методология\n\n",
        "Для каждого multi-token слова сортируем токены по времени unmask (step, pos).\n",
        "Для **каждого** открытия отдельно:\n",
        "1. Восстанавливаем sequence `x` **на конце этого step**\n",
        "2. Forward → hidden states (последний слой)\n",
        "3. Считаем метрики **только в этот момент** (без усреднения по другим шагам)\n\n",
        "**open_idx=1** — момент открытия первого токена слова\n",
        "**open_idx=2** — момент открытия второго\n",
        "**open_idx=3** — третьего (если есть)\n\n",
        "---\n\n",
        "## 1. Cosine similarity hidden'ов (на конкретном step открытия)\n\n",
        "Контроль **честный**: сравниваем `masked sibling ↔ other masked` (оба ещё `[MASK]` на этом шаге).\n",
        "НЕ сравниваем opened↔other — там REAL vs MASK, это разные режимы.\n\n",
        "| Момент | n | within pairs | opened↔masked sib | masked sib↔other | opened↔open | Δ(opened↔sib − sib↔other) |\n",
        "|--------|---|--------------|-------------------|------------------|-------------|------------------------------|\n",
    ]

    for k in sorted(by_open):
        rows = by_open[k]
        lines.append(
            f"| открыт {k}-й | {len(rows)} | "
            f"{fmt(agg(rows, lambda r: r['within_pairs_mean']))} | "
            f"{fmt(agg(rows, lambda r: r['opened_vs_masked_mean']))} | "
            f"{fmt(agg(rows, lambda r: r['sib_masked_vs_other_mean']))} | "
            f"{fmt(agg(rows, lambda r: r['opened_vs_open_mean']))} | "
            f"{fmt(agg(rows, lambda r: r['paired_delta_mean']), signed=True)} |\n"
        )

    lines += [
        "\n**Как читать:**\n",
        "- Всё на **одном step** — момент открытия k-го токена слова\n",
        "- `opened↔masked sib` — hidden только что открытого (REAL) vs hidden sibling ещё `[MASK]`\n",
        "- `masked sib↔other` — **контроль**: hidden sibling `[MASK]` vs hidden другой `[MASK]` позиции (вне слова)\n",
        "- `Δ(opened↔sib − sib↔other)` — насколько opened ближе к sibling, чем sibling к random masked\n",
        "- `opened↔open` — REAL vs REAL (уже открытые куски слова)\n\n",
        "---\n\n",
        "## 2. Logit lens с hidden только что открытого токена\n\n",
        "Проецируем hidden **открытого на этом шаге** токена → предсказываем still-masked siblings.\n\n",
        "| Момент | n | sib top-1 | sib top-10 | median rank | native top1 на sib-позиции |\n",
        "|--------|---|-----------|------------|-------------|----------------------------|\n",
    ]
    for k in sorted(by_open):
        rows = by_open[k]
        ranks = [r["sib_rank_med"] for r in rows if r["sib_rank_med"] is not None]
        st1 = agg(rows, lambda r: r["sib_top1"])
        st10 = agg(rows, lambda r: r["sib_top10"])
        nat = agg(rows, lambda r: r["native_top1"])
        med_rank = f"{statistics.median(ranks):.0f}" if ranks else "—"
        lines.append(
            f"| открыт {k}-й | {len(rows)} | "
            f"{f'{100*st1:.1f}%' if st1 is not None else '—'} | "
            f"{f'{100*st10:.1f}%' if st10 is not None else '—'} | "
            f"{med_rank} | "
            f"{f'{100*nat:.1f}%' if nat is not None else '—'} |\n"
        )

    lines += [
        "\n- **native top1 на sib-позиции** — logit lens с hidden masked sibling (как модель обычно предсказывает)\n\n",
        "---\n\n",
        "## 3. Контекст момента\n\n",
        "| Момент | mean mask_ratio | mean opened conf | mean masked sibs left |\n",
        "|--------|-----------------|------------------|-----------------------|\n",
    ]
    for k in sorted(by_open):
        rows = by_open[k]
        lines.append(
            f"| открыт {k}-й | {fmt(agg(rows, lambda r: r['mask_ratio']))} | "
            f"{fmt(agg(rows, lambda r: r['opened_conf']))} | "
            f"{fmt(agg(rows, lambda r: r['n_masked_siblings_left']))} |\n"
        )

    # per ntok breakdown for open_idx 1 and 2
    lines += ["\n---\n\n## 4. Разбивка по длине слова (1-е открытие)\n\n"]
    ev1 = by_open.get(1, [])
    lines.append("| ntok | n | opened↔masked sib | masked sib↔other | Δ paired |\n|------|---|-------------------|------------------|----------|\n")
    for ntok in sorted(set(e["ntok"] for e in ev1)):
        sub = [e for e in ev1 if e["ntok"] == ntok]
        lines.append(
            f"| {ntok} | {len(sub)} | {fmt(agg(sub, lambda r: r['opened_vs_masked_mean']))} | "
            f"{fmt(agg(sub, lambda r: r['sib_masked_vs_other_mean']))} | "
            f"{fmt(agg(sub, lambda r: r['paired_delta_mean']), signed=True)} |\n"
        )

    lines += ["\n## 5. Примеры по моментам\n\n"]
    for k in [1, 2, 3]:
        rows = by_open.get(k, [])
        if not rows:
            continue
        best = sorted(rows, key=lambda r: r["sib_rank_med"] or 9999)[:3]
        worst = sorted(rows, key=lambda r: r["sib_rank_med"] or 9999)[-3:]
        lines.append(f"### Открытие {k}-го токена — лучшие (низкий rank sibling)\n\n")
        for r in best:
            sh = r["sib_hits"][0] if r["sib_hits"] else {}
            lines.append(
                f"- `{r['word']}`: открыли `{r['opened_tok']}` (conf={r['opened_conf']:.2f}), "
                f"lens_top1=`{r['lens_top1_tok']}`, sibling `{sh.get('final_tok','')}` rank={sh.get('rank','—')}\n"
            )
        lines.append(f"\n### Открытие {k}-го токена — худшие\n\n")
        for r in worst:
            sh = r["sib_hits"][0] if r["sib_hits"] else {}
            lines.append(
                f"- `{r['word']}`: открыли `{r['opened_tok']}`, sibling `{sh.get('final_tok','')}` rank={sh.get('rank','—')}\n"
            )

    # dynamics summary: how metrics change 1->2->3
    lines += ["\n---\n\n## 6. Динамика: как меняется при каждом следующем открытии\n\n"]
    if 1 in by_open and 2 in by_open:
        o1 = agg(by_open[1], lambda r: r["opened_vs_masked_mean"])
        o2 = agg(by_open[2], lambda r: r["opened_vs_masked_mean"])
        c1 = agg(by_open[1], lambda r: r["sib_masked_vs_other_mean"])
        c2 = agg(by_open[2], lambda r: r["sib_masked_vs_other_mean"])
        d1 = agg(by_open[1], lambda r: r["paired_delta_mean"])
        d2 = agg(by_open[2], lambda r: r["paired_delta_mean"])
        t1 = agg(by_open[1], lambda r: r["sib_top10"])
        t2 = agg(by_open[2], lambda r: r["sib_top10"])
        lines.append(f"- opened↔masked sib: 1-е **{o1:.3f}** → 2-е **{o2:.3f}**\n")
        lines.append(f"- masked sib↔other (контроль): 1-е **{c1:.3f}** → 2-е **{c2:.3f}**\n")
        lines.append(f"- Δ paired: 1-е **{d1:+.3f}** → 2-е **{d2:+.3f}**\n")
        lines.append(f"- logit lens sib top-10: 1-е **{100*t1:.1f}%** → 2-е **{100*t2:.1f}%**\n")
    if 3 in by_open:
        w3 = agg(by_open[3], lambda r: r["within_pairs_mean"])
        lines.append(f"- 3-е открытие: within-word cos **{w3:.3f}**, n={len(by_open[3])}\n")

    if final_states:
        fw = [s["final_within_mean"] for s in final_states]
        cp = [s["cross_pos_top1"] for s in final_states if s["cross_pos_top1"] is not None]
        lines += [
            "\n---\n\n## 7. Финальное состояние (все токены слова REAL)\n\n",
            "После unmask **последнего** токена слова: cos между hidden всех позиций (REAL↔REAL).\n\n",
            f"| Метрика | Value |\n|---------|-------|\n",
            f"| mean cos(within word, all REAL) | {statistics.mean(fw):.3f} |\n",
            f"| median | {statistics.median(fw):.3f} |\n",
            f"| logit lens: top1 sibling-токена с hidden каждой позиции | "
            f"{100*statistics.mean(cp):.1f}% |\n\n",
        ]
        ev1_within = agg(by_open.get(1, []), lambda r: r["within_pairs_mean"])
        if ev1_within is not None:
            lines.append(
                f"Сравнение: 1-е открытие within mean **{ev1_within:.3f}** → "
                f"финал **{statistics.mean(fw):.3f}**.\n\n"
            )

    lines += [
        "\n**Интерпретация динамики:** при каждом следующем открытии в слове остаётся меньше masked siblings, "
        "контекст слова обогащается открытыми токенами — ожидаем рост similarity и улучшение предсказания оставшихся кусков.\n",
    ]

    out.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--limit-words", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="word_hidden_dynamics.md")
    parser.add_argument("--save-events", default="")
    args = parser.parse_args()

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base", trust_remote_code=True, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()

    samples = collect_samples(Path(args.checkpoint), tokenizer, args.limit_traces, args.limit_words, args.seed)
    print(f"words={len(samples)}")

    all_events = []
    final_states = []
    for i, s in enumerate(samples):
        trace = load_trace(s.trace_path)
        all_events.extend(analyze_word(model, tokenizer, s))
        fs = analyze_final_word_state(model, tokenizer, trace, s.inst, s.prompt_len)
        if fs:
            final_states.append(fs)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(samples)} → {len(all_events)} events")

    if args.save_events:
        Path(args.save_events).write_text(json.dumps(all_events, ensure_ascii=False, indent=2), encoding="utf-8")

    render_report(all_events, len(samples), args.checkpoint, Path(args.out), final_states=final_states)
    print(f"Wrote {args.out} ({len(all_events)} opening events)")


if __name__ == "__main__":
    main()

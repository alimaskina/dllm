#!/usr/bin/env python3
"""Hidden-state similarity within multi-token words + logit lens from first unmask."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
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


def load_trace_row(ckpt: Path, trace_rel: str) -> dict:
    return load_trace(ckpt / trace_rel)


def prompt_len_for(trace: dict, tokenizer) -> int:
    ids = tokenizer(trace["prompt_text"], add_special_tokens=False)["input_ids"]
    return len(ids)


def rebuild_x(trace: dict, step: int, tokenizer, device) -> torch.Tensor:
    """Reconstruct full sequence x at end of `step` (after unmask in that step)."""
    st = trace["steps_trace"][step]
    comp = st["completion_tokens"]
    prompt_ids = tokenizer(trace["prompt_text"], add_special_tokens=False)["input_ids"]
    x = torch.tensor([prompt_ids + comp], dtype=torch.long, device=device)
    return x


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def logit_lens_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    """hidden: [hidden_dim] -> logits [vocab]"""
    w = model.model.transformer.ff_out.weight  # [vocab, hidden]
    return F.linear(hidden.float(), w.float())


def first_token(inst: WordInstance):
    return min(inst.tokens, key=lambda t: (t.step, t.pos_comp))


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
            if not is_lexical(inst.word) or len(inst.positions) < 2:
                continue
            out.append(Sample(inst=inst, trace_path=trace_path, prompt_len=plen))
            if len(out) >= limit_words:
                return out
    return out


def analyze_sample(model, tokenizer, sample: Sample, layer: int = -1) -> dict | None:
    trace = load_trace(sample.trace_path)
    inst = sample.inst
    ft = first_token(inst)
    plen = sample.prompt_len
    x = rebuild_x(trace, ft.step, tokenizer, model.device)
    attn = torch.ones_like(x)
    out = model(x, attention_mask=attn, output_hidden_states=True)
    h = out.hidden_states[layer][0]  # [seq, dim]

    word_abs = [plen + p for p in inst.positions]
    ft_abs = plen + ft.pos_comp
    final = {t.pos_comp: t.token_id for t in inst.tokens}

    # hidden sim within word
    within = []
    for i, pi in enumerate(word_abs):
        for pj in word_abs[i + 1 :]:
            within.append(cos_sim(h[pi], h[pj]))

    # control: first unmasked vs random masked non-word positions
    st = trace["steps_trace"][ft.step]
    comp = st["completion"]
    word_set = set(inst.positions)
    other_masked = [
        plen + rel
        for rel in range(len(comp["masked"]))
        if comp["masked"][rel] and rel not in word_set
    ]
    cross = [cos_sim(h[ft_abs], h[p]) for p in other_masked[: min(8, len(other_masked))]]

    # logit lens from first unmasked hidden -> predict sibling tokens
    lens_logits = logit_lens_logits(model, h[ft_abs])
    lens_top1 = int(lens_logits.argmax().item())
    lens_top10 = set(lens_logits.topk(10).indices.tolist())
    lens_top50 = set(lens_logits.topk(50).indices.tolist())

    sibling_hits = []
    for pos in inst.positions:
        if pos == ft.pos_comp:
            continue
        tid = final[pos]
        sibling_hits.append(
            {
                "pos": pos,
                "final_id": tid,
                "final_tok": tokenizer.decode([tid]),
                "top1": tid == lens_top1,
                "top10": tid in lens_top10,
                "top50": tid in lens_top50,
                "rank": int((lens_logits >= lens_logits[tid]).sum().item()),
                "logit": float(lens_logits[tid].item()),
            }
        )

    # native model pred at masked sibling positions (from trace)
    trace_preds = []
    for s in inst.sibling_conf_at_first:
        if not s["still_masked"]:
            continue
        trace_preds.append(s["predicted_token_id"] == final[s["pos_comp"]])

    # logit lens FROM each masked sibling position (standard - what model sees there)
    sibling_lens_top1 = []
    for pos in inst.positions:
        if pos == ft.pos_comp:
            continue
        ll = logit_lens_logits(model, h[plen + pos])
        sibling_lens_top1.append(int(ll.argmax().item()) == final[pos])

    return {
        "word": inst.word,
        "ntok": len(inst.positions),
        "first_step": ft.step,
        "first_conf": ft.confidence,
        "within_mean": statistics.mean(within) if within else None,
        "within_min": min(within) if within else None,
        "cross_mean": statistics.mean(cross) if cross else None,
        "cross_n": len(cross),
        "delta_sim": (statistics.mean(within) - statistics.mean(cross)) if within and cross else None,
        "sibling_top1": sum(x["top1"] for x in sibling_hits) / len(sibling_hits) if sibling_hits else None,
        "sibling_top10": sum(x["top10"] for x in sibling_hits) / len(sibling_hits) if sibling_hits else None,
        "sibling_top50": sum(x["top50"] for x in sibling_hits) / len(sibling_hits) if sibling_hits else None,
        "sibling_rank_med": statistics.median([x["rank"] for x in sibling_hits]) if sibling_hits else None,
        "trace_sibling_acc": sum(trace_preds) / len(trace_preds) if trace_preds else None,
        "sibling_pos_lens_top1": sum(sibling_lens_top1) / len(sibling_lens_top1) if sibling_lens_top1 else None,
        "lens_top1_tok": tokenizer.decode([lens_top1]),
        "sibling_hits": sibling_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--limit-words", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=-1, help="hidden layer index (-1 = last)")
    parser.add_argument("--out", default="word_hidden_logit_lens.md")
    args = parser.parse_args()

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    ckpt = Path(args.checkpoint)
    samples = collect_samples(ckpt, tokenizer, args.limit_traces, args.limit_words, args.seed)
    print(f"Collected {len(samples)} word samples")

    results = []
    for i, s in enumerate(samples):
        r = analyze_sample(model, tokenizer, s, layer=args.layer)
        if r:
            results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)}")

    within = [r["within_mean"] for r in results if r["within_mean"] is not None]
    cross = [r["cross_mean"] for r in results if r["cross_mean"] is not None]
    deltas = [r["delta_sim"] for r in results if r["delta_sim"] is not None]
    top1 = [r["sibling_top1"] for r in results if r["sibling_top1"] is not None]
    top10 = [r["sibling_top10"] for r in results if r["sibling_top10"] is not None]
    top50 = [r["sibling_top50"] for r in results if r["sibling_top50"] is not None]
    ranks = [r["sibling_rank_med"] for r in results if r["sibling_rank_med"] is not None]
    trace_acc = [r["trace_sibling_acc"] for r in results if r["trace_sibling_acc"] is not None]
    pos_lens = [r["sibling_pos_lens_top1"] for r in results if r["sibling_pos_lens_top1"] is not None]

    lines = [
        "# Hidden states внутри multi-token слова + logit lens\n\n",
        f"Samples: {len(results)} strict lexical words from `{args.checkpoint}`\n",
        f"Layer: {args.layer} (last hidden before ff_out)\n",
        "**Фильтр:** `is_lexical` (без word+punctuation)\n\n",
        "## 1. Похожи ли hidden'ы токенов одного слова?\n\n",
        "Cosine similarity hidden states в момент первого unmask (после unmask первого токена).\n\n",
        "| Метрика | Value |\n|---------|-------|\n",
        f"| mean cos(within word pairs) | {statistics.mean(within):.3f} |\n",
        f"| median cos(within) | {statistics.median(within):.3f} |\n",
        f"| mean cos(first_unmasked vs other masked) | {statistics.mean(cross):.3f} |\n",
        f"| median cos(cross) | {statistics.median(cross):.3f} |\n",
        f"| mean Δ(within − cross) | {statistics.mean(deltas):+.3f} |\n",
        f"| % words where within > cross | {100*sum(1 for r in results if r['delta_sim'] and r['delta_sim']>0)/len(deltas):.1f}% |\n\n",
        "**Вывод:** ",
    ]
    if statistics.mean(deltas) > 0.02:
        lines.append(
            f"да, но **слабо**: within-word cos выше на Δ≈{statistics.mean(deltas):.2f} "
            f"({100*sum(1 for r in results if r['delta_sim'] and r['delta_sim']>0)/len(deltas):.0f}% слов).\n\n"
        )
    else:
        lines.append("нет значимого эффекта.\n\n")

    lines += [
        "## 2. Logit lens с hidden первого unmask-токена\n\n",
        "Берём `hidden[first_unmasked_pos]`, проецируем через `model.transformer.ff_out` → logits. "
        "Проверяем, попадают ли **финальные** sibling-токены в top-k.\n\n",
        "| Метрика | Value |\n|---------|-------|\n",
        f"| sibling in top-1 | {100*statistics.mean(top1):.1f}% |\n",
        f"| sibling in top-10 | {100*statistics.mean(top10):.1f}% |\n",
        f"| sibling in top-50 | {100*statistics.mean(top50):.1f}% |\n",
        f"| median rank of sibling token | {statistics.median(ranks):.0f} |\n",
        f"| trace sibling pred==final (baseline) | {100*statistics.mean(trace_acc):.1f}% |\n",
        f"| logit lens top1 at sibling **position** hidden | {100*statistics.mean(pos_lens):.1f}% |\n\n",
        "**Интерпретация:** logit lens с позиции первого unmask — это НЕ то же самое, что pred на masked sibling позиции. "
        "Сравниваем оба.\n\n",
        "### Примеры (top sibling rank / hits)\n\n",
    ]

    # interesting examples
    by_rank = sorted(results, key=lambda r: r["sibling_rank_med"] or 9999)
    for r in by_rank[:5]:
        lines.append(f"- `{r['word']}` ({r['ntok']}tok): lens_top1=`{r['lens_top1_tok']}`, ")
        if r["sibling_hits"]:
            sh = r["sibling_hits"][0]
            lines.append(f"sibling `{sh['final_tok']}` rank={sh['rank']} top10={sh['top10']}\n")
    lines.append("\n### Примеры (худшие)\n\n")
    for r in by_rank[-5:]:
        lines.append(f"- `{r['word']}` ({r['ntok']}tok): lens_top1=`{r['lens_top1_tok']}`, ")
        if r["sibling_hits"]:
            sh = r["sibling_hits"][0]
            lines.append(f"sibling `{sh['final_tok']}` rank={sh['rank']}\n")

    lines += [
        "\n## 3. Краткий ответ\n\n",
        f"1. **Hidden similarity:** within-word cos ≈ {statistics.mean(within):.2f} vs cross ≈ {statistics.mean(cross):.2f} "
        f"(Δ={statistics.mean(deltas):+.2f}).\n",
        f"2. **Logit lens с первого unmask:** sibling token в top-1 только {100*statistics.mean(top1):.1f}%, "
        f"в top-10 {100*statistics.mean(top10):.1f}% — **слово целиком из одного hidden не восстанавливается**.\n",
        f"3. На самой masked sibling-позиции logit lens top1 = {100*statistics.mean(pos_lens):.1f}% "
        f"(≈ trace baseline {100*statistics.mean(trace_acc):.1f}%) — модель предсказывает sibling там, где он стоит, не с первого токена.\n",
    ]

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"within={statistics.mean(within):.3f} cross={statistics.mean(cross):.3f} "
          f"lens_top1={100*statistics.mean(top1):.1f}% lens_top10={100*statistics.mean(top10):.1f}%")


if __name__ == "__main__":
    main()

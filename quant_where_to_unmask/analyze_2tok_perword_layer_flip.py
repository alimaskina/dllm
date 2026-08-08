#!/usr/bin/env python3
"""Per-word: does preferred subword flip across layers?"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from analyze_hidden_jump_at_unmask import completion_after_step, prompt_ids_for, rebuild_x
from analyze_multitoken_words import analyze_trace, load_trace
from analyze_word_attention_phases import classify_queries, word_snapshots
from llada_attn_capture import attn_mass, forward_attn, get_blocks
from multitoken_word_filters import is_lexical


def mean_out(attn: torch.Tensor, queries: list[int], targets: list[int]) -> float:
    return statistics.mean(attn_mass(attn[q], targets) for q in queries)


def share(a: float, b: float) -> float:
    t = a + b
    return a / t if t > 0 else 0.5


def first_second_positions(inst) -> tuple[int, int]:
    by_time = sorted(inst.tokens, key=lambda t: (t.step, t.pos_comp))
    return by_time[0].pos_comp, by_time[1].pos_comp


@dataclass
class WordLayerProfile:
    word: str
    phase: str
    shares: dict[int, float] = field(default_factory=dict)  # layer -> first_share

    def prefer_first_layers(self) -> list[int]:
        return [l for l, s in self.shares.items() if s > 0.5]

    def prefer_second_layers(self) -> list[int]:
        return [l for l, s in self.shares.items() if s < 0.5]

    def flips(self) -> bool:
        return bool(self.prefer_first_layers()) and bool(self.prefer_second_layers())

    def strong_flip(self, thr: float = 0.1) -> bool:
        devs = [abs(s - 0.5) for s in self.shares.values()]
        if not devs or max(devs) < thr:
            return False
        return self.flips()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/results_wikitext_fp16_g64_n256")
    parser.add_argument("--limit-traces", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--out", default="word_attention_2tok_perword_layer_flip.md")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    n_layers = len(get_blocks(model))
    layer_ids = set(range(n_layers))

    ckpt = Path(args.checkpoint)
    rows = []
    for rf in sorted(ckpt.glob("rank*.jsonl")):
        rows.extend(json.loads(l) for l in rf.open(encoding="utf-8") if l.strip())
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit_traces]

    # (word, phase, step) -> WordLayerProfile
    profiles: dict[tuple[str, str, int], WordLayerProfile] = {}

    for ri, row in enumerate(rows):
        trace_path = ckpt / row["trace_path"]
        trace = load_trace(trace_path)
        plen = len(prompt_ids_for(trace, tokenizer))
        gen_length = trace["gen_length"]
        instances = [
            inst
            for inst in analyze_trace(trace_path, tokenizer, method="ws", kind_filter="alpha")
            if is_lexical(inst.word) and len(inst.positions) == 2
        ]
        if not instances:
            continue

        needed: set[int] = set()
        plans: list[tuple] = []
        for inst in instances:
            for phase, step_idx, comp in word_snapshots(inst, trace, plen):
                plans.append((inst, phase, step_idx, comp))
                needed.add(step_idx)

        attn_cache: dict[int, dict[int, torch.Tensor]] = {}
        for step_idx in sorted(needed):
            if step_idx < len(trace["steps_trace"]):
                comp = trace["steps_trace"][step_idx]["completion_tokens"]
            else:
                comp = completion_after_step(trace, len(trace["steps_trace"]) - 1)
            x = rebuild_x(trace, comp, tokenizer, args.device)
            attn_cache[step_idx] = forward_attn(model, x, layer_ids)

        for inst, phase, step_idx, comp in plans:
            key = (str(trace_path), inst.word, phase, step_idx)
            if key not in profiles:
                profiles[key] = WordLayerProfile(word=inst.word, phase=phase)
            prof = profiles[key]

            p_first, p_second = first_second_positions(inst)
            t_first, t_second = plen + p_first, plen + p_second
            x_ids = prompt_ids_for(trace, tokenizer) + comp
            groups = classify_queries(plen, gen_length, set(inst.positions), x_ids)
            others = groups["other_masked"] + groups["other_open"]
            if not others:
                continue

            for layer in layer_ids:
                attn = attn_cache[step_idx][layer]
                m_first = mean_out(attn, others, [t_first])
                m_second = mean_out(attn, others, [t_second])
                prof.shares[layer] = share(m_first, m_second)

        del attn_cache
        torch.cuda.empty_cache()
        if (ri + 1) % 16 == 0:
            print(f"  {ri + 1}/{len(rows)}")

    def summarize_phase(phase: str) -> list[str]:
        phase_profs = [p for p in profiles.values() if p.phase == phase and len(p.shares) == n_layers]
        if not phase_profs:
            return [f"## Фаза `{phase}`: нет данных\n\n"]

        n = len(phase_profs)
        flips = [p for p in phase_profs if p.flips()]
        strong = [p for p in phase_profs if p.strong_flip(0.10)]
        unanimous_first = [p for p in phase_profs if all(s > 0.5 for s in p.shares.values())]
        unanimous_second = [p for p in phase_profs if all(s < 0.5 for s in p.shares.values())]

        n_flip_layers = [len(p.prefer_first_layers()) for p in flips]
        # how balanced is the split among flippers
        balance = []
        for p in flips:
            nf = len(p.prefer_first_layers())
            balance.append(min(nf, n_layers - nf) / (n_layers / 2))  # 1 = perfectly balanced flip

        lines = [
            f"## Фаза `{phase}` (n={n} слов)\n\n",
            "Для каждого слова: на каких слоях first_share>0.5 (→1-й unmasked) vs <0.5 (→2-й).\n\n",
            f"- **Flip** (есть и те и другие слои): **{len(flips)}** ({100*len(flips)/n:.1f}%)\n",
            f"- **Strong flip** (|share−0.5|>0.10 хотя бы на одном слое + flip): **{len(strong)}** ({100*len(strong)/n:.1f}%)\n",
            f"- **Всегда 1-й** на всех 32 слоях: **{len(unanimous_first)}** ({100*len(unanimous_first)/n:.1f}%)\n",
            f"- **Всегда 2-й** на всех 32 слоях: **{len(unanimous_second)}** ({100*len(unanimous_second)/n:.1f}%)\n",
        ]
        if flips:
            avg_first = statistics.mean(len(p.prefer_first_layers()) for p in flips)
            lines.append(
                f"- Среди flip-слов: в среднем **{avg_first:.1f}/32** слоёв prefer 1-й\n\n"
            )

        lines.append("### Примеры flip-слов (strong)\n\n")
        strong.sort(key=lambda p: max(abs(s - 0.5) for s in p.shares.values()), reverse=True)
        for p in strong[:10]:
            first_ls = p.prefer_first_layers()
            second_ls = p.prefer_second_layers()
            max_dev = max(abs(s - 0.5) for s in p.shares.values())
            lines.append(
                f"- **`{p.word}`**: 1-й на L{min(first_ls)}–L{max(first_ls)} "
                f"({len(first_ls)} слоёв), 2-й на {len(second_ls)} слоёв, max|Δ|={max_dev:.2f}\n"
            )
            # compact profile
            bar = []
            for layer in range(n_layers):
                s = p.shares[layer]
                bar.append("1" if s > 0.5 else "2")
            lines.append(f"  ```\n  {''.join(bar)}\n  ```\n")
        lines.append("\n")
        return lines

    lines = [
        "# Per-word: меняется ли preferred subword по слоям?\n\n",
        f"Traces: **{len(rows)}**, layers: **0..{n_layers-1}**\n\n",
        "1 = чужие смотрят больше на **1-й unmasked**, 2 = на **2-й**.\n\n",
    ]
    for phase in ("before_first", "between", "all_open"):
        lines += summarize_phase(phase)

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

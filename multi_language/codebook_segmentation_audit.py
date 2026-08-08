#!/usr/bin/env python3
"""
Feasibility §5: controlled codebook task — DLM vs matched AR.

Same underlying symbol, answer encoded as k random subtoken ids (k=1,2,4).
Compare denoising accuracy: diffusion (Dream) vs causal LM (Qwen2.5-7B).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from model_loader import get_dlm_spec, load_causal, load_dlm, resolve_mask_id
from tokenization_audit import TOKENIZER_PRESETS
from trajectory_utils import confidence_trajectory


from data_sources import load_opus_texts
from word_utils import segment_words


def collect_codes_with_k(tokenizer, k: int, *, max_samples: int = 3000) -> list[list[int]]:
    """OPUS EN words that tokenize to exactly k subtokens."""
    texts = load_opus_texts("en", max_samples=max_samples)
    seen: set[tuple[int, ...]] = set()
    found: list[list[int]] = []
    for text in texts:
        for word in segment_words(text, "en"):
            ids = tokenizer.encode(word, add_special_tokens=False)
            if len(ids) != k:
                continue
            key = tuple(ids)
            if key in seen:
                continue
            seen.add(key)
            found.append(ids)
    if len(found) < 30:
        raise RuntimeError(f"Only found {len(found)} OPUS codes with k={k}")
    return found


def sample_code(codes: list[list[int]], rng: random.Random) -> list[int]:
    return list(rng.choice(codes))


def build_prompt(
    tokenizer,
    code_ids: list[int],
    *,
    fewshot: list[list[int]] | None = None,
) -> list[int]:
    """Few-shot in-context: show decoded examples, then empty slot for target."""
    lines = ["Record the code:"]
    if fewshot:
        for ex in fewshot:
            lines.append(tokenizer.decode(ex, skip_special_tokens=True))
    prompt = tokenizer.encode("\n".join(lines), add_special_tokens=False)
    return prompt + code_ids


@torch.no_grad()
def eval_dlm(
    model,
    tokenizer,
    code_ids: list[int],
    *,
    mask_id: int,
    steps: int,
    device: str,
    ar_shift: bool,
    oracle: bool,
    fewshot: list[list[int]] | None = None,
) -> float:
    ids = build_prompt(tokenizer, code_ids, fewshot=fewshot)
    prompt_len = len(ids) - len(code_ids)
    word_pos = [list(range(prompt_len, len(ids)))]
    _, _, acc = confidence_trajectory(
        model,
        ids,
        word_pos,
        oracle=oracle,
        mask_id=mask_id,
        steps=steps,
        remasking="low_confidence",
        device=device,
        ar_shift=ar_shift,
    )
    return acc or 0.0


@torch.no_grad()
def eval_ar(model, tokenizer, code_ids: list[int], device: str, *, fewshot: list[list[int]] | None = None) -> float:
    """Greedy left-to-right generation after prompt."""
    prompt = build_prompt(tokenizer, [], fewshot=fewshot)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in code_ids:
        out = model(input_ids)
        logits = out.logits[0, -1]
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device, dtype=torch.long)],
            dim=1,
        )
    return sum(a == b for a, b in zip(generated, code_ids)) / len(code_ids)


@torch.no_grad()
def eval_ar_teacher(model, tokenizer, code_ids: list[int], device: str, *, fewshot: list[list[int]] | None = None) -> float:
    """Teacher-forced AR: gold prefix for each code token (upper bound)."""
    prompt = build_prompt(tokenizer, [], fewshot=fewshot)
    prefix = list(prompt)
    correct = 0
    for tid in code_ids:
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        logits = model(input_ids).logits[0, -1]
        pred = int(torch.argmax(logits).item())
        if pred == tid:
            correct += 1
        prefix.append(tid)
    return correct / len(code_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlm", default="dream", choices=("dream", "llada", "fast_dllm"))
    parser.add_argument("--ar", default="qwen", help="AR preset key from TOKENIZER_PRESETS")
    parser.add_argument("--k-values", default="1,2,4")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oracle-dlm", action="store_true", help="DLM uses oracle reveal (upper bound)")
    parser.add_argument("--fewshot", type=int, default=0, help="Number of in-context examples per trial")
    parser.add_argument("--compare-oracle", action="store_true", help="Also run DLM with oracle reveal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "codebook_segmentation.json",
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    dlm_id = TOKENIZER_PRESETS[args.dlm]
    ar_id = TOKENIZER_PRESETS[args.ar]
    spec = get_dlm_spec(args.dlm)

    print(f"Loading DLM {dlm_id}...")
    tokenizer = AutoTokenizer.from_pretrained(dlm_id, trust_remote_code=True)
    mask_id = resolve_mask_id(args.dlm, tokenizer, spec)
    dlm = load_dlm(args.dlm, args.device)
    dlm.eval()

    print(f"Loading AR {ar_id}...")
    ar_tok = AutoTokenizer.from_pretrained(ar_id, trust_remote_code=True)
    ar = load_causal(ar_id, args.device, dtype=torch.bfloat16)
    ar.eval()

    codes_by_k = {k: collect_codes_with_k(tokenizer, k) for k in k_values}

    results = []
    for k in k_values:
        dlm_accs: list[float] = []
        dlm_oracle_accs: list[float] = []
        ar_accs: list[float] = []
        ar_tf_accs: list[float] = []
        codes = codes_by_k[k]
        for _ in tqdm(range(args.n_trials), desc=f"k={k}"):
            code = sample_code(codes, rng)
            fewshot = None
            if args.fewshot > 0:
                pool = [c for c in codes if c != code]
                fewshot = [sample_code(pool, rng) for _ in range(min(args.fewshot, len(pool)))]

            dlm_accs.append(
                eval_dlm(
                    dlm,
                    tokenizer,
                    code,
                    mask_id=mask_id,
                    steps=args.steps,
                    device=args.device,
                    ar_shift=spec.ar_shift,
                    oracle=args.oracle_dlm,
                    fewshot=fewshot,
                )
            )
            if args.compare_oracle and not args.oracle_dlm:
                dlm_oracle_accs.append(
                    eval_dlm(
                        dlm,
                        tokenizer,
                        code,
                        mask_id=mask_id,
                        steps=args.steps,
                        device=args.device,
                        ar_shift=spec.ar_shift,
                        oracle=True,
                        fewshot=fewshot,
                    )
                )
            ar_accs.append(eval_ar(ar, ar_tok, code, args.device, fewshot=fewshot))
            ar_tf_accs.append(eval_ar_teacher(ar, ar_tok, code, args.device, fewshot=fewshot))

        row = {
            "k": k,
            "n_trials": args.n_trials,
            "fewshot": args.fewshot,
            "dlm_mean_tok_acc": statistics.mean(dlm_accs),
            "ar_mean_tok_acc": statistics.mean(ar_accs),
            "ar_teacher_mean_tok_acc": statistics.mean(ar_tf_accs),
            "dlm_minus_ar": statistics.mean(dlm_accs) - statistics.mean(ar_accs),
            "ratio_ar_over_dlm": (
                statistics.mean(ar_accs) / statistics.mean(dlm_accs)
                if statistics.mean(dlm_accs) > 0
                else None
            ),
        }
        if dlm_oracle_accs:
            row["dlm_oracle_mean_tok_acc"] = statistics.mean(dlm_oracle_accs)
            row["dlm_oracle_minus_free"] = statistics.mean(dlm_oracle_accs) - statistics.mean(dlm_accs)
        results.append(row)
        msg = (
            f"k={k}: DLM={row['dlm_mean_tok_acc']:.3f}  AR={row['ar_mean_tok_acc']:.3f}  "
            f"AR_tf={row['ar_teacher_mean_tok_acc']:.3f}"
        )
        if dlm_oracle_accs:
            msg += f"  DLM_oracle={row['dlm_oracle_mean_tok_acc']:.3f}"
        print(msg)

    payload = {
        "dlm": args.dlm,
        "ar": args.ar,
        "fewshot": args.fewshot,
        "method": "OPUS word codebook; DLM=denoise, AR=greedy L2R, AR_tf=teacher-forced upper bound",
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Dream-v0-Base-7B WikiText continuations with LLaDA-compatible step traces.

Dream uses AR-shifted logits (logits[i] <- model[i-1]) and maskgit_plus unmasking.
We force ~1 token/step via steps=gen_length and record pre-unmask state each step.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from generate_wikitext import build_prompts


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0):
    if temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        x0 = torch.distributions.Categorical(probs=probs).sample()
        confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    return confidence, x0, probs


@torch.inference_mode()
def generate_traced(
    model,
    tokenizer,
    prompt_ids: list[int],
    *,
    gen_length: int,
    steps: int,
    mask_id: int,
    temperature: float = 0.0,
    device: str,
) -> dict:
    """One sample; returns LLaDA-like trace dict fields for steps_trace."""
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = prompt.shape[1]
    x = F.pad(prompt, (0, gen_length), value=mask_id)
    attention_mask = "full"
    tok_idx = None

    timesteps = torch.linspace(1, 1e-3, steps + 1, device=device)
    steps_trace = []

    for i in range(steps):
        mask_index = x == mask_id
        if not mask_index.any():
            break

        # Dream forward: logits then AR-shift (official generation_utils)
        logits = model(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # full-sequence confidence / preds (for sibling analysis)
        conf_full, x0_full, probs_full = sample_tokens(logits[0], temperature=temperature)
        # margin / entropy optional
        sorted_probs, _ = torch.sort(probs_full, dim=-1, descending=True)
        margins = (sorted_probs[:, 0] - sorted_probs[:, 1]).tolist()
        # neg entropy
        eps = 1e-10
        neg_ent = (probs_full * (probs_full + eps).log()).sum(dim=-1).tolist()

        # transfer count like Dream maskgit_plus
        t = timesteps[i]
        s = timesteps[i + 1]
        num_mask = int(mask_index.sum().item())
        n_transfer = int(num_mask * (1 - s / t)) if i < steps - 1 else num_mask
        n_transfer = max(1, n_transfer) if num_mask > 0 else 0
        # for comparable k=1 when gen_length==steps, clamp to 1 until last dumps rest
        if steps == gen_length and i < steps - 1:
            n_transfer = min(1, num_mask)
        elif steps == gen_length and i == steps - 1:
            n_transfer = num_mask

        mask_logits = logits[mask_index]
        confidence, x0 = sample_tokens(mask_logits, temperature=temperature)[:2]

        full_confidence = torch.full((x.shape[1],), -torch.inf, device=device, dtype=logits.dtype)
        full_confidence[mask_index[0]] = confidence
        _, transfer_pos = torch.topk(full_confidence, n_transfer)

        # record BEFORE commit
        comp_start = prompt_len
        completion_tokens = [int(x[0, comp_start + r].item()) for r in range(gen_length)]
        masked_flags = [completion_tokens[r] == mask_id for r in range(gen_length)]
        pred_ids = [int(x0_full[comp_start + r].item()) for r in range(gen_length)]
        confs = [float(conf_full[comp_start + r].item()) for r in range(gen_length)]

        unmasked = []
        x_new = x.clone()
        for pos in transfer_pos.tolist():
            if not mask_index[0, pos]:
                continue
            # map from mask_index flat — x0 is only over masked positions
            # easier: use x0_full at pos (same sampling as mask path when temp=0)
            tid = int(x0_full[pos].item())
            x_new[0, pos] = tid
            if pos >= comp_start:
                unmasked.append(
                    {
                        "pos": int(pos),
                        "pos_comp": int(pos - comp_start),
                        "token_id": tid,
                        "token": tokenizer.decode([tid]),
                        "confidence": float(conf_full[pos].item()),
                    }
                )

        steps_trace.append(
            {
                "step": i,
                "block": 0,
                "step_in_block": i,
                "k": len(unmasked),
                "n_masked_before": sum(masked_flags),
                "mask_ratio": sum(masked_flags) / gen_length,
                "gap": None,
                "unmasked": unmasked,
                "completion_tokens": completion_tokens,
                "completion": {
                    "masked": masked_flags,
                    "confidence": confs,
                    "predicted_token_id": pred_ids,
                    "token_margin": margins[comp_start : comp_start + gen_length],
                    "neg_entropy": neg_ent[comp_start : comp_start + gen_length],
                },
            }
        )
        x = x_new

    final_completion = [int(x[0, prompt_len + r].item()) for r in range(gen_length)]
    return {
        "steps_trace": steps_trace,
        "final_completion": final_completion,
        "gen_length": gen_length,
        "steps": steps,
        "block_length": gen_length,
        "remasking": "maskgit_plus",
        "mask_id": mask_id,
        "prompt_len": prompt_len,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Dream-org/Dream-v0-Base-7B")
    parser.add_argument("--checkpoint-dir", default="checkpoints/results_dream_wikitext_fp16_g64_n64")
    parser.add_argument("--reuse-prompts", default="checkpoints/results_wikitext_fp16_g64_n256/prompts.json")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = Path(args.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "traces" / "rank0").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    mask_id = tokenizer.mask_token_id
    assert mask_id is not None

    if args.reuse_prompts and Path(args.reuse_prompts).exists():
        prompts = json.loads(Path(args.reuse_prompts).read_text())[: args.n]
        print(f"Reusing {len(prompts)} prompts from {args.reuse_prompts}")
    else:
        prompts = build_prompts(
            dataset="wikitext-103-raw-v1",
            split="validation",
            n=args.n,
            prompt_tokens=48,
            min_chars=120,
            seed=args.seed,
            model_path=args.model,
        )

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    rank_rows = []
    t0 = time.time()
    for i, row in enumerate(prompts):
        prompt_text = row["prompt"] if "prompt" in row else row.get("prompt_text")
        # encode with Dream tokenizer (may differ from LLaDA prompt token count)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        out = generate_traced(
            model,
            tokenizer,
            prompt_ids,
            gen_length=args.gen_length,
            steps=args.steps,
            mask_id=mask_id,
            device=args.device,
        )
        response = tokenizer.decode(out["final_completion"], skip_special_tokens=True)
        trace = {
            "idx": i,
            "doc_id": row.get("doc_id", i),
            "rank": 0,
            "quant": "bf16",
            "model": args.model,
            "gen_length": args.gen_length,
            "steps": args.steps,
            "block_length": args.gen_length,
            "remasking": "maskgit_plus",
            "mask_id": mask_id,
            "prompt_text": prompt_text,
            "gold_answer": None,
            "final_completion": out["final_completion"],
            "steps_trace": out["steps_trace"],
        }
        rel = f"traces/rank0/{i}.json.gz"
        with gzip.open(ckpt / rel, "wt", encoding="utf-8") as f:
            json.dump(trace, f)
        rank_rows.append(
            {
                "idx": i,
                "doc_id": row.get("doc_id", i),
                "prompt_text": prompt_text,
                "response": response,
                "trace_path": rel,
            }
        )
        if (i + 1) % 8 == 0 or i + 1 == len(prompts):
            print(f"  {i + 1}/{len(prompts)}  elapsed={time.time() - t0:.0f}s")

    (ckpt / "rank0.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rank_rows),
        encoding="utf-8",
    )
    summary = {
        "dataset": "wikitext-103-raw-v1",
        "split": "validation",
        "n": len(prompts),
        "model": args.model,
        "quant": "bf16",
        "gen_length": args.gen_length,
        "steps": args.steps,
        "mask_id": mask_id,
        "checkpoint_dir": str(ckpt),
        "elapsed_sec": time.time() - t0,
        "note": "Dream maskgit_plus; logits AR-shifted per official generation_utils; k forced to 1 when steps==gen_length",
    }
    (ckpt / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ckpt / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done → {ckpt} ({len(prompts)} traces, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

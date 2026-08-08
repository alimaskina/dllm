#!/usr/bin/env python3
"""Continued pretraining pilot: IID vs WORD vs SPAN (compute-matched)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from bitsandbytes.optim import AdamW8bit as OptimizerCls
    _OPT_NAME = "AdamW8bit"
except ImportError:
    OptimizerCls = AdamW
    _OPT_NAME = "AdamW"

ML = Path(__file__).resolve().parents[1]
if str(ML) not in sys.path:
    sys.path.insert(0, str(ML))

from causal_pilot.corruption import MASK_ID, Mode, corrupt_sequence
from causal_pilot.dataset import iter_training_examples, load_train_texts
from model_loader import load_llada
from tokenization_audit import TOKENIZER_PRESETS


def mdm_loss(model, noisy: torch.Tensor, gold: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    logits = model(noisy).logits
    # [L, V] at masked positions only
    lm = loss_mask[0].bool()
    if not lm.any():
        return torch.tensor(0.0, device=noisy.device, requires_grad=True)
    return F.cross_entropy(logits[0, lm], gold[0, lm])


def train_step(
    model,
    token_ids: list[int],
    word_positions: list[list[int]],
    *,
    mode: Mode,
    t: float,
    rng: random.Random,
    intervention_prob: float,
    device: str,
) -> tuple[torch.Tensor, dict]:
    res = corrupt_sequence(
        token_ids,
        word_positions,
        mode=mode,
        t=t,
        rng=rng,
        intervention_prob=intervention_prob,
    )
    gold = torch.tensor([token_ids], dtype=torch.long, device=device)
    noisy = torch.tensor([res.noisy_ids], dtype=torch.long, device=device)
    loss_mask = torch.tensor([res.loss_mask], dtype=torch.bool, device=device)
    loss = mdm_loss(model, noisy, gold, loss_mask)
    stats = {
        "intervened": int(res.intervened),
        "n_masked": int(sum(res.loss_mask)),
        "unit_k": res.unit_k,
    }
    return loss, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("iid", "word", "span"))
    parser.add_argument("--model", default="llada")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("/home/alimaskina/dllm/model/LLaDA-8B-Base"),
    )
    parser.add_argument("--train-source", default="opus_en")
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--min-seq-len", type=int, default=16)
    parser.add_argument("--intervention-prob", type=float, default=0.25)
    parser.add_argument("--t-min", type=float, default=0.15)
    parser.add_argument("--t-max", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    model_id = str(args.init_checkpoint) if args.init_checkpoint.exists() else TOKENIZER_PRESETS[args.model]
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = load_llada(model_id, args.device, dtype=torch.float16)
    model.train()

    texts = load_train_texts(args.train_source, max_samples=args.max_train_samples)
    example_iter = iter_training_examples(
        texts,
        tokenizer,
        min_len=args.min_seq_len,
        max_len=args.max_seq_len,
        rng=rng,
    )

    opt = OptimizerCls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: {_OPT_NAME}")

    run_dir = args.out_dir / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["init_checkpoint"] = str(args.init_checkpoint)
    config["mask_id"] = MASK_ID
    (run_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    losses: list[float] = []
    intervened = 0
    total = 0

    pbar = tqdm(range(args.max_steps), desc=args.mode)
    for step in pbar:
        token_ids, word_positions = next(example_iter)
        t = args.t_min + (args.t_max - args.t_min) * rng.random()
        loss, stats = train_step(
            model,
            token_ids,
            word_positions,
            mode=args.mode,
            t=t,
            rng=rng,
            intervention_prob=args.intervention_prob,
            device=args.device,
        )
        if stats["n_masked"] == 0:
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(float(loss.item()))
        intervened += stats["intervened"]
        total += 1
        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{losses[-1]:.3f}", iv=f"{intervened}/{total}")

    # Save weights (HF format)
    save_path = run_dir / "final"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    summary = {
        "mode": args.mode,
        "steps": args.max_steps,
        "mean_loss_last100": sum(losses[-100:]) / min(100, len(losses)) if losses else None,
        "intervention_rate": intervened / max(total, 1),
        "save_path": str(save_path),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {save_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

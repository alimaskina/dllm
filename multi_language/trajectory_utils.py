"""Shared helpers for decoding / oracle trajectory experiments."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

QWT = Path(__file__).resolve().parents[1] / "quant_where_to_unmask"
if str(QWT) not in sys.path:
    sys.path.insert(0, str(QWT))

from oracle_trajectory_random import (
    aggregate_obs,
    compute_mismatch,
    p_train,
    tokenize_sentence,
    tokenize_sentence_with_meta,
)

__all__ = [
    "tokenize_sentence",
    "tokenize_sentence_with_meta",
    "aggregate_obs",
    "compute_mismatch",
    "p_train",
    "confidence_schedule",
    "confidence_trajectory",
    "observations_from_generate_trace",
    "word_positions_in_gen",
]


def confidence_schedule(logits: torch.Tensor, x: torch.Tensor, mask_id: int, remasking: str) -> torch.Tensor:
    x0 = torch.argmax(logits, dim=-1)
    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "topk_margin":
        p = F.softmax(logits, dim=-1)
        top2 = torch.topk(p, k=2, dim=-1).values
        conf = top2[..., 0] - top2[..., 1]
    elif remasking == "l2r":
        # Left-to-right: reveal lower indices first (AR-like unmask order).
        conf = -torch.arange(x.shape[1], device=x.device, dtype=logits.dtype).view(1, -1)
        conf = conf.expand(x.shape[0], -1)
    else:
        conf = torch.rand_like(logits[:, :, 0])
    conf = conf.clone()
    conf[x != mask_id] = -float("inf")
    return conf, x0


@torch.no_grad()
def confidence_trajectory(
    model,
    token_ids: list[int],
    word_positions: list[list[int]],
    *,
    oracle: bool,
    mask_id: int,
    steps: int,
    remasking: str,
    device: str,
    collect_nll: bool = False,
    word_metas: list[dict] | None = None,
    ar_shift: bool = False,
    logits_fn=None,
) -> tuple[list[tuple[int, float, bool]], list[dict] | None, float | None]:
    """
    Reverse trajectory with confidence scheduling.

    oracle=True  → reveal gold tokens (exp 3b)
    oracle=False → reveal model argmax (exp 3c, errors accumulate)

    Returns (observations, nll_records, token_accuracy).
    nll_records: per masked position before reveal — optional for link #4.
    """
    from generate import get_num_transfer_tokens

    n = len(token_ids)
    x = torch.full((1, n), mask_id, dtype=torch.long, device=device)
    gold = torch.tensor([token_ids], dtype=torch.long, device=device)
    masked_set = set(range(n))
    observations: list[tuple[int, float, bool]] = []
    nll_records: list[dict] = [] if collect_nll else None
    n_correct = 0
    meta_by_pos: dict[int, dict] = {}
    if word_metas:
        for wm in word_metas:
            for p in wm["positions"]:
                meta_by_pos[p] = wm

    mask_index = torch.ones((1, n), dtype=torch.bool, device=device)
    num_transfer = get_num_transfer_tokens(mask_index, steps)[0].tolist()

    for step_i in range(steps):
        t = len(masked_set) / n
        for positions in word_positions:
            k = len(positions)
            unresolved = all(p in masked_set for p in positions)
            observations.append((k, t, unresolved))

        k_reveal = num_transfer[step_i]
        if k_reveal <= 0 or not masked_set:
            break

        if logits_fn is not None:
            logits = logits_fn(model, x)
        else:
            logits = model(x).logits
            if ar_shift:
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        conf, x0 = confidence_schedule(logits, x, mask_id, remasking)

        if collect_nll:
            log_probs = F.log_softmax(logits[0], dim=-1)
            for pos in masked_set:
                tid = int(gold[0, pos].item())
                nll = -float(log_probs[pos, tid].item())
                # tag if pos belongs to a fully-unresolved multi-token word
                word_k = 1
                whole_unresolved = True
                for positions in word_positions:
                    if pos in positions:
                        word_k = len(positions)
                        whole_unresolved = all(p in masked_set for p in positions)
                        break
                rec = {
                    "k": word_k,
                    "t": t,
                    "nll": nll,
                    "whole_word_unresolved": whole_unresolved and word_k >= 2,
                    "pos": pos,
                }
                if pos in meta_by_pos:
                    rec["word"] = meta_by_pos[pos]["word"]
                    rec["char_len"] = meta_by_pos[pos]["char_len"]
                nll_records.append(rec)

        _, idxs = torch.topk(conf[0], k=min(k_reveal, len(masked_set)))
        for pos in idxs.tolist():
            if pos not in masked_set:
                continue
            pred = int(x0[0, pos].item())
            target = int(gold[0, pos].item())
            if pred == target:
                n_correct += 1
            x[0, pos] = target if oracle else pred
            masked_set.remove(pos)

    acc = n_correct / n if n else None
    return observations, nll_records, acc


def word_positions_in_gen(
    text: str,
    lang: str,
    tokenizer,
    gen_start: int,
    gen_end: int,
    min_k: int = 2,
) -> list[list[int]]:
    """Map multi-token words fully inside [gen_start, gen_end) to gen-local indices."""
    _, word_pos = tokenize_sentence(text, lang, tokenizer)
    out: list[list[int]] = []
    for positions in word_pos:
        if len(positions) < min_k:
            continue
        if min(positions) >= gen_start and max(positions) < gen_end:
            out.append([p - gen_start for p in positions])
    return out


def observations_from_generate_trace(
    trace: list[dict],
    word_positions_gen: list[list[int]],
    gen_length: int,
) -> list[tuple[int, float, bool]]:
    """Extract (k, t, unresolved) from generate(..., record_trace=True) steps."""
    observations: list[tuple[int, float, bool]] = []
    for step in trace:
        t = step.get("n_masked_before", 0) / max(gen_length, 1)
        masked = step["completion"]["masked"]
        masked_set = {i for i, m in enumerate(masked) if m}
        for positions in word_positions_gen:
            k = len(positions)
            unresolved = all(p in masked_set for p in positions)
            observations.append((k, t, unresolved))
    return observations

"""Compute-matched corruption: IID, WORD (lexical unit), SPAN (contiguous non-word)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

Mode = Literal["iid", "word", "span"]

MASK_ID = 126336


@dataclass
class CorruptionResult:
    noisy_ids: list[int]
    loss_mask: list[bool]
    mode_applied: Mode
    intervened: bool
    unit_k: int | None = None
    target_masked_count: int = 0


def _word_position_sets(word_positions: list[list[int]]) -> set[frozenset[int]]:
    return {frozenset(wp) for wp in word_positions}


def sample_span_positions(
    n: int,
    word_positions: list[list[int]],
    rng: random.Random,
    *,
    min_k: int = 3,
    max_k: int = 8,
) -> list[int] | None:
    """Contiguous span with k>=min_k that is NOT exactly one word's subtokens."""
    if n < min_k:
        return None
    word_sets = _word_position_sets(word_positions)
    candidates: list[list[int]] = []
    for length in range(min_k, min(max_k, n) + 1):
        for start in range(0, n - length + 1):
            positions = list(range(start, start + length))
            if frozenset(positions) in word_sets:
                continue
            candidates.append(positions)
    if not candidates:
        return None
    return list(rng.choice(candidates))


def apply_iid_mask(n: int, t: float, rng: random.Random) -> list[bool]:
    return [rng.random() < t for _ in range(n)]


def rebalance_mask_count(
    mask: list[bool],
    target_count: int,
    protected: set[int],
    rng: random.Random,
) -> list[bool]:
    """Adjust mask so sum(mask)==target_count without touching protected positions."""
    n = len(mask)
    mask = list(mask)
    protected = set(protected)

    def count() -> int:
        return sum(mask)

    while count() > target_count:
        outs = [p for p in range(n) if mask[p] and p not in protected]
        if not outs:
            break
        mask[rng.choice(outs)] = False

    while count() < target_count:
        outs = [p for p in range(n) if not mask[p] and p not in protected]
        if not outs:
            break
        mask[rng.choice(outs)] = True

    return mask


def corrupt_sequence(
    token_ids: list[int],
    word_positions: list[list[int]],
    *,
    mode: Mode,
    t: float,
    rng: random.Random,
    intervention_prob: float = 0.25,
    min_word_k: int = 3,
) -> CorruptionResult:
    """
    Compute-matched corruption.

    1. Sample IID mask with ratio t → target masked count M.
    2. WORD/SPAN (with prob intervention_prob): force chosen unit fully masked,
       then rebalance to keep exactly M masked tokens by unmasking/masking outside.
    """
    n = len(token_ids)
    iid_mask = apply_iid_mask(n, t, rng)
    target_m = sum(iid_mask)

    if mode == "iid" or target_m == 0:
        loss_mask = iid_mask
        noisy = [MASK_ID if m else tid for m, tid in zip(loss_mask, token_ids)]
        return CorruptionResult(noisy, loss_mask, "iid", False, None, target_m)

    intervened = False
    unit: list[int] | None = None
    unit_k: int | None = None

    if rng.random() < intervention_prob:
        if mode == "word":
            cands = [wp for wp in word_positions if len(wp) >= min_word_k]
            if cands:
                unit = list(rng.choice(cands))
                unit_k = len(unit)
                intervened = True
        elif mode == "span":
            unit = sample_span_positions(n, word_positions, rng, min_k=min_word_k)
            if unit:
                unit_k = len(unit)
                intervened = True

    if not intervened or unit is None:
        loss_mask = iid_mask
        noisy = [MASK_ID if m else tid for m, tid in zip(loss_mask, token_ids)]
        return CorruptionResult(noisy, loss_mask, mode, False, None, target_m)

    # Cannot keep unit fully masked if it exceeds IID target count
    if len(unit) > target_m:
        loss_mask = iid_mask
        noisy = [MASK_ID if m else tid for m, tid in zip(loss_mask, token_ids)]
        return CorruptionResult(noisy, loss_mask, mode, False, None, target_m)

    mask = list(iid_mask)
    for p in unit:
        mask[p] = True
    mask = rebalance_mask_count(mask, target_m, set(unit), rng)

    loss_mask = mask
    noisy = [MASK_ID if m else tid for m, tid in zip(loss_mask, token_ids)]
    return CorruptionResult(noisy, loss_mask, mode, True, unit_k, target_m)

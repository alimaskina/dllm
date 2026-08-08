"""Oracle / confidence trajectories for Fast-dLLM v2 (block MDM)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model_loader import FAST_DLLM_MASK_ID
from trajectory_utils import confidence_schedule


def _ar_shift_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


@torch.no_grad()
def fast_dllm_confidence_trajectory(
    model,
    token_ids: list[int],
    word_positions: list[list[int]],
    *,
    oracle: bool,
    mask_id: int = FAST_DLLM_MASK_ID,
    block_size: int = 128,
    small_block_size: int = 32,
    remasking: str = "low_confidence",
    device: str = "cuda",
    threshold: float = 1.0,
) -> tuple[list[tuple[int, float, bool]], float | None]:
    """
    Single-block MDM trajectory on a fully masked span (PoC).

    Mirrors low_confidence unmasking inside one block; records (k,t,unresolved)
    compatible with aggregate_obs / compute_mismatch.
    """
    n = len(token_ids)
    if n > block_size or n < 8:
        return [], None

    # Pad to block_size for Fast-dLLM forward
    pad_len = block_size - n
    gold = token_ids + [mask_id] * pad_len
    x = torch.tensor([gold], dtype=torch.long, device=device)
    x[0, :] = mask_id  # start fully masked (oracle audit like LLaDA)

    gold_t = torch.tensor([gold], dtype=torch.long, device=device)
    masked_set = set(range(n))
    observations: list[tuple[int, float, bool]] = []
    n_correct = 0

    model.eval()

    inner_cap = n * 4
    step = 0
    while masked_set and step < inner_cap:
        t = len(masked_set) / n
        for positions in word_positions:
            k = len(positions)
            unresolved = all(p in masked_set for p in positions)
            observations.append((k, t, unresolved))

        logits = model.forward(
            input_ids=x[:, :block_size],
            use_cache=False,
            block_size=block_size,
        ).logits
        logits = _ar_shift_logits(logits)

        conf, x0 = confidence_schedule(logits, x[:, :block_size], mask_id, remasking)

        # Unmask at most 1 position per outer step (comparable to gradual reveal)
        conf_masked = conf[0, :n].clone()
        conf_masked[x[0, :n] != mask_id] = -float("inf")
        if not (conf_masked > -1e8).any():
            break
        pos = int(torch.argmax(conf_masked).item())
        if pos not in masked_set:
            # fallback: any masked
            pos = min(masked_set)

        pred = int(x0[0, pos].item())
        target = int(gold_t[0, pos].item())
        if pred == target:
            n_correct += 1
        x[0, pos] = target if oracle else pred
        masked_set.discard(pos)
        step += 1

    acc = n_correct / n if n else None
    return observations, acc

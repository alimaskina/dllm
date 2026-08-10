"""Oracle denoising trajectories (confidence / random)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from common import LoadedModel, model_logits


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """LLaDA linear schedule: how many tokens to reveal at each denoising step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@dataclass
class TokenReveal:
    position: int
    reveal_step: int
    reveal_confidence: float | None = None


def _confidence_scores(
    logits: torch.Tensor,
    x: torch.Tensor,
    mask_id: int,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = torch.argmax(logits, dim=-1)
    if policy == "confidence":
        probs = F.softmax(logits, dim=-1)
        conf = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        conf = torch.rand_like(logits[:, :, 0])
    conf = conf.clone()
    conf[x != mask_id] = -float("inf")
    return conf, x0


@torch.no_grad()
def run_oracle_trajectory(
    bundle: LoadedModel,
    input_ids: list[int],
    policy: str = "confidence",
    steps: int = 32,
    rng: random.Random | None = None,
) -> list[TokenReveal]:
    """
    Oracle reverse trajectory: reveal gold tokens according to policy.

    policy="confidence" — low-confidence-first scheduling (model forward each step).
    policy="random"     — random masked positions each step (no model forward).

    Tokens revealed on the same denoising step share the same reveal_step index.
    """
    if policy not in {"confidence", "random"}:
        raise ValueError(f"Unknown policy {policy!r}; expected 'confidence' or 'random'")

    n = len(input_ids)
    device = bundle.device
    mask_id = bundle.mask_token_id
    x = torch.full((1, n), mask_id, dtype=torch.long, device=device)
    gold = torch.tensor([input_ids], dtype=torch.long, device=device)
    masked_set = set(range(n))

    mask_index = torch.ones((1, n), dtype=torch.bool, device=device)
    num_transfer = get_num_transfer_tokens(mask_index, steps)[0].tolist()

    reveals: dict[int, TokenReveal] = {}
    if rng is None:
        rng = random.Random(0)

    for step_i in range(steps):
        k_reveal = num_transfer[step_i]
        if k_reveal <= 0 or not masked_set:
            break

        if policy == "confidence":
            logits = model_logits(bundle, x)
            conf, _x0 = _confidence_scores(logits, x, mask_id, policy)
            _, idxs = torch.topk(conf[0], k=min(k_reveal, len(masked_set)))
            pick = [int(p) for p in idxs.tolist() if p in masked_set]
        else:
            candidates = list(masked_set)
            rng.shuffle(candidates)
            pick = candidates[: min(k_reveal, len(candidates))]

        if policy == "confidence":
            conf_vals = {int(p): float(conf[0, p].item()) for p in pick}
        else:
            conf_vals = {p: None for p in pick}

        for pos in pick:
            x[0, pos] = gold[0, pos]
            masked_set.remove(pos)
            reveals[pos] = TokenReveal(
                position=pos,
                reveal_step=step_i,
                reveal_confidence=conf_vals.get(pos),
            )

    missing = sorted(masked_set)
    if missing:
        raise RuntimeError(f"Oracle trajectory did not reveal positions: {missing}")

    return [reveals[i] for i in range(n)]

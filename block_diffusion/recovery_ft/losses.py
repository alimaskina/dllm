"""Losses for the recovery study: shifted masked CE and generalized JSD.

Both apply the upstream one-position shift — the Fast-dLLM-v2 decoder predicts
the token at position ``p`` from the hidden state at ``p-1``
(``logits = cat([logits[:, :1], logits[:, :-1]])`` in ``batch_sample``) — and
both score only the positions that were actually masked.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def supervised_positions(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shifted (logit_index, target) pairs for the supervised tokens.

    Returns ``(flat_index_into_logits[:, :-1], targets)``.
    """
    shift_labels = labels[:, 1:]
    keep = shift_labels != -100
    return keep, shift_labels


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    p_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Mean CE over masked answer tokens. Returns ``(loss, num_tokens)``.

    ``p_mask`` (per-token masking probability) enables the standard masked
    diffusion 1/p reweighting; upstream's ``loss_function`` does not use it, so
    it is off by default.
    """
    keep, shift_labels = supervised_positions(labels)
    n = int(keep.sum())
    if n == 0:
        return logits.sum() * 0.0, 0

    shift_logits = logits[:, :-1]
    sel_logits = shift_logits[keep].float()
    sel_targets = shift_labels[keep]
    per_token = F.cross_entropy(sel_logits, sel_targets, reduction="none")

    if p_mask is not None:
        w = p_mask[:, 1:][keep].clamp(min=1e-4)
        per_token = per_token / w
    return per_token.mean(), n


def generalized_jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Per-row ``JSD_beta(teacher || student)`` as defined in GKD (arXiv 2306.13649).

    ``JSD_b(P||Q) = b*KL(P||M) + (1-b)*KL(Q||M)``, ``M = b*P + (1-b)*Q``.
    ``b -> 0`` recovers the mode-seeking reverse KL ``KL(Q||P)``; ``b -> 1``
    the forward KL.  Inputs are ``[N, vocab]``; gradients flow through the
    student only (pass a detached teacher).
    """
    log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    log_q = F.log_softmax(student_logits.float(), dim=-1)

    if beta <= 0.0:
        return (log_q.exp() * (log_q - log_p)).sum(-1)
    if beta >= 1.0:
        return (log_p.exp() * (log_p - log_q)).sum(-1)

    log_m = torch.logaddexp(
        log_p + math.log(beta), log_q + math.log1p(-beta)
    )
    kl_pm = (log_p.exp() * (log_p - log_m)).sum(-1)
    kl_qm = (log_q.exp() * (log_q - log_m)).sum(-1)
    return beta * kl_pm + (1.0 - beta) * kl_qm


def masked_jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, int]:
    """Shifted JSD averaged over masked answer tokens. Returns ``(loss, num_tokens)``."""
    keep, _ = supervised_positions(labels)
    n = int(keep.sum())
    if n == 0:
        return student_logits.sum() * 0.0, 0

    s = student_logits[:, :-1][keep]
    t = teacher_logits[:, :-1][keep].detach()
    return generalized_jsd(s, t, beta=beta).mean(), n

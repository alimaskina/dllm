"""MATH training data and the upstream block-diffusion noising scheme.

Prompts come from ``eval.benchmarks.competition_math_prompt`` — the same string
MATH-500 evaluation uses — and are encoded with ``eval.common.encode_prompt``,
so nothing about the prompt format differs between training and the reported
numbers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ..eval.benchmarks import competition_math_prompt
from ..eval.common import encode_prompt

MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

MASK_TOKEN_ID = 151665
STOP_TOKEN_ID = 151645


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def load_math_train(
    *,
    limit: int | None = None,
    seed: int = 1234,
    exclude_math500: bool = True,
) -> list[dict]:
    """Hendrycks MATH *train*, optionally minus anything appearing in MATH-500.

    MATH-500 is drawn from the MATH *test* split so the two are disjoint by
    construction; the filter is a belt-and-braces check that reports what it
    removed rather than assuming.
    """
    from datasets import load_dataset

    blocked: set[str] = set()
    if exclude_math500:
        try:
            blocked = {
                _normalize(r["problem"])
                for r in load_dataset("HuggingFaceH4/MATH-500", split="test")
            }
        except Exception as exc:  # pragma: no cover - offline fallback
            print(f"[data] WARNING: MATH-500 unavailable for dedup ({exc})")

    rows: list[dict] = []
    dropped = 0
    for cfg in MATH_CONFIGS:
        for r in load_dataset("EleutherAI/hendrycks_math", cfg, split="train"):
            if blocked and _normalize(r["problem"]) in blocked:
                dropped += 1
                continue
            rows.append(
                {
                    "problem": r["problem"],
                    "solution": r["solution"],
                    "level": r.get("level"),
                    "subject": cfg,
                }
            )

    random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    print(f"[data] MATH train: {len(rows)} examples ({dropped} MATH-500 duplicates dropped)")
    return rows


@dataclass
class Example:
    input_ids: torch.Tensor   # [L], block-aligned
    labels: torch.Tensor      # [L], -100 outside the answer
    prompt_len: int
    answer_len: int


def encode_example(
    tokenizer,
    problem: str,
    solution: str,
    *,
    block_size: int,
    max_len: int,
    device,
) -> Example | None:
    """Tokenize one problem/solution pair into a block-aligned training example.

    Returns ``None`` when the answer does not fit: truncating a solution would
    teach the model to stop mid-derivation.
    """
    prompt_ids = encode_prompt(
        tokenizer,
        competition_math_prompt(problem),
        max_input_tokens=max_len,
        use_chat_template=True,
        device=device,
    )[0].tolist()
    answer_ids = tokenizer(solution, add_special_tokens=False)["input_ids"] + [STOP_TOKEN_ID]

    total = len(prompt_ids) + len(answer_ids)
    if total > max_len:
        return None
    pad_n = ((total + block_size - 1) // block_size) * block_size - total

    return Example(
        input_ids=torch.tensor(
            prompt_ids + answer_ids + [tokenizer.pad_token_id] * pad_n,
            dtype=torch.long, device=device,
        ),
        labels=torch.tensor(
            [-100] * len(prompt_ids) + list(answer_ids) + [-100] * pad_n,
            dtype=torch.long, device=device,
        ),
        prompt_len=len(prompt_ids),
        answer_len=len(answer_ids),
    )


@dataclass
class NoisyBatch:
    """Doubled ``[x_t ; x_0]`` batch ready for ``blockdiff_logits``."""

    input_ids: torch.Tensor        # [B, 2L]
    labels: torch.Tensor           # [B, L], -100 except on masked answer tokens
    is_masked_token: torch.Tensor  # [B, 2L], True on masked x_t positions
    p_mask: torch.Tensor           # [B, L]


def make_noisy_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    block_size: int,
    mask_id: int = MASK_TOKEN_ID,
    generator: torch.Generator | None = None,
    complementary: bool = True,
    eps: float = 1e-3,
) -> NoisyBatch:
    """Upstream Fast-dLLM-v2 noising, made explicit and reproducible.

    One masking rate ``t ~ U(eps, 1)`` per *block*; every answer token in that
    block is masked independently with probability ``t``; only masked tokens are
    supervised. ``complementary`` appends the mirror batch, as upstream does.
    """
    b, seq = input_ids.shape
    if seq % block_size:
        raise ValueError(f"sequence length {seq} is not a multiple of block_size {block_size}")
    device = input_ids.device
    n_blocks = seq // block_size

    t = torch.rand((b, n_blocks, 1), device=device, generator=generator)
    p = ((1 - eps) * t + eps).expand(b, n_blocks, block_size).reshape(b, seq)
    mask_indices = torch.rand((b, seq), device=device, generator=generator) < p

    answer = labels != -100
    variants = [mask_indices] + ([~mask_indices] if complementary else [])

    ids_out, lab_out, msk_out, p_out = [], [], [], []
    for mi in variants:
        mi = mi & answer                      # never mask the prompt or padding
        noisy = torch.where(mi, mask_id, input_ids)
        lab = labels.clone()
        lab[~mi] = -100                       # supervise the masked tokens only
        ids_out.append(torch.cat([noisy, input_ids], dim=1))
        msk_out.append(torch.cat([noisy == mask_id, torch.zeros_like(mi)], dim=1))
        lab_out.append(lab)
        p_out.append(p)

    return NoisyBatch(
        input_ids=torch.cat(ids_out, dim=0),
        labels=torch.cat(lab_out, dim=0),
        is_masked_token=torch.cat(msk_out, dim=0),
        p_mask=torch.cat(p_out, dim=0),
    )

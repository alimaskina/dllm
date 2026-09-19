"""MATH training data + the upstream block-diffusion noising scheme.

Prompts are built with the *same* ``build_chat_prompt`` the evaluation harness
uses (``sparse_kv_exp/benchmark_utils.py``), so nothing about the prompt format
differs between training and the GSM8K / MATH500 / LongBench evaluations.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SPARSE = _HERE.parent / "sparse_kv_exp"
for _p in (str(_SPARSE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_utils import build_chat_prompt  # noqa: E402

MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645


def _normalize_problem(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def load_math_train(
    *,
    num_examples: int | None = None,
    seed: int = 1234,
    exclude_math500: bool = True,
) -> list[dict]:
    """Hendrycks MATH *train* split, optionally minus anything in MATH-500.

    MATH-500 is drawn from the MATH *test* split, so the two are disjoint by
    construction; the filter is a belt-and-braces check that also reports how
    many problems it removed.
    """
    from datasets import load_dataset

    blocked: set[str] = set()
    if exclude_math500:
        try:
            ds500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
            blocked = {_normalize_problem(r["problem"]) for r in ds500}
        except Exception as exc:  # pragma: no cover - offline fallback
            print(f"[data] WARNING: could not load MATH-500 for dedup ({exc})")

    rows: list[dict] = []
    dropped = 0
    for cfg in MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        for r in ds:
            if blocked and _normalize_problem(r["problem"]) in blocked:
                dropped += 1
                continue
            rows.append(
                {
                    "problem": r["problem"],
                    "solution": r["solution"],
                    "level": r.get("level"),
                    "type": r.get("type"),
                    "subject": cfg,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_examples is not None:
        rows = rows[:num_examples]
    print(
        f"[data] MATH train: {len(rows)} examples "
        f"({dropped} dropped as MATH-500 duplicates)"
    )
    return rows


@dataclass
class Example:
    input_ids: torch.Tensor  # [L]
    labels: torch.Tensor     # [L], -100 outside the answer
    prompt_len: int
    answer_len: int


def encode_example(
    tokenizer,
    problem: str,
    solution: str,
    *,
    block_size: int,
    max_len: int,
    device=None,
) -> Example | None:
    """Tokenize one problem/solution pair into a block-aligned training example.

    Returns ``None`` when the answer does not fit — truncating a solution would
    teach the model to stop mid-derivation.
    """
    prompt = build_chat_prompt(tokenizer, problem)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(solution, add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [FAST_DLLM_STOP_TOKEN]

    total = len(prompt_ids) + len(answer_ids)
    if total > max_len:
        return None

    padded = ((total + block_size - 1) // block_size) * block_size
    pad_n = padded - total

    ids = prompt_ids + answer_ids + [tokenizer.pad_token_id] * pad_n
    labels = (
        [-100] * len(prompt_ids)
        + list(answer_ids)
        + [-100] * pad_n
    )
    return Example(
        input_ids=torch.tensor(ids, dtype=torch.long, device=device),
        labels=torch.tensor(labels, dtype=torch.long, device=device),
        prompt_len=len(prompt_ids),
        answer_len=len(answer_ids),
    )


@dataclass
class NoisyBatch:
    """Doubled ``[x_t ; x_0]`` batch ready for :func:`blockdiff.blockdiff_logits`."""

    input_ids: torch.Tensor       # [B, 2L]
    labels: torch.Tensor          # [B, L], -100 except on masked answer tokens
    is_masked_token: torch.Tensor  # [B, 2L], True on masked x_t positions
    p_mask: torch.Tensor          # [B, L] per-token masking probability


def make_noisy_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    block_size: int,
    mask_id: int = FAST_DLLM_MASK_ID,
    generator: torch.Generator | None = None,
    complementary: bool = True,
    eps: float = 1e-3,
) -> NoisyBatch:
    """Upstream Fast-dLLM-v2 noising, made explicit and reproducible.

    One masking rate ``t ~ U(eps, 1)`` is drawn per *block*, every answer token
    in that block is masked independently with probability ``t``, and only the
    masked tokens are supervised.  ``complementary=True`` appends the mirror
    batch (the complement of every masking decision), as upstream does.
    """
    b, seq = input_ids.shape
    if seq % block_size != 0:
        raise ValueError(f"sequence length {seq} not divisible by block_size {block_size}")
    device = input_ids.device
    n_blocks = seq // block_size

    t = torch.rand((b, n_blocks, 1), device=device, generator=generator)
    p = ((1 - eps) * t + eps).expand(b, n_blocks, block_size).reshape(b, seq)
    draw = torch.rand((b, seq), device=device, generator=generator)
    mask_indices = draw < p

    variants = [mask_indices]
    if complementary:
        variants.append(~mask_indices)

    out_ids, out_labels, out_is_masked, out_p = [], [], [], []
    answer = labels != -100
    for mi in variants:
        mi = mi & answer                       # never mask the prompt or padding
        noisy = torch.where(mi, mask_id, input_ids)
        lab = labels.clone()
        lab[~mi] = -100                        # supervise the masked tokens only
        doubled = torch.cat([noisy, input_ids], dim=1)
        is_masked = torch.cat([noisy == mask_id, torch.zeros_like(mi)], dim=1)
        out_ids.append(doubled)
        out_labels.append(lab)
        out_is_masked.append(is_masked)
        out_p.append(p)

    return NoisyBatch(
        input_ids=torch.cat(out_ids, dim=0),
        labels=torch.cat(out_labels, dim=0),
        is_masked_token=torch.cat(out_is_masked, dim=0),
        p_mask=torch.cat(out_p, dim=0),
    )

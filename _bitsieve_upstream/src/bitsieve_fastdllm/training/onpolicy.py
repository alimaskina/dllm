"""On-policy sampling from the student, through the real BitSieve decoder.

Branch C samples with a dense bf16 cache; branch E samples through the same
``BitSieveGenerator`` the evaluation runs — packed low-bit cache, one-shot
selection, real kernels — so the sequences carry the student's genuine
denoising trajectory under the regime it is being trained for.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from ..config import ExperimentConfig
from ..eval.benchmarks import competition_math_prompt
from ..eval.common import encode_prompt
from ..runtime.generator import BitSieveGenerator


@dataclass
class Sample:
    problem: str
    completion: str
    generated_tokens: int
    blocks_sparse: int | None
    blocks_dense_bypass: int | None


def sampling_config(
    base: ExperimentConfig,
    *,
    dense: bool,
    max_new_tokens: int,
) -> ExperimentConfig:
    """Derive the config the student samples under.

    ``dense=True`` gives the bf16, no-selection control (branch C); otherwise the
    config is used as-is, which for the shipped ``proposed_*`` files means the
    packed low-bit cache with the budget the evaluation reports.
    """
    cfg = copy.deepcopy(base)
    cfg.generation.max_new_tokens = int(max_new_tokens)
    cfg.collect_diagnostics = False
    cfg.coverage_diagnostics = False
    if dense:
        cfg.semantic = "dense"
        cfg.quant.k_bits = 16
        cfg.quant.v_bits = 16
    return cfg


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    problems: list[str],
    *,
    config: ExperimentConfig,
    max_input_tokens: int = 1024,
) -> list[Sample]:
    """Sample one completion per problem. The model is unpatched again on exit."""
    gen = BitSieveGenerator(model, tokenizer, config)
    out: list[Sample] = []
    try:
        for problem in problems:
            ids = encode_prompt(
                tokenizer,
                competition_math_prompt(problem),
                max_input_tokens=max_input_tokens,
                use_chat_template=True,
                device=next(model.parameters()).device,
            )
            res = gen.generate(ids)
            m = res.metrics
            out.append(
                Sample(
                    problem=problem,
                    completion=res.texts[0],
                    generated_tokens=int(res.generated_ids.shape[1]),
                    blocks_sparse=m.get("blocks_sparse"),
                    blocks_dense_bypass=m.get("blocks_dense_bypass"),
                )
            )
    finally:
        gen.patch.unpatch()
    return out

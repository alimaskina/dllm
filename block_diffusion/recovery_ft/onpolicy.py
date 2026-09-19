"""On-policy sampling from the student, dense or through the real sparse+quant loop.

Branch C samples with the student's ordinary dense decoding; branch E samples
with ``batch_sample_sparse_kv`` — the *same* code path the evaluation harness
runs — so the sequences carry the student's real denoising trajectory under a
quantized cache with top-k selection in the loop.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent / "sparse_kv_exp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark_utils import build_chat_prompt  # noqa: E402
from config import ExperimentConfig, PrecisionConfig, SelectorConfig  # noqa: E402
from generation import batch_sample_sparse_kv  # noqa: E402
from model_utils import default_small_block_size  # noqa: E402

FAST_DLLM_STOP_TOKEN = 151645


def sparse_gen_config(
    *,
    k_bits: str = "4",
    v_bits: str = "4",
    topk: int | None = 64,
    topk_pct: float | None = None,
    block_size: int = 32,
    max_new_tokens: int = 512,
) -> ExperimentConfig:
    """The inference config branch E samples under (matches the eval grid)."""
    sel: dict = dict(mode="all_mean", per_head=True)
    if topk_pct is not None:
        sel["topk_pct"] = topk_pct
        sel["topk"] = 256
    else:
        sel["topk"] = int(topk)
    return ExperimentConfig(
        name=f"onpolicy_k{k_bits}v{v_bits}_"
             + (f"pct{topk_pct}" if topk_pct is not None else f"top{topk}"),
        baseline="sparse_quant_kv",
        sparse_old_cache=True,
        selector=SelectorConfig(**sel),
        selector_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
        exec_precision=PrecisionConfig(k_bits=k_bits, v_bits=v_bits, q_bits="fp16"),
        block_size=block_size,
        small_block_size=default_small_block_size(block_size),
        threshold=1.0,
        max_new_tokens=max_new_tokens,
        log_selected_indices=False,
        save_full_cost_steps=False,
    )


@dataclass
class Sample:
    problem: str
    completion: str
    gen_tokens: int
    coverage: float | None


@torch.no_grad()
def generate_batch(
    base_model,
    tokenizer,
    problems: list[str],
    *,
    exp_cfg: ExperimentConfig | None,
    upstream_batch_sample,
    block_size: int = 32,
    max_new_tokens: int = 512,
) -> list[Sample]:
    """Sample one completion per problem. ``exp_cfg=None`` -> dense decoding."""
    if exp_cfg is None:
        base_model.mdm_sample = types.MethodType(upstream_batch_sample, base_model)
    else:
        base_model.mdm_sample = types.MethodType(batch_sample_sparse_kv, base_model)

    out: list[Sample] = []
    for problem in problems:
        prompt = build_chat_prompt(tokenizer, problem)
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(base_model.device)
        plen = int(ids.shape[1])
        common = dict(
            input_ids=ids,
            tokenizer=tokenizer,
            block_size=block_size,
            small_block_size=default_small_block_size(block_size),
            max_new_tokens=max_new_tokens,
            min_len=plen,
            seq_len=torch.tensor([plen], device=base_model.device),
            threshold=1.0,
        )
        log: list = []
        if exp_cfg is None:
            res = base_model.mdm_sample(**common)
        else:
            res = base_model.mdm_sample(**common, exp_config=exp_cfg, experiment_log=log)

        seq = res[0]
        text = tokenizer.decode(seq[plen:], skip_special_tokens=True)
        cov = None
        if log:
            vals = [
                layer["attention_mass_captured"]
                for blk in log[-1].get("blocks", [])
                for layer in blk.get("layers", {}).values()
                if "attention_mass_captured" in layer
            ]
            cov = sum(vals) / len(vals) if vals else None
        out.append(
            Sample(
                problem=problem,
                completion=text,
                gen_tokens=int(seq.shape[0] - plen),
                coverage=cov,
            )
        )
    return out

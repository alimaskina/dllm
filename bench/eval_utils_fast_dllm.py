"""Shared GSM8K eval helpers for sparse_kv_exp runs."""

from __future__ import annotations

import random
import re
import types

import numpy as np
import torch
from lm_eval import tasks

from config import ExperimentConfig
from fast_dllm_sparse import batch_sample_sparse_kv


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_answer(text: str) -> str | None:
    m = re.search(r"####\s*(-?[\d.,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return nums[-1] if nums else None


def format_gsm8k_question(question: str) -> str:
    ctx = f"Question: {question}\nAnswer:"
    return ctx.replace(
        "Answer:",
        "Please reason step by step, and put your final answer within \\boxed{{}}.",
    )


def build_chat_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": format_gsm8k_question(question)}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load_gsm8k_samples(n: int, seed: int) -> list[dict]:
    task_dict = tasks.get_task_dict(["gsm8k"])
    task = task_dict["gsm8k"]
    task._config.num_fewshot = 0
    task.set_fewshot_seed(seed=seed)
    task.build_all_requests(limit=n, rank=0, world_size=1)
    samples = []
    for i, inst in enumerate(task._instances):
        doc = inst.doc
        samples.append(
            {
                "idx": i,
                "question": doc["question"],
                "gold_answer": task.doc_to_target(doc),
                "gold_num": extract_answer(task.doc_to_target(doc)),
            }
        )
    return samples


def attach_sampler(model, exp_cfg: ExperimentConfig, upstream_batch_sample) -> None:
    if exp_cfg.baseline == "original":
        model.mdm_sample = types.MethodType(upstream_batch_sample, model)
    else:
        model.mdm_sample = types.MethodType(batch_sample_sparse_kv, model)

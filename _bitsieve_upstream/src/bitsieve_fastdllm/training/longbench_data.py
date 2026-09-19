"""LongBench training examples: a cached prompt plus one answer window.

LongBench ships only a test split, so "train on some tasks" means two disjoint
splits at once:

* **across tasks** -- train on a chosen subset, hold the rest out entirely, so
  transfer to an unseen task is measurable;
* **within a trained task** -- train on the first ``per_task`` examples and
  evaluate from ``--example-offset`` onward, so a task that *was* trained on is
  still scored on examples the adapter never saw.

The layout mirrors what the decoder does with a prompt: the block-aligned part
goes into the cache, and the ragged remainder starts the first generated block.
Those remainder tokens are context, not supervision, so they stay unmasked and
carry ``-100``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..eval.benchmarks import load_benchmark, max_new_tokens_for
from ..eval.common import encode_prompt

STOP_TOKEN_ID = 151645

# Multi-hop passage QA to train on; MuSiQue is the same family held out, and
# NarrativeQA is a different genre and a much longer context.
DEFAULT_TRAIN_TASKS = ("2wikimqa", "hotpotqa")
DEFAULT_HELDOUT_TASKS = ("musique", "narrativeqa")


@dataclass
class LongExample:
    """One long-context training example."""

    prefix_ids: torch.Tensor   # [P], P a multiple of block_size -> the cache
    window_ids: torch.Tensor   # [A], the ragged prompt tail plus the answer
    labels: torch.Tensor       # [A], -100 outside the answer
    task: str
    example_id: str
    prefix_len: int
    answer_len: int


def load_longbench_train(
    tokenizer,
    *,
    tasks: tuple[str, ...] = DEFAULT_TRAIN_TASKS,
    per_task: int = 150,
    block_size: int = 32,
    max_cache_tokens: int = 32768,
    min_answer_tokens: int = 1,
    device=None,
) -> list[LongExample]:
    """Build training examples from the first ``per_task`` examples of each task.

    Prompts are truncated with the evaluation's own ``encode_prompt`` policy and
    the evaluation's own budget, so training sees the same text the benchmark
    would show it.
    """
    out: list[LongExample] = []
    for task in tasks:
        max_input = max_cache_tokens - max_new_tokens_for(task)
        kept = 0
        for ex in load_benchmark(task, tokenizer=tokenizer, limit=per_task, split="test"):
            if not ex.references:
                continue
            answer = str(ex.references[0]).strip()
            if not answer:
                continue

            prompt_ids = encode_prompt(
                tokenizer, ex.prompt, max_input_tokens=max_input,
                use_chat_template=True, device=device,
            )[0]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            answer_ids = answer_ids + [STOP_TOKEN_ID]
            if len(answer_ids) < min_answer_tokens:
                continue

            # The decoder caches floor(len/bs)*bs prompt tokens and starts the
            # first generated block with what is left over.
            p = (int(prompt_ids.shape[0]) // block_size) * block_size
            if p == 0:
                continue
            remainder = prompt_ids[p:].tolist()

            window = remainder + list(answer_ids)
            pad = (-len(window)) % block_size
            labels = [-100] * len(remainder) + list(answer_ids) + [-100] * pad
            window = window + [tokenizer.pad_token_id] * pad

            out.append(
                LongExample(
                    prefix_ids=prompt_ids[:p].to(device),
                    window_ids=torch.tensor(window, dtype=torch.long, device=device),
                    labels=torch.tensor(labels, dtype=torch.long, device=device),
                    task=task,
                    example_id=ex.example_id,
                    prefix_len=p,
                    answer_len=len(answer_ids),
                )
            )
            kept += 1
        print(f"[data] longbench {task}: {kept} training examples")
    return out

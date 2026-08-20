"""LongBench data loading, prompting, and scoring helpers."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import torch

_LB_DIR = Path(__file__).resolve().parent
_DATA_DIR = _LB_DIR / "data" / "longbench" / "data"
_PROMPTS = json.loads((_LB_DIR / "longbench_configs.json").read_text())
_MAX_GEN = json.loads((_LB_DIR / "longbench_maxlen.json").read_text())

DEFAULT_TASKS = ["2wikimqa", "narrativeqa", "qmsum", "repobench-p"]

# repobench-p: no chat template (LongBench convention)
NO_CHAT_TASKS = {"repobench-p", "lcc"}


def load_longbench_samples(
    task: str,
    *,
    num_examples: int | None = None,
    seed: int = 1234,
) -> list[dict]:
    path = _DATA_DIR / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"LongBench data missing: {path}. Run data download first.")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if num_examples is not None and num_examples < len(rows):
        import random

        rng = random.Random(seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        idx = sorted(idx[:num_examples])
        rows = [rows[i] for i in idx]
    samples = []
    for i, row in enumerate(rows):
        samples.append(
            {
                "idx": i,
                "task": task,
                "_id": row.get("_id"),
                "input": row["input"],
                "context": row["context"],
                "answers": row["answers"],
                "length": row.get("length", 0),
                "all_classes": row.get("all_classes"),
            }
        )
    return samples


def build_prompt(task: str, sample: dict) -> str:
    fmt = _PROMPTS[task]
    return fmt.format(context=sample["context"], input=sample["input"])


def truncate_prompt_middle(tokenizer, prompt: str, max_prompt_tokens: int) -> tuple[str, int]:
    """LongBench-style middle truncation to fit context window."""
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) <= max_prompt_tokens:
        return prompt, len(ids)
    half = max_prompt_tokens // 2
    truncated = tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(
        ids[-half:], skip_special_tokens=True
    )
    return truncated, max_prompt_tokens


def prepare_inputs(
    tokenizer,
    task: str,
    sample: dict,
    *,
    model_max_tokens: int = 32768,
    max_gen: int | None = None,
) -> tuple[torch.Tensor, str, int]:
    """Tokenize LongBench example; return input_ids, prompt_text, prompt_token_len."""
    max_gen = max_gen or _MAX_GEN[task]
    max_prompt_tokens = model_max_tokens - max_gen - 64

    prompt = build_prompt(task, sample)
    prompt, _ = truncate_prompt_middle(tokenizer, prompt, max_prompt_tokens)

    if task in NO_CHAT_TASKS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    else:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(text, return_tensors="pt").input_ids

    return input_ids, prompt, int(input_ids.shape[1])


def max_new_tokens_for_task(task: str) -> int:
    return int(_MAX_GEN[task])


# ── minimal LongBench metrics (EN tasks) ───────────────────────────────────

def _normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1_score(prediction: str, ground_truth: str, **_kw) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_score(prediction: str, ground_truth: str, **_kw) -> float:
    try:
        from rouge import Rouge

        rouge = Rouge()
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        return float(scores["rouge-l"]["f"])
    except Exception:
        # fallback: token F1
        return qa_f1_score(prediction, ground_truth)


def code_sim_score(prediction: str, ground_truth: str, **_kw) -> float:
    try:
        from fuzzywuzzy import fuzz

        return float(fuzz.ratio(prediction, ground_truth) / 100.0)
    except Exception:
        import difflib

        return difflib.SequenceMatcher(None, prediction, ground_truth).ratio()


TASK_METRIC = {
    "2wikimqa": qa_f1_score,
    "narrativeqa": qa_f1_score,
    "qmsum": rouge_score,
    "repobench-p": code_sim_score,
}


def score_prediction(task: str, prediction: str, answers: list[str], all_classes=None) -> float:
    metric = TASK_METRIC[task]
    pred = prediction.strip()
    best = 0.0
    for ans in answers:
        best = max(best, float(metric(pred, ans, all_classes=all_classes)))
    return best

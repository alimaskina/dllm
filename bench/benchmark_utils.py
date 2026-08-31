"""GSM8K, MATH500, GPQA Diamond loading and grading."""

from __future__ import annotations

import random
import re

from datasets import load_dataset
from lm_eval import tasks

DEFAULT_BENCHMARK_TASKS = ["gsm8k", "math500", "gpqa_diamond"]

MAX_NEW_TOKENS = {
    "gsm8k": 2048,
    "math500": 1024,
    "gpqa_diamond": 1536,
}

_LM_TASK = {
    "gsm8k": "gsm8k",
    "math500": "hendrycks_math500",
}


def extract_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def extract_gsm8k_answer(text: str) -> str | None:
    m = re.search(r"####\s*(-?[\d.,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    boxed = extract_boxed(text)
    if boxed:
        nums = re.findall(r"-?\d+(?:\.\d+)?", boxed)
        return nums[-1] if nums else boxed.strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return nums[-1] if nums else None


def normalize_math(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    return s.lower()


def extract_gpqa_choice(text: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed:
        m = re.search(r"\b([ABCD])\b", boxed.upper())
        if m:
            return m.group(1)
    for pat in [
        r"(?:final answer|answer)\s*[:is]*\s*\(?([ABCD])\)?",
        r"\(([ABCD])\)",
        r"\b([ABCD])\b\s*$",
    ]:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(1).upper()
    return None


def format_question(question: str) -> str:
    if "Answer:" in question:
        return question.replace(
            "Answer:",
            "Please reason step by step, and put your final answer within \\boxed{}.",
        )
    return (
        f"{question}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


def build_chat_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": format_question(question)}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load_benchmark_samples(task: str, *, num_examples: int, seed: int = 1234) -> list[dict]:
    if task not in DEFAULT_BENCHMARK_TASKS:
        raise ValueError(f"unknown task {task!r}, expected one of {DEFAULT_BENCHMARK_TASKS}")

    rng = random.Random(seed)

    if task == "gpqa_diamond":
        ds = load_dataset("aradhye/gpqa_diamond", split="train")
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:num_examples]
        return [
            {
                "idx": j,
                "task": task,
                "question": ds[int(i)]["problem"],
                "gold": str(ds[int(i)]["answer"]).strip().upper(),
                "grader": "gpqa",
            }
            for j, i in enumerate(idxs)
        ]

    lm_name = _LM_TASK[task]
    td = tasks.get_task_dict([lm_name])
    lm_task = td[lm_name]
    lm_task._config.num_fewshot = 0
    lm_task.set_fewshot_seed(seed=seed)
    lm_task.build_all_requests(limit=num_examples, rank=0, world_size=1)

    out = []
    for i, inst in enumerate(lm_task._instances):
        doc = inst.doc
        if task == "gsm8k":
            gold = extract_gsm8k_answer(lm_task.doc_to_target(doc))
            question = doc["question"]
            grader = "gsm8k"
        else:
            gold = doc["answer"]
            question = doc["problem"]
            grader = "math"
        out.append(
            {
                "idx": i,
                "task": task,
                "question": question,
                "gold": gold,
                "grader": grader,
            }
        )
    return out


def extract_prediction(grader: str, text: str):
    if grader == "gpqa":
        return extract_gpqa_choice(text)
    if grader == "math":
        boxed = extract_boxed(text)
        return boxed if boxed is not None else text.strip()[:120]
    return extract_gsm8k_answer(text)


def grade_sample(grader: str, gold, prediction) -> bool:
    if prediction is None:
        return False
    if grader == "gpqa":
        return str(prediction).strip().upper() == str(gold).strip().upper()
    if grader == "math":
        pg = normalize_math(extract_boxed(prediction) or prediction)
        gg = normalize_math(gold)
        return pg == gg
    return str(prediction) == str(gold)


def max_new_tokens_for_task(task: str) -> int:
    return MAX_NEW_TOKENS[task]

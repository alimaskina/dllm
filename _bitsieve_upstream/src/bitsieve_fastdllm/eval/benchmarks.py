from __future__ import annotations

import json
import os
import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class BenchmarkExample:
    example_id: str
    prompt: str
    references: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


LONG_BENCH_CONFIGS = {
    "hotpotqa": "hotpotqa",
    "narrativeqa": "narrativeqa",
    "qasper": "qasper",
    "multifieldqa_en": "multifieldqa_en",
    "qmsum": "qmsum",
    "gov_report": "gov_report",
    "multi_news": "multi_news",
    "trec": "trec",
    "samsum": "samsum",
    "passage_count": "passage_count",
    "passage_retrieval_en": "passage_retrieval_en",
    "repobench-p": "repobench-p",
    "repobench_p": "repobench-p",
    "triviaqa": "triviaqa",
    "lcc": "lcc",
    "2wikimqa": "2wikimqa",
    "musique": "musique",
}
# Deliberately not wired in: the Chinese LongBench tasks (dureader, vcsum, lsht,
# multifieldqa_zh, passage_retrieval_zh) - every prompt and generation in this project is
# English, and scoring them needs jieba segmentation this project does not depend on.


def _take(dataset: Iterable, limit: int | None):
    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        yield idx, row


def _load_hf(
    name: str,
    config: str | None,
    split: str,
    *,
    revision: str | None = None,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the project dependencies to use benchmark datasets") from exc


    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    if config is None:
        return load_dataset(name, **kwargs)
    return load_dataset(name, config, **kwargs)


def load_gsm8k(limit: int | None = None, split: str = "test") -> list[BenchmarkExample]:
    ds = _load_hf(
        "openai/gsm8k",
        "main",
        split,
        revision=os.environ.get("GSM8K_REVISION"),
    )
    out = []
    for idx, row in _take(ds, limit):
        answer = str(row["answer"])
        ref = answer.split("####")[-1].strip()
        prompt = (
            "Solve the following grade-school mathematics problem. Explain the reasoning clearly, "
            "then put only the final answer inside \\boxed{...}.\n\n"
            f"Problem: {row['question']}\n\nSolution:"
        )
        out.append(BenchmarkExample(str(idx), prompt, [ref], {"raw_answer": answer}))
    return out


def load_math500(limit: int | None = None, split: str = "test") -> list[BenchmarkExample]:
    ds = _load_hf("HuggingFaceH4/MATH-500", None, split)
    out = []
    for idx, row in _take(ds, limit):
        problem = row.get("problem") or row.get("question")
        answer = row.get("answer") or row.get("solution")
        prompt = (
            "Solve the following competition mathematics problem rigorously. Show the key steps and "
            "put the final answer inside \\boxed{...}.\n\n"
            f"Problem: {problem}\n\nSolution:"
        )
        out.append(
            BenchmarkExample(
                str(row.get("unique_id", idx)),
                prompt,
                [str(answer)],
                {k: row[k] for k in ("subject", "level") if k in row},
            )
        )
    return out


# Verbatim from THUDM/LongBench/LongBench/config/dataset2prompt.json (English tasks only -
# see the comment by LONG_BENCH_CONFIGS for why the Chinese tasks are excluded). Kept as
# raw {context}/{input} templates rather than an f-string per task, so a diff against the
# upstream JSON stays trivial.
_LONGBENCH_OFFICIAL_PROMPTS: dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question asconcisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the "
        "story asconcisely as you can, using a single phrase if possible. Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as "
        "you can, using a single phrase or sentence if possible. If the question cannot be "
        "answered based on the information in the article, write \"unanswerable\". If the "
        "question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not "
        "provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the "
        "above article as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following "
        "question based on the above text, only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the "
        "report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the "
        "query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\n"
        "Answer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of "
        "questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not "
        "output any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be "
        "duplicates. Please carefully read these paragraphs and determine how many unique "
        "paragraphs there are after removing duplicates. In other words, how many "
        "non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final "
        "count of unique paragraphs after removing duplicates. The output format should only "
        "contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine "
        "which paragraph the abstract is from.\n\n{context}\n\nThe following is an "
        "abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is "
        "from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe "
        "answer is: "
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}


def _longbench_prompt(task: str, context: str, question: str) -> str:
    template = _LONGBENCH_OFFICIAL_PROMPTS.get(task)
    if template is not None:
        return template.format(context=context, input=question)
    # Only reachable for a task added to LONG_BENCH_CONFIGS without an official template
    # above - not a normal code path, so fail loudly rather than silently scoring a task
    # against a prompt nobody chose.
    raise KeyError(
        f"no official LongBench prompt template for {task!r} - add one to "
        "_LONGBENCH_OFFICIAL_PROMPTS before wiring it into LONG_BENCH_CONFIGS"
    )


def _longbench_archive_path() -> Path:
    override = os.environ.get("LONG_BENCH_ARCHIVE_PATH")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"LONG_BENCH_ARCHIVE_PATH does not exist or is not a file: {path}"
            )
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "LongBench loading requires huggingface-hub (normally installed with transformers)"
        ) from exc

    repo_id = os.environ.get("LONG_BENCH_REPO_ID", "zai-org/LongBench")
    revision = os.environ.get("LONG_BENCH_REVISION") or None
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="data.zip",
            revision=revision,
        )
    )


def load_longbench(
    task: str,
    limit: int | None = None,
    split: str = "test",
) -> list[BenchmarkExample]:
    if split != "test":
        raise ValueError(f"LongBench only provides a test split, got split={split!r}")

    config = LONG_BENCH_CONFIGS[task]
    archive = _longbench_archive_path()
    member = f"data/{config}.jsonl"

    out: list[BenchmarkExample] = []
    with zipfile.ZipFile(archive) as zf:
        try:
            raw = zf.open(member)
        except KeyError as exc:
            available = sorted(
                name for name in zf.namelist() if name.startswith("data/") and name.endswith(".jsonl")
            )
            raise RuntimeError(
                f"LongBench archive {archive} does not contain {member}; "
                f"available data files: {available[:8]}{'...' if len(available) > 8 else ''}"
            ) from exc

        with raw:
            for idx, line in enumerate(raw):
                if limit is not None and idx >= limit:
                    break
                row = json.loads(line.decode("utf-8"))
                refs = row.get("answers", row.get("answer", []))
                if isinstance(refs, str):
                    refs = [refs]
                prompt = _longbench_prompt(
                    config, str(row.get("context", "")), str(row.get("input", ""))
                )
                out.append(
                    BenchmarkExample(
                        str(row.get("_id", idx)),
                        prompt,
                        [str(x) for x in refs],
                        {
                            "task": config,
                            "length": row.get("length"),
                            "all_classes": row.get("all_classes"),
                        },
                    )
                )
    return out


def build_niah(
    tokenizer,
    *,
    context_lengths: list[int],
    depths: list[float],
    seed: int = 1234,
) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    filler_sentence = (
        "In an old technical notebook, researchers discussed ordinary experiments, schedules, "
        "weather, books, and unrelated observations. "
    )
    filler_tokens = tokenizer.encode(filler_sentence, add_special_tokens=False)
    out: list[BenchmarkExample] = []
    for length in context_lengths:
        for depth in depths:
            key = str(rng.randint(1_000_000, 9_999_999))
            needle = (
                f" Important fact: the pass key is {key}. Remember that {key} is the pass key. "
            )
            needle_ids = tokenizer.encode(needle, add_special_tokens=False)
            target_filler = max(1, length - len(needle_ids) - 96)
            repeats = (target_filler + len(filler_tokens) - 1) // len(filler_tokens)
            body = (filler_tokens * repeats)[:target_filler]
            insert = min(len(body), max(0, int(round(depth * len(body)))))
            context_ids = body[:insert] + needle_ids + body[insert:]
            context = tokenizer.decode(context_ids, skip_special_tokens=True)
            prompt = (
                "There is one important pass key hidden in the text. Find it and answer with the "
                "seven-digit key only.\n\n"
                f"Text:\n{context}\n\nWhat is the pass key?\nAnswer:"
            )
            out.append(
                BenchmarkExample(
                    f"n{length}-d{depth:.2f}",
                    prompt,
                    [key],
                    {"context_length": length, "depth": depth},
                )
            )
    return out


# Per-benchmark output budgets, in one place so the multi-GPU runner and a
# direct `python -m bitsieve_fastdllm.eval.quality` invocation cannot disagree.
# Math needs room for a full chain of thought: a truncated CoT never emits its
# \boxed{...}, and the grader then falls back to "last number in the text",
# which scores by accident rather than by reasoning.
MATH_BENCHMARKS = frozenset({"gsm8k", "math500", "math-500"})
DEFAULT_MAX_NEW_TOKENS = 512
MAX_NEW_TOKENS = {
    "gsm8k": 2048,
    "math500": 2048,
    "math-500": 2048,
    "niah": 64,
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}


def max_new_tokens_for(benchmark: str | None) -> int:
    """Output budget for a benchmark; 128 for the synthetic performance points."""
    if not benchmark:
        return 128
    return MAX_NEW_TOKENS.get(benchmark.lower(), DEFAULT_MAX_NEW_TOKENS)


def load_benchmark(
    name: str,
    *,
    tokenizer=None,
    limit: int | None = None,
    split: str = "test",
    niah_contexts: list[int] | None = None,
    niah_depths: list[float] | None = None,
    seed: int = 1234,
) -> list[BenchmarkExample]:
    key = name.lower()
    if key == "gsm8k":
        return load_gsm8k(limit, split)
    if key in {"math500", "math-500"}:
        return load_math500(limit, split)
    if key in LONG_BENCH_CONFIGS:
        return load_longbench(key, limit, split)
    if key == "niah":
        if tokenizer is None:
            raise ValueError("tokenizer is required for NIAH")
        examples = build_niah(
            tokenizer,
            context_lengths=niah_contexts or [8192, 16384, 28672],
            depths=niah_depths or [0.0, 0.25, 0.5, 0.75, 1.0],
            seed=seed,
        )
        return examples[:limit] if limit is not None else examples
    raise ValueError(f"unknown benchmark: {name}")

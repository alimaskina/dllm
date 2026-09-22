"""Which benchmarks actually make these models reason long, and where is the metric still loose?

The eviction study needs a benchmark with two properties at once: the model must
generate *long* (otherwise the cache never grows past the capacity floor and the
policy never engages), and the metric must be *unsaturated* (otherwise nothing
can be lost or gained). Neither property is a property of the benchmark alone --
both depend on the decoder, so this probe measures them per model.

The diagnostic is a budget sweep: run the same examples under two generation
ceilings. Three outcomes, each meaning something different:

  * score rises with the ceiling        -> the extra tokens buy reasoning
  * score flat, all runs hit the ceiling -> the model loops; length is not thought
  * score flat, runs stop on their own   -> the ceiling was never the constraint

Per cell we record the score, the median total context, how many runs hit the
ceiling, and how many produced a \\boxed{} at all -- the last one guards the
score itself, because a grader reading a format the model does not emit measures
the grader.

Usage:
  PYTHONPATH=src python scripts/probe_benchmarks.py run \\
      --model fastdllm --device cuda:0 --limit 10 --max-new-tokens 2048 \\
      --out results/benchprobe
  PYTHONPATH=src python scripts/probe_benchmarks.py report --out results/benchprobe

The two models need different environments (Fast-dLLM pins transformers 4.53,
Dream needs 5.x), so run them separately and report over the shared directory.

OlympiadBench is deliberately absent: its gold answers are symbolic LaTeX
("$\\frac{1}{2n+2}$", "$\\binom{2n}{n}$") and a string-normalising grader scores
a flat 0.00 on it regardless of what the model writes. Adding it back needs a
sympy equivalence check, not another prompt.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import torch

LETTERS = "ABCDEFGHIJ"


# -- grading ---------------------------------------------------------------
def boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...}, brace-matched. None if there is none."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    i += 7
    depth, out = 1, ""
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if not depth:
                break
        out += c
        i += 1
    return out.strip()


def norm(s: object) -> str:
    return re.sub(r"[\s\$\\,]|\\left|\\right|\\!", "", str(s)).lower().strip(".")


def grade_math(pred: str, gold: object) -> tuple[float, str, str]:
    """Returns (score, extracted, which rule fired)."""
    b, rule = boxed(pred), "boxed"
    if b is None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
        b, rule = (nums[-1] if nums else ""), "last-number"
    nb, ng = norm(b), norm(gold)
    # bool() is load-bearing: `nb.lstrip("0") and ...` evaluates to a *string*,
    # and float("204") would be 204.0 rather than a 0/1 score.
    hit = bool(nb == ng or (ng and nb.lstrip("0") == ng.lstrip("0")))
    return float(hit), b, rule


def grade_mc(pred: str, gold: object, n: int = 10) -> tuple[float, str, str]:
    """Multiple choice. Returns (score, extracted letter, which rule fired).

    The fallback order matters. An early version searched for the *first*
    ``answer|option`` in the text, which reliably caught the model enumerating
    the choices ("Option A. ...") rather than its verdict, and threw away
    correct answers. Both fallbacks now read from the end of the generation,
    where the conclusion is.
    """
    letters = LETTERS[:n]
    b = boxed(pred) or ""
    m = re.search(rf"\b([{letters}])\b", b.upper())
    if m is not None:
        return float(m.group(1) == str(gold).upper()), m.group(1), "boxed"

    upper = pred.upper()
    cand = re.findall(rf"ANSWER\s*(?:IS|:)\s*\**\s*\(?([{letters}])\b", upper)
    rule = "answer-is"
    if not cand:
        cand = re.findall(rf"(?<![A-Z])([{letters}])(?![A-Z])", upper)
        rule = "last-letter"
    if not cand:
        return 0.0, "", "none"
    return float(cand[-1] == str(gold).upper()), cand[-1], rule


# -- prompts and benchmarks ------------------------------------------------
MATH_PROMPT = (
    "Solve the following competition mathematics problem rigorously. Show the key "
    "steps and put the final answer inside \\boxed{{...}}.\n\nProblem: {q}\n\nSolution:"
)
MC_PROMPT = (
    "Answer the following multiple-choice question. Reason step by step, then put "
    "the letter of the correct option inside \\boxed{{...}}.\n\nQuestion: {q}\n\n"
    "Options:\n{o}\n\nAnswer:"
)


def _options(row) -> str:
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))


BENCHMARKS = {
    "aime2024": dict(
        source=("HuggingFaceH4/aime_2024", None, "train"),
        prompt=lambda r: MATH_PROMPT.format(q=r["problem"]),
        grade=lambda r, t: grade_math(t, r["answer"]),
    ),
    "aime2025": dict(
        source=("opencompass/AIME2025", "AIME2025-I", "test"),
        prompt=lambda r: MATH_PROMPT.format(q=r["question"]),
        grade=lambda r, t: grade_math(t, r["answer"]),
    ),
    "gpqa": dict(
        source=("aradhye/gpqa_diamond", None, "train"),
        prompt=lambda r: MATH_PROMPT.format(q=r["problem"]),
        grade=lambda r, t: grade_mc(t, r["answer"], 4),
    ),
    "mmlupro": dict(
        source=("TIGER-Lab/MMLU-Pro", None, "test"),
        prompt=lambda r: MC_PROMPT.format(q=r["question"], o=_options(r)),
        grade=lambda r, t: grade_mc(t, r["answer"], 10),
    ),
}


def load_rows(key: str, limit: int):
    from datasets import load_dataset

    repo, config, split = BENCHMARKS[key]["source"]
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    return list(ds)[:limit]


# -- decoders --------------------------------------------------------------
class FastDLLMRunner:
    """Dense bf16 Fast-dLLM-v2 through the BitSieve generator (no sparsity, no quant)."""

    name = "fastdllm"
    model_id = "Efficient-Large-Model/Fast_dLLM_v2_7B"

    def __init__(self, device: str, max_new_tokens: int, block_size: int, threshold: float):
        from bitsieve_fastdllm.config import ExperimentConfig
        from bitsieve_fastdllm.eval.common import load_fast_dllm
        from bitsieve_fastdllm.runtime.generator import BitSieveGenerator

        self.device = device
        root = Path(__file__).resolve().parent.parent
        self.model, self.tokenizer = load_fast_dllm(
            self.model_id, dtype=torch.bfloat16, device=device
        )
        raw = ExperimentConfig.load(str(root / "configs" / "official_dense_bf16.yaml")).to_dict()
        raw["generation"].update(
            max_new_tokens=max_new_tokens, threshold=threshold, block_size=block_size
        )
        self.generator = BitSieveGenerator(self.model, self.tokenizer, ExperimentConfig.from_dict(raw))

    def generate(self, input_ids):
        result = self.generator.generate(input_ids)
        return result.texts[0], int(result.metrics["generated_tokens_per_request"])

    def close(self):
        self.generator.patch.unpatch()


class DreamRunner:
    """DreamReasoner-8B through its own block-diffusion entry point."""

    name = "dream"
    model_id = "Dream-org/DreamReasoner-8B"

    def __init__(self, device: str, max_new_tokens: int, block_size: int, threshold: float):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device, self.max_new_tokens = device, max_new_tokens
        self.block_size, self.threshold = block_size, threshold
        torch.cuda.set_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
            .to(device)
            .eval()
        )

    def generate(self, input_ids):
        out = self.model.block_diffusion_generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            block_length=self.block_size,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            remasking_strategy="low_confidence_dynamic",
            confidence_threshold=self.threshold,
            eb_threshold=0.35,
            use_kv_cache=True,
        )
        generated = int(out.shape[1]) - int(input_ids.shape[1])
        text = self.tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
        return text, generated

    def close(self):
        pass


RUNNERS = {"fastdllm": FastDLLMRunner, "dream": DreamRunner}


# -- run -------------------------------------------------------------------
def cell_path(out: Path, model: str, bench: str, budget: int) -> Path:
    return out / f"{model}__{bench}__g{budget}.jsonl"


def run(args) -> None:
    from bitsieve_fastdllm.eval.common import encode_prompt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = RUNNERS[args.model](
        args.device, args.max_new_tokens, args.block_size, args.threshold
    )
    (out / f"environment__{args.model}.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "executable": sys.executable,
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "model": runner.model_id,
                "settings": {k: v for k, v in vars(args).items() if k != "func"},
            },
            indent=2,
        )
    )
    try:
        for bench in args.benchmarks.split(","):
            bench = bench.strip()
            if bench not in BENCHMARKS:
                raise SystemExit(f"unknown benchmark {bench!r}; known: {sorted(BENCHMARKS)}")
            path = cell_path(out, args.model, bench, args.max_new_tokens)
            if path.exists() and sum(1 for _ in path.open()) >= args.limit and not args.force:
                print(f"  [есть] {path.name}", flush=True)
                continue
            rows = load_rows(bench, args.limit)
            spec = BENCHMARKS[bench]
            with path.open("w") as fh:
                for i, row in enumerate(rows):
                    ids = encode_prompt(
                        runner.tokenizer,
                        spec["prompt"](row),
                        max_input_tokens=args.max_input_tokens,
                        use_chat_template=True,
                        device=args.device,
                    )
                    text, generated = runner.generate(ids)
                    score, extracted, rule = spec["grade"](row, text)
                    fh.write(
                        json.dumps(
                            {
                                "index": i,
                                "score": score,
                                "extracted": extracted,
                                "rule": rule,
                                "prompt_tokens": int(ids.shape[1]),
                                "generated_tokens": generated,
                                "context_tokens": int(ids.shape[1]) + generated,
                                # -32 = one block of slack: the decoder stops on a
                                # block boundary, so it never lands exactly on the cap.
                                "capped": bool(generated >= args.max_new_tokens - 32),
                                "has_boxed": "\\boxed" in text,
                                "tail": text[-400:],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    fh.flush()
            print(f"  [готово] {path.name}", flush=True)
    finally:
        runner.close()


# -- report ----------------------------------------------------------------
def report(args) -> None:
    out = Path(args.out)
    cells: dict[tuple[str, str], dict[int, dict]] = {}
    for path in sorted(out.glob("*__*__g*.jsonl")):
        model, bench, budget = path.stem.split("__")
        rows = [json.loads(line) for line in path.open() if line.strip()]
        if not rows:
            continue
        cells.setdefault((model, bench), {})[int(budget[1:])] = {
            "n": len(rows),
            "score": statistics.mean(r["score"] for r in rows),
            "context": statistics.median(r["context_tokens"] for r in rows),
            "capped": sum(r["capped"] for r in rows),
            "boxed": sum(r["has_boxed"] for r in rows),
            "fallback": sum(r["rule"] != "boxed" for r in rows),
        }

    print("| модель | бенчмарк | потолок | n | score | медиана контекста | упёрлись | boxed | грейд не из boxed |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (model, bench), budgets in sorted(cells.items()):
        for budget in sorted(budgets):
            c = budgets[budget]
            print(
                f"| {model} | {bench} | {budget} | {c['n']} | {c['score']:.2f} | "
                f"{c['context']:.0f} | {c['capped']}/{c['n']} | {c['boxed']}/{c['n']} | "
                f"{c['fallback']}/{c['n']} |"
            )

    print()
    print("Пригодность для исследования вытеснения (нужно: длинно И метрика не насыщена):")
    for (model, bench), budgets in sorted(cells.items()):
        if len(budgets) < 2:
            continue
        lo, hi = min(budgets), max(budgets)
        delta = budgets[hi]["score"] - budgets[lo]["score"]
        capped_lo = budgets[lo]["capped"] / budgets[lo]["n"]
        if delta > 0.05:
            verdict = f"годится: +{delta:.2f} от бюджета {lo}->{hi}"
        elif capped_lo < 0.5:
            verdict = "нет: модель заканчивает сама, потолок не ограничивал"
        else:
            verdict = f"нет: упирается в потолок, но бюджет ничего не даёт ({delta:+.2f})"
        print(f"  {model:9s} {bench:9s} {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--model", choices=sorted(RUNNERS), required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--benchmarks", default="aime2024,aime2025,gpqa,mmlupro")
    r.add_argument("--limit", type=int, default=10)
    r.add_argument("--max-new-tokens", type=int, default=2048)
    r.add_argument("--max-input-tokens", type=int, default=4096)
    r.add_argument("--block-size", type=int, default=32)
    r.add_argument("--threshold", type=float, default=0.95)
    r.add_argument("--out", default="results/benchprobe")
    r.add_argument("--force", action="store_true", help="redo cells that already have enough rows")
    r.set_defaults(func=run)

    p = sub.add_parser("report")
    p.add_argument("--out", default="results/benchprobe")
    p.set_defaults(func=report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

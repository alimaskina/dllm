from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable


_BOX_RE = re.compile(r"\\boxed\s*\{")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def extract_last_boxed(text: str) -> str | None:
    starts = [m.start() for m in _BOX_RE.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    brace = text.find("{", start)
    depth = 0
    for pos in range(brace, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : pos]
    return None


def numeric_normalize(value: str) -> str:
    x = value.strip()
    x = x.replace("\\$", "").replace("$", "")
    x = x.replace("{,}", "").replace(",", "")
    x = x.replace("\\%", "").replace("%", "")
    x = x.strip(" {}[]()\n\t.")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", x):
        sign = ""
        if x.startswith(("+", "-")):
            sign, x = x[0], x[1:]
        if "." in x:
            x = x.rstrip("0").rstrip(".")
        x = x.lstrip("0") or "0"
        if x.startswith("."):
            x = "0" + x
        return sign + x
    return x.lower().replace(" ", "")


def normalize_math(value: str) -> str:
    x = value.strip().lower()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\!": "",
        "\\,": "",
        "\\ ": "",
        " ": "",
        "\n": "",
        "\t": "",
    }
    for old, new in replacements.items():
        x = x.replace(old, new)
    x = x.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    x = x.replace("{,}", "").replace("\\$", "").replace("$", "")
    return x.strip(".")


def extract_math_answer(text: str) -> str:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed
    cues = ["final answer", "answer is", "therefore", "thus"]
    lower = text.lower()
    tail = text
    for cue in cues:
        idx = lower.rfind(cue)
        if idx >= 0:
            tail = text[idx + len(cue) :]
            break
    nums = _NUMBER_RE.findall(tail)
    if nums:
        return nums[-1]
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else text.strip().splitlines()[-1] if text.strip() else ""


def math_exact_match(prediction: str, references: Iterable[str]) -> float:
    pred = extract_math_answer(prediction)
    p_math = normalize_math(pred)
    p_num = numeric_normalize(pred)
    for ref in references:
        candidates = [ref]
        boxed = extract_last_boxed(ref)
        if boxed is not None:
            candidates.append(boxed)
        for candidate in candidates:
            if p_math == normalize_math(candidate) or p_num == numeric_normalize(candidate):
                return 1.0
    return 0.0


def normalize_qa(text: str) -> str:
    x = text.lower().replace("_", " ")
    x = _PUNCT.sub(" ", x)
    x = _ARTICLES.sub(" ", x)
    return " ".join(x.split())


def qa_f1(prediction: str, reference: str) -> float:
    p = normalize_qa(prediction).split()
    r = normalize_qa(reference).split()
    if not p or not r:
        return float(p == r)
    counts: dict[str, int] = {}
    for token in r:
        counts[token] = counts.get(token, 0) + 1
    common = 0
    for token in p:
        if counts.get(token, 0):
            common += 1
            counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2 * precision * recall / (precision + recall)


def best_qa_f1(prediction: str, references: Iterable[str]) -> float:
    return max((qa_f1(prediction, x) for x in references), default=0.0)


def rouge_l(prediction: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(reference, prediction)["rougeL"].fmeasure)
    except Exception:

        p = prediction.split()
        r = reference.split()
        row = [0] * (len(r) + 1)
        for a in p:
            previous = 0
            for j, b in enumerate(r, 1):
                old = row[j]
                row[j] = previous + 1 if a == b else max(row[j], row[j - 1])
                previous = old
        lcs = row[-1]
        return 0.0 if not p or not r else 2 * lcs / (len(p) + len(r))


def best_rouge_l(prediction: str, references: Iterable[str]) -> float:
    return max((rouge_l(prediction, x) for x in references), default=0.0)


def edit_similarity(prediction: str, reference: str) -> float:
    return SequenceMatcher(None, prediction.strip(), reference.strip()).ratio()


def best_edit_similarity(prediction: str, references: Iterable[str]) -> float:
    return max((edit_similarity(prediction, x) for x in references), default=0.0)


def score_prediction(benchmark: str, prediction: str, references: list[str]) -> float:
    name = benchmark.lower()
    if name in {"gsm8k", "math500", "math-500"}:
        return math_exact_match(prediction, references)
    if name in {"qmsum", "narrativeqa"}:
        return best_rouge_l(prediction, references)
    if name in {"repobench-p", "repobench_p", "lcc"}:
        return best_edit_similarity(prediction, references)
    if name == "niah":
        return float(any(normalize_qa(ref) in normalize_qa(prediction) for ref in references))
    return best_qa_f1(prediction, references)

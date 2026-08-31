#!/usr/bin/env python3
"""Post-hoc re-grader for gsm8k/math500 results.jsonl.

Fixes issues found in the original grader:
- extract_gsm8k_answer breaks on \\$70{,}000 (LaTeX escaped comma inside braces),
  returning "000" instead of "70000".
- gsm8k grading is exact string equality, so gold=26 vs pred=26.00 counts wrong.

Strategy:
- Re-extract the last \\boxed{...} block (matching-braces aware) from generated_answer.
- Normalize both gold and prediction: strip $, {,}, commas, trailing .0+.
- For math grader, keep the existing normalize_math logic in addition.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


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


def normalize_number_string(s: str) -> str:
    """Strip LaTeX $, {,}, commas, whitespace, trailing zeros after decimal."""
    s = str(s)
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("{,}", "").replace(",", "")
    s = s.replace("\\text", "").replace("\\!", "")
    s = re.sub(r"\\[a-zA-Z]+\s*\{[^}]*\}", "", s)  # strip \foo{...}
    s = re.sub(r"\s+", "", s)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return s
    num = m.group(0)
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num


def normalize_math(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    return s.lower()


def regrade_gsm8k(gold: str, gen: str) -> tuple[bool, str]:
    boxed = extract_boxed(gen)
    if boxed is not None:
        pred = normalize_number_string(boxed)
    else:
        nums = re.findall(r"-?\d+(?:\.\d+)?", gen)
        pred = nums[-1] if nums else ""
    gold_n = normalize_number_string(gold)
    return (pred == gold_n and pred != ""), pred


def regrade_math(gold: str, gen: str) -> tuple[bool, str]:
    boxed = extract_boxed(gen)
    if boxed is None:
        return False, ""
    pred_raw = boxed
    # Try both: normalize_math AND number-string normalization (for numeric answers)
    g_math = normalize_math(gold)
    p_math = normalize_math(pred_raw)
    if p_math == g_math:
        return True, pred_raw
    g_num = normalize_number_string(gold)
    p_num = normalize_number_string(pred_raw)
    if g_num and p_num and g_num == p_num:
        return True, pred_raw
    return False, pred_raw


def regrade_file(path: Path) -> dict:
    rs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    orig_correct = 0
    new_correct = 0
    flipped_to_correct = 0
    flipped_to_wrong = 0
    diffs = []
    for r in rs:
        grader = r.get("grader") or ("math" if r["task"] == "math500" else "gsm8k")
        gold = r["gold"]
        gen = r["generated_answer"]
        if grader == "math" or r["task"] == "math500":
            ok, pred = regrade_math(gold, gen)
        else:
            ok, pred = regrade_gsm8k(gold, gen)
        if r["correct"]:
            orig_correct += 1
        if ok:
            new_correct += 1
        if ok and not r["correct"]:
            flipped_to_correct += 1
            diffs.append({"ex": r["example_id"], "gold": gold, "old_pred": r["prediction"], "new_pred": pred, "flip": "wrong->right"})
        elif r["correct"] and not ok:
            flipped_to_wrong += 1
            diffs.append({"ex": r["example_id"], "gold": gold, "old_pred": r["prediction"], "new_pred": pred, "flip": "right->wrong"})
    n = len(rs)
    return {
        "file": str(path),
        "n": n,
        "orig_acc_pct": round(100 * orig_correct / n, 2),
        "new_acc_pct": round(100 * new_correct / n, 2),
        "delta_correct": new_correct - orig_correct,
        "flipped_to_correct": flipped_to_correct,
        "flipped_to_wrong": flipped_to_wrong,
        "diffs": diffs,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--show-diffs", action="store_true")
    args = p.parse_args()

    all_summaries = []
    for f in args.files:
        s = regrade_file(f)
        all_summaries.append(s)
        print(f"\n{f}")
        print(f"  n={s['n']}  orig_acc={s['orig_acc_pct']}%  new_acc={s['new_acc_pct']}%  Δcorrect={s['delta_correct']:+d}")
        print(f"  flipped_to_correct={s['flipped_to_correct']}  flipped_to_wrong={s['flipped_to_wrong']}")
        if args.show_diffs and s["diffs"]:
            for d in s["diffs"][:20]:
                print(f"    ex{d['ex']}  gold={d['gold']!r}  old={d['old_pred']!r}  new={d['new_pred']!r}  [{d['flip']}]")


if __name__ == "__main__":
    main()

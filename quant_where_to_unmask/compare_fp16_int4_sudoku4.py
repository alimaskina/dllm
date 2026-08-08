#!/usr/bin/env python3
import json
from pathlib import Path

from lm_eval import tasks
from lm_eval.tasks import TaskManager

from tasks.sudoku4.utils import clean_generation, is_valid_sudoku_dream


def load_ckpt(d: str) -> list[dict]:
    rows = []
    for p in sorted(Path(d).glob("rank*.jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("doc_id", r.get("idx", 0)))
    return rows


def status(row: dict, doc: dict) -> tuple[bool, bool, str]:
    gen = clean_generation(row["response"])
    gold = doc["target"]
    exact = gen.replace("\n", "") == gold.replace("\n", "")
    valid = is_valid_sudoku_dream(doc["puzzle_input"], gen)
    return exact, valid, gen


def main():
    tm = TaskManager(include_path="tasks")
    task = tasks.get_task_dict(["sudoku4_8shot"], task_manager=tm)["sudoku4_8shot"]
    docs = list(task.test_docs())

    fp16 = load_ckpt("checkpoints/results_sudoku_fp16_4x4_8shot_n256_2gpu")
    int4 = load_ckpt("checkpoints/results_sudoku_int4_4x4_8shot_n256_2gpu")
    by_fp = {r["doc_id"]: r for r in fp16}
    by_i4 = {r["doc_id"]: r for r in int4}

    cats = {
        "both_ok": [],
        "fp16_only": [],
        "int4_only": [],
        "both_fail_valid": [],
        "both_fail_exact_valid": [],
        "fp16_valid_int4_invalid": [],
        "int4_valid_fp16_invalid": [],
    }

    for i, doc in enumerate(docs):
        if i not in by_fp or i not in by_i4:
            continue
        ex_f, val_f, gen_f = status(by_fp[i], doc)
        ex_i, val_i, gen_i = status(by_i4[i], doc)
        item = {
            "doc_id": i,
            "input": doc["puzzle_input"],
            "gold": doc["target"],
            "fp16": gen_f,
            "int4": gen_i,
        }
        if ex_f and ex_i:
            cats["both_ok"].append(item)
        elif ex_f:
            cats["fp16_only"].append(item)
        elif ex_i:
            cats["int4_only"].append(item)
        elif val_f and val_i:
            cats["both_fail_exact_valid"].append(item)
        elif val_f and not val_i:
            cats["fp16_valid_int4_invalid"].append(item)
        elif val_i and not val_f:
            cats["int4_valid_fp16_invalid"].append(item)
        else:
            cats["both_fail_valid"].append(item)

    print("=== SUMMARY ===")
    print(f"FP16 exact: {sum(1 for i, d in enumerate(docs) if i in by_fp and status(by_fp[i], d)[0])}/100")
    print(f"INT4 exact: {sum(1 for i, d in enumerate(docs) if i in by_i4 and status(by_i4[i], d)[0])}/100")
    for k, v in cats.items():
        print(f"{k}: {len(v)}")

    def show(title, items, n=None):
        print(f"\n=== {title} ===")
        for x in items[: n or len(items)]:
            print(f"#{x['doc_id']}")
            print("INPUT:", x["input"])
            print("GOLD :", x["gold"])
            print("FP16 :", x["fp16"])
            print("INT4 :", x["int4"])
            print()

    show("BOTH CORRECT (2 examples)", cats["both_ok"], 2)
    show("FP16 ONLY", cats["fp16_only"])
    show("INT4 ONLY", cats["int4_only"])
    show("BOTH FAIL invalid (3 examples)", cats["both_fail_valid"], 3)
    show("BOTH FAIL valid but wrong (2 examples)", cats["both_fail_exact_valid"], 2)
    show("FP16 valid / INT4 invalid (2 examples)", cats["fp16_valid_int4_invalid"], 2)
    show("INT4 valid / FP16 invalid (2 examples)", cats["int4_valid_fp16_invalid"], 2)

    both_wrong = [
        i
        for i in range(len(docs))
        if i in by_fp
        and i in by_i4
        and not status(by_fp[i], docs[i])[0]
        and not status(by_i4[i], docs[i])[0]
    ]
    same = sum(
        1
        for i in both_wrong
        if clean_generation(by_fp[i]["response"]).replace("\n", "")
        == clean_generation(by_i4[i]["response"]).replace("\n", "")
    )
    print(f"=== BOTH WRONG: {len(both_wrong)} cases, identical wrong answer: {same} ({same/len(both_wrong):.0%}) ===")


if __name__ == "__main__":
    main()

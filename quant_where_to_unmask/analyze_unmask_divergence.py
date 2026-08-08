#!/usr/bin/env python3
"""Compare FP16 vs INT4 diffusion traces: where-to-unmask vs what-to-unmask."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from lm_eval import tasks
from lm_eval.tasks import TaskManager

from tasks.sudoku4.utils import clean_generation


def load_ckpt_rows(ckpt_dir: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for path in sorted(ckpt_dir.glob("rank*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                rows[row["doc_id"]] = row
    return rows


def load_trace(ckpt_dir: Path, row: dict) -> dict | None:
    rel = row.get("trace_path")
    if not rel:
        return None
    path = ckpt_dir / rel
    if not path.exists():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def unmasked_map(step: dict) -> dict[int, int]:
    return {u["pos_comp"]: u["token_id"] for u in step.get("unmasked", [])}


def unmasked_tokens(step: dict) -> dict[int, str]:
    return {u["pos_comp"]: u.get("token", str(u["token_id"])) for u in step.get("unmasked", [])}


def topk_positions(step: dict, k: int = 3) -> list[int]:
    comp = step.get("completion", {})
    conf = comp.get("confidence") or []
    masked = comp.get("masked") or []
    ranked = []
    for pos, is_m, c in zip(range(len(masked)), masked, conf):
        if is_m and c is not None:
            ranked.append((pos, c))
    ranked.sort(key=lambda x: -x[1])
    return [p for p, _ in ranked[:k]]


def exact_match(response: str, gold: str) -> bool:
    return clean_generation(response).replace("\n", "") == gold.replace("\n", "")


def analyze_pair(fp16_trace: dict, int4_trace: dict) -> dict:
    steps_f = fp16_trace["steps_trace"]
    steps_i = int4_trace["steps_trace"]
    n_steps = min(len(steps_f), len(steps_i))

    per_step = []
    first_where_diff = None
    first_what_diff = None
    first_any_diff = None

    where_match_steps = 0
    what_match_steps = 0
    both_match_steps = 0

    for s in range(n_steps):
        sf, si = steps_f[s], steps_i[s]
        wf = unmasked_map(sf)
        wi = unmasked_map(si)
        pos_f = next(iter(wf), None)
        pos_i = next(iter(wi), None)
        tok_f = wf.get(pos_f) if pos_f is not None else None
        tok_i = wi.get(pos_i) if pos_i is not None else None

        where_same = pos_f == pos_i
        what_same = tok_f == tok_i if where_same else False

        if where_same:
            where_match_steps += 1
        if what_same:
            what_match_steps += 1
        if where_same and what_same:
            both_match_steps += 1

        top1_f = sf.get("gap", {}).get("top1_pos_comp")
        top1_i = si.get("gap", {}).get("top1_pos_comp")
        top1_same = top1_f == top1_i

        # Would INT4's chosen pos be in FP16's top-3 masked ranking?
        top3_f = topk_positions(sf, 3)
        top3_i = topk_positions(si, 3)
        cross_rank_f = pos_i in top3_f if pos_i is not None else False
        cross_rank_i = pos_f in top3_i if pos_f is not None else False

        if not where_same and first_where_diff is None:
            first_where_diff = s
        if where_same and not what_same and first_what_diff is None:
            first_what_diff = s
        if not (where_same and what_same) and first_any_diff is None:
            first_any_diff = s

        per_step.append(
            {
                "step": s,
                "where_same": where_same,
                "what_same": what_same,
                "pos_fp16": pos_f,
                "pos_int4": pos_i,
                "tok_fp16": tok_f,
                "tok_int4": tok_i,
                "top1_same": top1_same,
                "margin_fp16": sf.get("gap", {}).get("boundary_margin"),
                "margin_int4": si.get("gap", {}).get("boundary_margin"),
                "cross_rank": cross_rank_f and cross_rank_i,
            }
        )

    return {
        "n_steps": n_steps,
        "where_match_steps": where_match_steps,
        "what_match_steps": what_match_steps,
        "both_match_steps": both_match_steps,
        "first_where_diff": first_where_diff,
        "first_what_diff": first_what_diff,
        "first_any_diff": first_any_diff,
        "per_step": per_step,
    }


def classify_outcome(fp16_resp: str, int4_resp: str, gold: str) -> str:
    ex_f = exact_match(fp16_resp, gold)
    ex_i = exact_match(int4_resp, gold)
    if ex_f and ex_i:
        return "both_ok"
    if ex_f:
        return "fp16_only"
    if ex_i:
        return "int4_only"
    return "both_wrong"


def first_diff_kind(per_step: list[dict], step_idx: int | None) -> str | None:
    if step_idx is None:
        return "never"
    st = per_step[step_idx]
    if st["where_same"] and not st["what_same"]:
        return "what_only"
    if not st["where_same"]:
        return "where"
    return "other"


def summarize(samples: list[dict], label: str) -> None:
    n = len(samples)
    if n == 0:
        print(f"\n{label}: no samples")
        return

    def mean(xs):
        return statistics.mean(xs) if xs else 0.0

    print(f"\n{'=' * 70}")
    print(label)
    print(f"samples: {n}")

    where_frac = [s["where_match_steps"] / s["n_steps"] for s in samples]
    what_frac = [s["what_match_steps"] / s["n_steps"] for s in samples]
    both_frac = [s["both_match_steps"] / s["n_steps"] for s in samples]
    print("\n--- per-step agreement (mean over samples) ---")
    print(f"  where match: {mean(where_frac):.1%}")
    print(f"  what match (given same where): {mean(what_frac):.1%}")
    print(f"  both where+what: {mean(both_frac):.1%}")
    cond_what = []
    for s in samples:
        w = sum(1 for st in s["per_step"] if st["where_same"])
        t = sum(1 for st in s["per_step"] if st["where_same"] and st["what_same"])
        if w:
            cond_what.append(t / w)
    if cond_what:
        print(f"  P(what same | where same): {mean(cond_what):.1%}")

    first_any = [s["first_any_diff"] for s in samples if s["first_any_diff"] is not None]
    print(f"\n--- first divergence step ---")
    print(f"  never diverged: {sum(1 for s in samples if s['first_any_diff'] is None)}/{n}")
    if first_any:
        print(f"  mean step: {mean(first_any):.1f}  median: {statistics.median(first_any):.0f}")

    kinds = Counter(first_diff_kind(s["per_step"], s["first_any_diff"]) for s in samples)
    print(f"  first diff type: {dict(kinds)}")

    # At first divergence: tight margins?
    margins_f = []
    margins_i = []
    for s in samples:
        idx = s["first_any_diff"]
        if idx is None:
            continue
        st = s["per_step"][idx]
        if st["margin_fp16"] is not None:
            margins_f.append(st["margin_fp16"])
        if st["margin_int4"] is not None:
            margins_i.append(st["margin_int4"])
    if margins_f:
        print(f"  boundary_margin@first_diff FP16: mean={mean(margins_f):.4f}")
        print(f"  boundary_margin@first_diff INT4: mean={mean(margins_i):.4f}")
        print(f"  tight (<0.05) @first_diff: FP16 {sum(m<0.05 for m in margins_f)/len(margins_f):.1%}, INT4 {sum(m<0.05 for m in margins_i)/len(margins_i):.1%}")

    top1_agree = mean(
        sum(1 for st in s["per_step"] if st["top1_same"]) / s["n_steps"] for s in samples
    )
    print(f"\n--- ranking before unmask (top1 pos_comp) ---")
    print(f"  top1 agreement per step: {top1_agree:.1%}")

    cross = mean(
        sum(1 for st in s["per_step"] if st["cross_rank"]) / s["n_steps"] for s in samples
    )
    print(f"  other's chosen pos in own top-3: {cross:.1%} of steps")


def summarize_by_outcome(all_samples: list[dict]) -> None:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for s in all_samples:
        by_cat[s["outcome"]].append(s)
    for cat in ("both_ok", "fp16_only", "int4_only", "both_wrong"):
        summarize(by_cat[cat], f"outcome={cat} (n={len(by_cat[cat])})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sudoku4_2shot_large")
    parser.add_argument("--fp16", default="checkpoints/results_sudoku_fp16_large_4x4_2shot_2gpu")
    parser.add_argument("--int4", default="checkpoints/results_sudoku_int4_large_4x4_2shot_2gpu")
    parser.add_argument("--examples", type=int, default=3, help="show N fp16_only divergence examples")
    args = parser.parse_args()

    fp16_dir = Path(args.fp16)
    int4_dir = Path(args.int4)

    tm = TaskManager(include_path="tasks")
    task = tasks.get_task_dict([args.task], task_manager=tm)[args.task]
    docs = {i: doc for i, doc in enumerate(task.test_docs())}

    fp16_rows = load_ckpt_rows(fp16_dir)
    int4_rows = load_ckpt_rows(int4_dir)

    all_samples = []
    missing = 0
    for doc_id in sorted(docs):
        if doc_id not in fp16_rows or doc_id not in int4_rows:
            missing += 1
            continue
        tr_f = load_trace(fp16_dir, fp16_rows[doc_id])
        tr_i = load_trace(int4_dir, int4_rows[doc_id])
        if tr_f is None or tr_i is None:
            missing += 1
            continue
        doc = docs[doc_id]
        outcome = classify_outcome(
            fp16_rows[doc_id]["response"],
            int4_rows[doc_id]["response"],
            doc["target"],
        )
        pair = analyze_pair(tr_f, tr_i)
        pair["doc_id"] = doc_id
        pair["outcome"] = outcome
        all_samples.append(pair)

    print(f"task={args.task}  analyzed={len(all_samples)}  missing={missing}")
    summarize(all_samples, "ALL")
    summarize_by_outcome(all_samples)

    # fp16_only: is divergence where-driven?
    fp16_only = [s for s in all_samples if s["outcome"] == "fp16_only"]
    if fp16_only:
        where_first = sum(1 for s in fp16_only if first_diff_kind(s["per_step"], s["first_any_diff"]) == "where")
        what_first = sum(1 for s in fp16_only if first_diff_kind(s["per_step"], s["first_any_diff"]) == "what_only")
        print(f"\n--- fp16_only ({len(fp16_only)}): first divergence ---")
        print(f"  where-driven: {where_first} ({where_first/len(fp16_only):.1%})")
        print(f"  what-driven (same pos): {what_first} ({what_first/len(fp16_only):.1%})")

    if args.examples and fp16_only:
        print(f"\n--- fp16_only examples (first {args.examples}) ---")
        for s in fp16_only[: args.examples]:
            idx = s["first_any_diff"]
            st = s["per_step"][idx] if idx is not None else None
            doc = docs[s["doc_id"]]
            print(f"\n#{s['doc_id']} first_diff_step={idx}")
            print("INPUT:", doc["puzzle_input"].strip())
            print("GOLD :", doc["target"].strip())
            if st:
                print(
                    f"  step {idx}: FP16 unmask pos={st['pos_fp16']} tok={st['tok_fp16']} | "
                    f"INT4 pos={st['pos_int4']} tok={st['tok_int4']} | "
                    f"margin f/i={st['margin_fp16']}/{st['margin_int4']}"
                )


if __name__ == "__main__":
    main()

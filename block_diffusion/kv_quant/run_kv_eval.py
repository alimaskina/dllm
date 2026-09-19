#!/usr/bin/env python3
"""Multi-dataset KV eval: baseline + uniform KIVI INT4/INT2."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
import types
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from lm_eval import tasks
from transformers import AutoModelForCausalLM, AutoTokenizer

import generation_kv_quant
import generation_kv_adaptive_v
import generation_kv_adaptive_k
from kv_cache_quant import DEFAULT_KIVI_GROUP_SIZE, DEFAULT_KIVI_RESIDUAL_LENGTH, KIVI_SCHEME
from model_utils import configure_block_size, default_small_block_size

MODEL_PATH = "Efficient-Large-Model/Fast_dLLM_v2_7B"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_BD_SIZE = 32
DEFAULT_THRESHOLD = 1.0
DEFAULT_NUM_FEWSHOT = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    return question.replace(
        "Answer:",
        "Please reason step by step, and put your final answer within \\boxed{}.",
    )


def build_chat_prompt(tokenizer, question: str) -> str:
    user_text = format_question(question) if "Answer:" in question else (
        f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load_task_samples(task: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if task == "hendrycks_math500":
        td = tasks.get_task_dict(["hendrycks_math500"])
        lm_task = td["hendrycks_math500"]
        lm_task._config.num_fewshot = DEFAULT_NUM_FEWSHOT
        lm_task.set_fewshot_seed(seed=seed)
        lm_task.build_all_requests(limit=n, rank=0, world_size=1)
        out = []
        for i, inst in enumerate(lm_task._instances):
            doc = inst.doc
            out.append(
                {
                    "idx": i,
                    "question": doc["problem"],
                    "gold": doc["answer"],
                    "grader": "math",
                }
            )
        return out

    if task == "gpqa_diamond":
        ds = load_dataset("aradhye/gpqa_diamond", split="train")
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:n]
        out = []
        for j, i in enumerate(idxs):
            row = ds[int(i)]
            out.append(
                {
                    "idx": j,
                    "question": row["problem"],
                    "gold": str(row["answer"]).strip().upper(),
                    "grader": "gpqa",
                }
            )
        return out

    if task == "gsm8k":
        td = tasks.get_task_dict(["gsm8k"])
        lm_task = td["gsm8k"]
        lm_task._config.num_fewshot = DEFAULT_NUM_FEWSHOT
        lm_task.set_fewshot_seed(seed=seed)
        lm_task.build_all_requests(limit=n, rank=0, world_size=1)
        out = []
        for i, inst in enumerate(lm_task._instances):
            doc = inst.doc
            gold = extract_gsm8k_answer(lm_task.doc_to_target(doc))
            out.append(
                {
                    "idx": i,
                    "question": doc["question"],
                    "gold": gold,
                    "grader": "gsm8k",
                }
            )
        return out

    raise ValueError(f"unknown task {task!r}")


def grade(grader: str, gold, pred) -> bool:
    if pred is None:
        return False
    if grader == "gpqa":
        return str(pred).strip().upper() == str(gold).strip().upper()
    if grader == "math":
        pg = normalize_math(extract_boxed(pred) or pred)
        gg = normalize_math(gold)
        return pg == gg
    return str(pred) == str(gold)


def extract_pred(grader: str, text: str):
    if grader == "gpqa":
        return extract_gpqa_choice(text)
    if grader == "math":
        boxed = extract_boxed(text)
        return boxed if boxed is not None else text.strip()[:120]
    return extract_gsm8k_answer(text)


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    threshold: float,
    kv_quant_bits: int = 0,
    kv_quant_scheme: str = "block",
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
) -> torch.Tensor:
    seq_len = torch.tensor([prompt_ids.shape[1]], device=prompt_ids.device)
    min_len = prompt_ids.shape[1]
    out = model.mdm_sample(
        prompt_ids,
        tokenizer=tokenizer,
        block_size=block_size,
        small_block_size=small_block_size,
        max_new_tokens=max_new_tokens,
        min_len=min_len,
        seq_len=seq_len,
        threshold=threshold,
        kv_quant_bits=kv_quant_bits,
        kv_quant_scheme=kv_quant_scheme,
        kivi_group_size=kivi_group_size,
        kivi_residual_length=kivi_residual_length,
    )
    return out[0]


@torch.no_grad()
def generate_one_adaptive_v(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    threshold: float,
    kv_quant_mode: str,
    high_precision_mass: float = 0.75,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
    adaptive_log: list | None = None,
) -> torch.Tensor:
    seq_len = torch.tensor([prompt_ids.shape[1]], device=prompt_ids.device)
    min_len = prompt_ids.shape[1]
    out = model.mdm_sample(
        prompt_ids,
        tokenizer=tokenizer,
        block_size=block_size,
        small_block_size=small_block_size,
        max_new_tokens=max_new_tokens,
        min_len=min_len,
        seq_len=seq_len,
        threshold=threshold,
        kv_quant_mode=kv_quant_mode,
        high_precision_mass=high_precision_mass,
        kivi_group_size=kivi_group_size,
        kivi_residual_length=kivi_residual_length,
        adaptive_log=adaptive_log,
    )
    return out[0]


@torch.no_grad()
def generate_one_adaptive_k(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    threshold: float,
    kv_quant_mode: str,
    k4_mass: float = 0.75,
    k2_mass_upper: float = 0.95,
    high_precision_mass: float = 0.75,
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
    adaptive_k_log: list | None = None,
    adaptive_v_log: list | None = None,
) -> torch.Tensor:
    seq_len = torch.tensor([prompt_ids.shape[1]], device=prompt_ids.device)
    min_len = prompt_ids.shape[1]
    out = model.mdm_sample(
        prompt_ids,
        tokenizer=tokenizer,
        block_size=block_size,
        small_block_size=small_block_size,
        max_new_tokens=max_new_tokens,
        min_len=min_len,
        seq_len=seq_len,
        threshold=threshold,
        kv_quant_mode=kv_quant_mode,
        k4_mass=k4_mass,
        k2_mass_upper=k2_mass_upper,
        high_precision_mass=high_precision_mass,
        kivi_group_size=kivi_group_size,
        kivi_residual_length=kivi_residual_length,
        adaptive_k_log=adaptive_k_log,
        adaptive_v_log=adaptive_v_log,
    )
    return out[0]


def detect_repetition_collapse(text: str, *, min_chunk: int = 40, min_repeats: int = 3) -> bool:
    """Heuristic: same substring repeated many times → generation collapse."""
    if len(text) < min_chunk * min_repeats:
        return False
    tail = text[-800:]
    for size in range(min_chunk, min(len(tail) // min_repeats + 1, 120)):
        chunk = tail[-size:]
        if chunk.count(chunk[: size // 2]) >= min_repeats * 2:
            return True
        if tail.count(chunk) >= min_repeats:
            return True
    return False


def summarize_adaptive_v_log(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "n_layer_block_rows": len(rows),
        "median_frac_v4": statistics.median(r["frac_v4"] for r in rows),
        "median_frac_v2": statistics.median(r["frac_v2"] for r in rows),
        "median_effective_v_bitwidth": statistics.median(r["effective_v_bitwidth"] for r in rows),
        "median_mass_v4": statistics.median(r["mass_v4"] for r in rows),
    }


def summarize_adaptive_k_log(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "n_block_rows": len(rows),
        "median_frac_k4": statistics.median(r["frac_k4"] for r in rows),
        "median_frac_k2": statistics.median(r["frac_k2"] for r in rows),
        "median_effective_k_bitwidth": statistics.median(r["effective_k_bitwidth"] for r in rows),
        "median_mass_k4": statistics.median(r["mass_k4"] for r in rows),
        "median_mass_k2": statistics.median(r["mass_k2"] for r in rows),
        "median_mass_k2_band": statistics.median(r["mass_k2_band"] for r in rows),
    }


def run_eval(
    model,
    tokenizer,
    samples: list[dict],
    *,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    threshold: float,
    kv_quant_bits: int = 0,
    kv_quant_scheme: str = "block",
    kivi_group_size: int = DEFAULT_KIVI_GROUP_SIZE,
    kivi_residual_length: int = DEFAULT_KIVI_RESIDUAL_LENGTH,
    run_label: str = "baseline",
    kv_quant_mode: str | None = None,
    high_precision_mass: float = 0.75,
    k4_mass: float = 0.75,
    k2_mass_upper: float = 0.95,
) -> dict:
    results = []
    correct = 0
    collapse_count = 0
    t0 = time.time()
    adaptive_v_rows: list = []
    adaptive_k_rows: list = []

    for sample in samples:
        prompt = build_chat_prompt(tokenizer, sample["question"])
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        prompt_len = input_ids.shape[1]

        if kv_quant_mode is not None:
            if kv_quant_mode in {"adaptive_k_mixed", "adaptive_k_mixed_force_k2"}:
                output_ids = generate_one_adaptive_k(
                    model,
                    tokenizer,
                    input_ids,
                    block_size=block_size,
                    small_block_size=small_block_size,
                    max_new_tokens=max_new_tokens,
                    threshold=threshold,
                    kv_quant_mode=kv_quant_mode,
                    k4_mass=k4_mass,
                    k2_mass_upper=k2_mass_upper,
                    high_precision_mass=high_precision_mass,
                    kivi_group_size=kivi_group_size,
                    kivi_residual_length=kivi_residual_length,
                    adaptive_k_log=adaptive_k_rows,
                    adaptive_v_log=adaptive_v_rows,
                )
            else:
                output_ids = generate_one_adaptive_v(
                    model,
                    tokenizer,
                    input_ids,
                    block_size=block_size,
                    small_block_size=small_block_size,
                    max_new_tokens=max_new_tokens,
                    threshold=threshold,
                    kv_quant_mode=kv_quant_mode,
                    high_precision_mass=high_precision_mass,
                    kivi_group_size=kivi_group_size,
                    kivi_residual_length=kivi_residual_length,
                    adaptive_log=adaptive_v_rows,
                )
        else:
            output_ids = generate_one(
                model,
                tokenizer,
                input_ids,
                block_size=block_size,
                small_block_size=small_block_size,
                max_new_tokens=max_new_tokens,
                threshold=threshold,
                kv_quant_bits=kv_quant_bits,
                kv_quant_scheme=kv_quant_scheme,
                kivi_group_size=kivi_group_size,
                kivi_residual_length=kivi_residual_length,
            )
        gen_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
        pred = extract_pred(sample["grader"], gen_text)
        is_correct = grade(sample["grader"], sample["gold"], pred)
        collapsed = detect_repetition_collapse(gen_text)
        collapse_count += int(collapsed)
        correct += int(is_correct)

        results.append(
            {
                "idx": sample["idx"],
                "gold": sample["gold"],
                "pred": pred,
                "correct": is_correct,
                "collapsed": collapsed,
                "generation": gen_text,
            }
        )
        print(
            f"[{run_label}] [{sample['idx'] + 1}/{len(samples)}] "
            f"correct={is_correct} collapsed={collapsed} pred={pred!r} gold={sample['gold']!r}"
        )

    elapsed = time.time() - t0
    acc = correct / len(samples) if samples else 0.0
    out = {
        "n_samples": len(samples),
        "correct": correct,
        "accuracy": acc,
        "collapse_count": collapse_count,
        "collapse_rate": collapse_count / len(samples) if samples else 0.0,
        "elapsed_sec": elapsed,
        "results": results,
    }
    if adaptive_v_rows:
        out["adaptive_v_stats"] = summarize_adaptive_v_log([r.to_dict() for r in adaptive_v_rows])
        out["adaptive_v_layer_block_stats"] = [r.to_dict() for r in adaptive_v_rows]
    if adaptive_k_rows:
        out["adaptive_k_stats"] = summarize_adaptive_k_log([r.to_dict() for r in adaptive_k_rows])
        out["adaptive_k_block_stats"] = [r.to_dict() for r in adaptive_k_rows]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["hendrycks_math500", "gpqa_diamond", "gsm8k"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--bd-size", type=int, default=DEFAULT_BD_SIZE)
    parser.add_argument("--small-block-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "kivi4",
            "kivi2",
            "kivi_suite",
            "k2v4_adaptive_suite",
            "uniform_k2v4",
            "adaptive_v_kivi",
            "adaptive_k_mixed",
            "adaptive_k_mixed_force_k2",
            "adaptive_k_mixed_suite",
        ],
        default="kivi_suite",
    )
    parser.add_argument(
        "--high-precision-mass",
        type=float,
        default=0.75,
        help="V4 tier cumulative attention mass threshold",
    )
    parser.add_argument(
        "--k4-mass",
        type=float,
        default=0.75,
        help="K4 tier cumulative attention mass threshold (adaptive_k_mixed)",
    )
    parser.add_argument(
        "--k2-mass-upper",
        type=float,
        default=0.95,
        help="Upper cumulative mass bound for K2 band logging (adaptive_k_mixed)",
    )
    parser.add_argument("--out-dir", default="checkpoints/kv_eval")
    args = parser.parse_args()

    set_seed(args.seed)
    sbs = args.small_block_size or default_small_block_size(args.bd_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model_path} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    configure_block_size(model, args.bd_size)

    adaptive_k_modes = {"adaptive_k_mixed", "adaptive_k_mixed_force_k2", "adaptive_k_mixed_suite"}
    adaptive_v_modes = {"k2v4_adaptive_suite", "uniform_k2v4", "adaptive_v_kivi"}

    def _bind_mdm_sample(mode_key: str, kw: dict) -> None:
        if kw.get("kv_quant_mode") in {"adaptive_k_mixed", "adaptive_k_mixed_force_k2"}:
            model.mdm_sample = types.MethodType(generation_kv_adaptive_k.batch_sample, model)
        elif kw.get("kv_quant_mode") is not None:
            model.mdm_sample = types.MethodType(generation_kv_adaptive_v.batch_sample, model)
        elif args.mode in adaptive_k_modes and mode_key.startswith("adaptive_k"):
            model.mdm_sample = types.MethodType(generation_kv_adaptive_k.batch_sample, model)
        elif args.mode in adaptive_v_modes:
            model.mdm_sample = types.MethodType(generation_kv_adaptive_v.batch_sample, model)
        else:
            model.mdm_sample = types.MethodType(generation_kv_quant.batch_sample, model)

    if args.mode in adaptive_k_modes and args.mode != "adaptive_k_mixed_suite":
        model.mdm_sample = types.MethodType(generation_kv_adaptive_k.batch_sample, model)
    elif args.mode in adaptive_v_modes:
        model.mdm_sample = types.MethodType(generation_kv_adaptive_v.batch_sample, model)
    else:
        model.mdm_sample = types.MethodType(generation_kv_quant.batch_sample, model)

    samples = load_task_samples(args.task, args.n, args.seed)
    print(f"Loaded {len(samples)} {args.task} samples")

    meta = {
        "model_path": args.model_path,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
        "bd_size": args.bd_size,
        "small_block_size": sbs,
        "threshold": args.threshold,
        "seed": args.seed,
        "n_samples": len(samples),
        "mode": args.mode,
        "kivi_group_size": DEFAULT_KIVI_GROUP_SIZE,
        "kivi_residual_length": DEFAULT_KIVI_RESIDUAL_LENGTH,
        "high_precision_mass": args.high_precision_mass,
        "k4_mass": args.k4_mass,
        "k2_mass_upper": args.k2_mass_upper,
        "policy_k2v4_adaptive": "K=KIVI INT2 uniform; V=INT4/INT2 by 75% attention mass",
        "policy_adaptive_k_mixed": "K=KIVI INT4/INT2 gather-by-tier; V=INT4/INT2 per-layer",
    }
    summary = {"meta": meta}
    common = dict(
        block_size=args.bd_size,
        small_block_size=sbs,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        kivi_group_size=DEFAULT_KIVI_GROUP_SIZE,
        kivi_residual_length=DEFAULT_KIVI_RESIDUAL_LENGTH,
    )

    common_adaptive_k = dict(
        k4_mass=args.k4_mass,
        k2_mass_upper=args.k2_mass_upper,
        high_precision_mass=args.high_precision_mass,
    )

    runs: list[tuple[str, dict]] = []
    if args.mode == "kivi_suite":
        runs = [
            ("baseline", dict(kv_quant_bits=0, kv_quant_scheme="block")),
            ("kv_kivi_int4", dict(kv_quant_bits=4, kv_quant_scheme=KIVI_SCHEME)),
            ("kv_kivi_int2", dict(kv_quant_bits=2, kv_quant_scheme=KIVI_SCHEME)),
        ]
    elif args.mode == "baseline":
        runs = [("baseline", dict(kv_quant_bits=0, kv_quant_scheme="block"))]
    elif args.mode == "kivi4":
        runs = [("kv_kivi_int4", dict(kv_quant_bits=4, kv_quant_scheme=KIVI_SCHEME))]
    elif args.mode == "kivi2":
        runs = [("kv_kivi_int2", dict(kv_quant_bits=2, kv_quant_scheme=KIVI_SCHEME))]
    elif args.mode == "uniform_k2v4":
        runs = [("uniform_k2v4", dict(kv_quant_mode="uniform_k2v4"))]
    elif args.mode == "adaptive_v_kivi":
        runs = [("adaptive_v_kivi", dict(kv_quant_mode="adaptive_v_kivi"))]
    elif args.mode == "k2v4_adaptive_suite":
        runs = [
            ("uniform_k2v4", dict(kv_quant_mode="uniform_k2v4")),
            ("adaptive_v_kivi", dict(kv_quant_mode="adaptive_v_kivi")),
        ]
    elif args.mode == "adaptive_k_mixed":
        runs = [("adaptive_k_mixed", dict(kv_quant_mode="adaptive_k_mixed"))]
    elif args.mode == "adaptive_k_mixed_force_k2":
        runs = [("adaptive_k_mixed_force_k2", dict(kv_quant_mode="adaptive_k_mixed_force_k2"))]
    elif args.mode == "adaptive_k_mixed_suite":
        # Baseline KIVI2 already in cross-dataset runs; suite = mixed + force-K2 sanity only.
        runs = [
            ("adaptive_k_mixed", dict(kv_quant_mode="adaptive_k_mixed")),
            ("adaptive_k_mixed_force_k2", dict(kv_quant_mode="adaptive_k_mixed_force_k2")),
        ]
    else:
        runs = [("kv_kivi_int2", dict(kv_quant_bits=2, kv_quant_scheme=KIVI_SCHEME))]

    for key, kw in runs:
        print(f"\n=== {key} ===")
        set_seed(args.seed)
        _bind_mdm_sample(key, kw)
        is_adaptive = kw.get("kv_quant_mode") is not None
        eval_kw = dict(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            run_label=key,
            **common,
        )
        if is_adaptive:
            eval_kw["kv_quant_mode"] = kw["kv_quant_mode"]
            eval_kw.update(common_adaptive_k)
        else:
            eval_kw.update(kw)
        summary[key] = run_eval(**eval_kw)
        print(f"{key} accuracy: {summary[key]['accuracy']:.1%}")
        if summary[key].get("collapse_count", 0):
            print(f"  collapse_rate: {summary[key]['collapse_rate']:.1%}")
        if "adaptive_k_stats" in summary[key]:
            st = summary[key]["adaptive_k_stats"]
            print(
                f"  K stats: eff_bits={st.get('median_effective_k_bitwidth', 0):.2f} "
                f"frac_k4={st.get('median_frac_k4', 0):.3f} "
                f"mass_k4={st.get('median_mass_k4', 0):.3f} "
                f"mass_k2={st.get('median_mass_k2', 0):.3f}"
            )
        if "adaptive_v_stats" in summary[key]:
            st = summary[key]["adaptive_v_stats"]
            print(
                f"  V stats: eff_bits={st.get('median_effective_v_bitwidth', 0):.2f} "
                f"frac_v4={st.get('median_frac_v4', 0):.3f} "
                f"mass_v4={st.get('median_mass_v4', 0):.3f}"
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

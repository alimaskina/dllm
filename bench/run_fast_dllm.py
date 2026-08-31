#!/usr/bin/env python3
"""Benchmark sweep matching the 7 setups from the note image.

Tasks: gsm8k / math500 (use --task).
Examples: use --num-examples (user wants 100).

Budgets: fixed topk K=32 and K=64 (replacing @2.5% and @5%).
Pass `--k 64` to run only that budget.

Setups (per image):
1) dense_fp16
2) fp16_all @32   (MAGE-proxy)  -> all_mean selector, sparse fp16 exec
3) fp16_all @64
4) fp16_middle @32 (HERALD-proxy) -> middle selector, sparse fp16 exec
5) fp16_middle @64
6) k4_16_v4 @32 (upper bound: attn keys fp16) -> keep-set from KIVI k4 probe,
   first-pass attention uses fp16 K/V, exec uses fp16 K and v4.
7) k4_16_v4 @64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

# Put both src/ (runtime) and bench/ (helpers) on sys.path.
_BENCH_DIR = Path(__file__).resolve().parent
_ROOT = _BENCH_DIR.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _BENCH_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from benchmark_utils import build_chat_prompt, extract_prediction, grade_sample, load_benchmark_samples, max_new_tokens_for_task  # noqa: E402
from config import ExperimentConfig, PrecisionConfig, SelectorConfig  # noqa: E402
from eval_utils_fast_dllm import attach_sampler, set_seed  # noqa: E402
from logging_utils import append_jsonl  # noqa: E402
from model_registry import load_model_and_tokenizer  # noqa: E402
from model_utils import configure_block_size, default_small_block_size  # noqa: E402


def _done_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.add((r["config"]["name"], r["task"], r["example_id"]))
    return out


@torch.no_grad()
def run_one(model, tokenizer, sample: dict, exp_cfg: ExperimentConfig) -> dict:
    task = sample["task"]
    max_gen = max_new_tokens_for_task(task)
    prompt = build_chat_prompt(tokenizer, sample["question"])
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = int(input_ids.shape[1])

    run_log: list = []
    seq_len = torch.tensor([prompt_len], device=model.device)
    small = exp_cfg.small_block_size or default_small_block_size(exp_cfg.block_size)
    common = dict(
        input_ids=input_ids,
        tokenizer=tokenizer,
        block_size=exp_cfg.block_size,
        small_block_size=small,
        max_new_tokens=max_gen,
        min_len=prompt_len,
        seq_len=seq_len,
        threshold=exp_cfg.threshold,
    )

    if exp_cfg.baseline == "original":
        out = model.mdm_sample(**common)
    else:
        out = model.mdm_sample(**common, exp_config=exp_cfg, experiment_log=run_log)

    output_ids = out[0]
    gen_text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
    pred = extract_prediction(sample["grader"], gen_text)
    correct = grade_sample(sample["grader"], sample["gold"], pred)

    state = run_log[-1] if run_log else {}
    covs_fp16 = [
        layer["fp16_ref_coverage"]
        for blk in state.get("blocks", [])
        for layer in blk.get("layers", {}).values()
        if layer.get("fp16_ref_coverage") is not None
    ]

    return {
        "task": task,
        "example_id": sample["idx"],
        "question": sample["question"],
        "gold": sample["gold"],
        "prediction": pred,
        "correct": bool(correct),
        "score": float(correct),
        "generated_answer": gen_text,
        "prompt_tokens": prompt_len,
        "gen_tokens": int(output_ids.shape[0] - prompt_len),
        "avg_topk_coverage_fp16_ref": sum(covs_fp16) / len(covs_fp16) if covs_fp16 else None,
        "config": exp_cfg.to_dict(),
        "num_blocks": len(state.get("blocks", [])),
    }


def dense_fp16_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="dense_fp16",
        baseline="original",
        sparse_old_cache=False,
    )


def fp16_all_cfg(*, k: int) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"fp16_all_k{k}",
        baseline="sparse_fp16",
        sparse_old_cache=True,
        selector=SelectorConfig(mode="all_mean", topk=k, topk_pct=None),
        selector_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
        exec_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
    )


def fp16_middle_cfg(*, k: int) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"fp16_middle_k{k}",
        baseline="sparse_fp16",
        sparse_old_cache=True,
        selector=SelectorConfig(mode="middle", topk=k, topk_pct=None),
        selector_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
        exec_precision=PrecisionConfig(k_bits="fp16", v_bits="fp16", q_bits="fp16"),
    )


def k4sel_v4_cfg(*, k: int) -> ExperimentConfig:
    """K and V both 4-bit KIVI (group_size=32) in the selector AND the exec
    phases. Keep-set picked from the selector attention which was itself
    computed on quantized K — no separate probe. Q stays fp16."""
    return ExperimentConfig(
        name=f"k4sel_v4_k{k}",
        baseline="sparse_quant_kv",
        sparse_old_cache=True,
        selector=SelectorConfig(mode="all_mean", topk=k, topk_pct=None),
        selector_precision=PrecisionConfig(k_bits="4", v_bits="4", q_bits="fp16"),  # type: ignore[arg-type]
        exec_precision=PrecisionConfig(k_bits="4", v_bits="4", q_bits="fp16"),  # type: ignore[arg-type]
        k_quant_scheme="kivi",
        v_quant_scheme="kivi",
        kivi_group_size=32,
        kivi_residual_length=32,
    )


def config_list(ks: list[int]) -> list[ExperimentConfig]:
    # Dense baseline is independent of k.
    out: list[ExperimentConfig] = [dense_fp16_cfg()]
    for k in ks:
        out.append(fp16_all_cfg(k=k))
        out.append(fp16_middle_cfg(k=k))
        out.append(k4sel_v4_cfg(k=k))
    return out


def summarize(records: list[dict]) -> dict:
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_key[(r["config"]["name"], r["task"])].append(r)
    rows = []
    for (cfg, task), rs in sorted(by_key.items()):
        n = len(rs)
        score = 100 * sum(r["score"] for r in rs) / n
        covs = [r["avg_topk_coverage_fp16_ref"] for r in rs if r.get("avg_topk_coverage_fp16_ref") is not None]
        cov = sum(covs) / len(covs) if covs else None
        rows.append({"config": cfg, "task": task, "n": n, "score_pct": round(score, 2), "cov_fp16ref": cov})
    return {"summaries": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("gsm8k", "math500"), required=True)
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--k",
        type=int,
        action="append",
        default=[],
        help="Fixed top-k budget(s). Repeatable. Default: 32 and 64.",
    )
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "results.jsonl"
    done = _done_keys(jsonl_path)

    set_seed(args.seed)
    device = torch.device(args.device)
    model, tokenizer, upstream = load_model_and_tokenizer("fast_dllm_v2_7b", device)
    ks = args.k or [32, 64]
    cfgs = config_list(ks)
    configure_block_size(model, cfgs[0].block_size)

    samples = load_benchmark_samples(args.task, num_examples=args.num_examples, seed=args.seed)
    print(f"{args.task} n={len(samples)} configs={len(cfgs)} on {args.device}", flush=True)

    for exp_cfg in cfgs:
        attach_sampler(model, exp_cfg, upstream)
        pending = [s for s in samples if (exp_cfg.name, args.task, s["idx"]) not in done]
        if not pending:
            print(f"\n=== {exp_cfg.name}: cached ===", flush=True)
            continue
        print(f"\n=== {exp_cfg.name}: RUN {len(pending)} ===", flush=True)
        since_analysis = 0
        for sample in pending:
            ex_id = sample["idx"]
            print(f"  ex{ex_id} START", flush=True)
            t0 = time.time()
            try:
                rec = run_one(model, tokenizer, sample, exp_cfg)
            except Exception:
                import traceback

                traceback.print_exc()
                raise
            append_jsonl(jsonl_path, rec)
            done.add((exp_cfg.name, args.task, ex_id))
            since_analysis += 1
            dt = time.time() - t0
            print(
                f"  ex{ex_id} score={rec['score']:.3f} cov_fp16ref={rec.get('avg_topk_coverage_fp16_ref')} {dt:.0f}s",
                flush=True,
            )
            if since_analysis >= 10:
                records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
                analysis = summarize(records)
                (out / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
                since_analysis = 0

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    analysis = summarize(records)
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nDone → {out}", flush=True)


if __name__ == "__main__":
    main()


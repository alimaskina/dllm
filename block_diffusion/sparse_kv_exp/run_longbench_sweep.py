#!/usr/bin/env python3
"""LongBench sweep: topk {64,128,256} × baseline / middle / extreme quant."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from config import ExperimentConfig  # noqa: E402
from eval_utils import attach_sampler, set_seed  # noqa: E402
from logging_utils import append_jsonl  # noqa: E402
from handoff_sweep_configs import (  # noqa: E402
    HANDOFF_FAMILIES,
    HANDOFF_NUM_EXAMPLES,
    HANDOFF_TOPK_PCTS,
    select_handoff_configs,
)
from longbench_sweep_configs import TOPKS, label_to_pct, select_sweep_configs  # noqa: E402
from longbench_utils import (  # noqa: E402
    DEFAULT_TASKS,
    load_longbench_samples,
    max_new_tokens_for_task,
    prepare_inputs,
    score_prediction,
)
from model_registry import get_model_spec, load_model_and_tokenizer  # noqa: E402
from model_utils import configure_block_size, default_small_block_size  # noqa: E402
from run_longbench import run_one, _done_keys  # noqa: E402


def _parse_config_meta(name: str) -> dict:
    if name == "baseline_dense_fp16":
        return {"family": "baseline", "topk": None, "topk_pct": None, "exec": "fp16"}
    m = re.match(r"middle_k(\d+)_fp16", name)
    if m:
        return {"family": "middle_fp16", "topk": int(m.group(1)), "topk_pct": None, "exec": "fp16"}
    m = re.match(r"middle_(p[\dp]+)_fp16", name)
    if m:
        return {
            "family": "middle_fp16",
            "topk": None,
            "topk_pct": label_to_pct(m.group(1)),
            "exec": "fp16",
        }
    m = re.match(r"uniform5_(p[\dp]+)_fp16", name)
    if m:
        return {
            "family": "uniform5",
            "topk": None,
            "topk_pct": label_to_pct(m.group(1)),
            "exec": "fp16",
        }
    m = re.match(r"extreme_k(\d+)_all_mean_k(\d+)v(\d+)", name)
    if m:
        return {
            "family": "extreme",
            "topk": int(m.group(1)),
            "topk_pct": None,
            "exec": f"k{m.group(2)}v{m.group(3)}",
        }
    m = re.match(r"extreme_(p[\dp]+)_all_mean_k(\d+)v(\d+)", name)
    if m:
        return {
            "family": "extreme",
            "topk": None,
            "topk_pct": label_to_pct(m.group(1)),
            "exec": f"k{m.group(2)}v{m.group(3)}",
        }
    return {"family": name, "topk": None, "topk_pct": None, "exec": "?"}


def summarize_sweep(records: list[dict]) -> dict:
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_key[(r["config"]["name"], r["task"])].append(r)

    rows = []
    for (cfg_name, task), rs in sorted(by_key.items()):
        meta = _parse_config_meta(cfg_name)
        scores = [r["score"] for r in rs]
        covs = [r["avg_topk_coverage"] for r in rs if r.get("avg_topk_coverage") is not None]
        caches = []
        for r in rs:
            by = (r.get("cost") or {}).get("by_cache_len") or {}
            if by:
                caches.append(min(int(k) for k in by))
        rows.append(
            {
                "config": cfg_name,
                "family": meta["family"],
                "topk": meta["topk"],
                "topk_pct": meta.get("topk_pct"),
                "exec": meta["exec"],
                "task": task,
                "n": len(rs),
                "score_pct": round(100 * sum(scores) / len(scores), 2),
                "avg_coverage": round(sum(covs) / len(covs), 4) if covs else None,
                "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in rs) / len(rs)),
                "avg_old_cache_first_block": round(sum(caches) / len(caches)) if caches else None,
                "avg_gen_tokens": round(sum(r["gen_tokens"] for r in rs) / len(rs)),
            }
        )
    return {"summaries": rows}


def _resolve_configs(
    *,
    sweep: str,
    topks: tuple[int, ...],
    topk_pcts: tuple[float, ...],
    families: list[str],
) -> tuple[list[ExperimentConfig], tuple[int, ...], tuple[float, ...]]:
    if sweep == "handoff":
        pcts = topk_pcts or HANDOFF_TOPK_PCTS
        fams = families if families != ["baseline", "middle", "extreme_k2v2", "extreme_k4v2", "extreme_k4v4"] else HANDOFF_FAMILIES
        return select_handoff_configs(topk_pcts=pcts, families=fams), (), tuple(pcts)
    return select_sweep_configs(topks=topks, topk_pcts=topk_pcts, families=families), topks, topk_pcts


def write_sweep_report(
    analysis: dict,
    path: Path,
    topks: tuple[int, ...] = (),
    topk_pcts: tuple[float, ...] = (),
    *,
    title: str = "LongBench Sweep Report",
    handoff: bool = False,
) -> None:
    rows = analysis["summaries"]
    tasks = sorted({r["task"] for r in rows})
    if handoff:
        families = [
            ("baseline", "baseline", None),
            ("middle_fp16", "middle", "fp16"),
            ("uniform5", "uniform5", "fp16"),
            ("extreme", "extreme", "k2v2"),
            ("extreme", "extreme", "k4v4"),
        ]
    else:
        families = [
            ("baseline", "baseline", None),
            ("middle_fp16", "middle", "fp16"),
            ("extreme", "extreme", "k2v2"),
            ("extreme", "extreme", "k4v2"),
            ("extreme", "extreme", "k4v4"),
        ]

    lines = [f"# {title}", ""]

    def write_table(task: str, col_key: str, columns: list, col_header: list[str] | None = None) -> None:
        if not columns:
            return
        headers = col_header or [str(c) for c in columns]
        lines.append(f"### {task} — {col_key}")
        lines.append("")
        header = "| Family | Exec | " + " | ".join(headers) + " |"
        sep = "|--------|------|" + "|".join(["------"] * len(columns)) + "|"
        lines.append(header)
        lines.append(sep)
        for family, fam_label, exec_label in families:
            if family == "baseline":
                r = next((x for x in rows if x["task"] == task and x["family"] == "baseline"), None)
                score = f"{r['score_pct']:.1f}" if r else "—"
                cells = [score] + ["—"] * (len(columns) - 1)
                lines.append(f"| baseline | fp16 | " + " | ".join(cells) + " |")
                continue
            cells = []
            for col in columns:
                if col_key == "topk":
                    r = next(
                        (
                            x
                            for x in rows
                            if x["task"] == task
                            and x["family"] == family
                            and x["topk"] == col
                            and x["exec"] == exec_label
                        ),
                        None,
                    )
                else:
                    r = next(
                        (
                            x
                            for x in rows
                            if x["task"] == task
                            and x["family"] == family
                            and x.get("topk_pct") == col
                            and x["exec"] == exec_label
                        ),
                        None,
                    )
                if r is None:
                    cells.append("—")
                else:
                    cov = f" ({r['avg_coverage']:.2f})" if r.get("avg_coverage") is not None else ""
                    cells.append(f"{r['score_pct']:.1f}{cov}")
            lines.append(f"| {fam_label} | {exec_label} | " + " | ".join(cells) + " |")
        lines.append("")

    for task in tasks:
        lines.append(f"## {task}")
        lines.append("")
        if topks:
            write_table(task, "fixed k", list(topks), [str(k) for k in topks])
        if topk_pcts:
            write_table(
                task,
                "cache %",
                list(topk_pcts),
                [f"{p:g}%" for p in topk_pcts],
            )
        lines.append("")

    lines.append(
        "_Coverage: fraction of reference old-cache mass (sum over all masked block "
        "queries, FP16 probe) retained by top-k. Comparable across middle / all_mean / "
        "quant. Percent configs: k = round(pct × old_cache_len) per block._"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        choices=("default", "handoff"),
        default="default",
        help="handoff = baseline + middle + uniform5 + 4×extreme × cache%% tiers",
    )
    parser.add_argument(
        "--model",
        default="fast_dllm_v2_7b",
        help="Model preset: fast_dllm_v2_7b | llada2_mini_16b",
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--topks", nargs="+", type=int, default=list(TOPKS))
    parser.add_argument(
        "--topk-pcts",
        nargs="+",
        type=float,
        default=[],
        help="Cache fraction in %% (e.g. 2.5 5 10 20). k=round(pct*old_cache_len) per block.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["baseline", "middle", "extreme_k2v2", "extreme_k4v2", "extreme_k4v4"],
        help="baseline | middle | uniform5 | extreme_k2v2 | extreme_k4v2 | extreme_k2v4 | extreme_k4v4",
    )
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--output-dir", type=str, default="results/longbench_sweep")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    handoff = args.sweep == "handoff"
    num_examples = args.num_examples
    if num_examples is None:
        num_examples = HANDOFF_NUM_EXAMPLES if handoff else 50
    topks = tuple(args.topks)
    topk_pcts = tuple(args.topk_pcts)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    meta_path = out_dir / "sweep_meta.json"

    if args.analyze_only:
        records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
        analysis = summarize_sweep(records)
        (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            handoff = meta.get("sweep") == "handoff"
            topks = tuple(meta.get("topks") or ())
            topk_pcts = tuple(meta.get("topk_pcts") or ())
        elif handoff and not topk_pcts:
            topk_pcts = HANDOFF_TOPK_PCTS
        write_sweep_report(
            analysis,
            out_dir / "report.md",
            topks,
            topk_pcts,
            handoff=handoff,
        )
        print(f"Report → {out_dir / 'report.md'}")
        return

    configs, report_topks, report_pcts = _resolve_configs(
        sweep=args.sweep,
        topks=topks,
        topk_pcts=topk_pcts,
        families=args.families,
    )
    meta_path.write_text(
        json.dumps(
            {
                "sweep": args.sweep,
                "model": args.model,
                "topks": list(report_topks),
                "topk_pcts": list(report_pcts),
                "families": args.families if not handoff else list(HANDOFF_FAMILIES),
                "num_examples": num_examples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    done = _done_keys(jsonl_path)
    set_seed(args.seed)

    device = torch.device(args.device)
    spec = get_model_spec(args.model)
    print(f"Model preset: {args.model} ({spec.hf_id})")
    print(
        f"Configs: {len(configs)}, examples/task: {num_examples}, "
        f"topks: {report_topks or '—'}, topk_pcts: {report_pcts or '—'}"
    )
    model, tokenizer, upstream = load_model_and_tokenizer(args.model, device)
    configure_block_size(model, configs[0].block_size)
    model_max = getattr(model.config, "max_position_embeddings", 32768)

    total_pending = 0
    for task in args.tasks:
        samples = load_longbench_samples(task, num_examples=num_examples, seed=args.seed)
        for exp_cfg in configs:
            total_pending += sum(
                1 for s in samples if (exp_cfg.name, task, s["idx"]) not in done
            )
    print(f"Pending runs: {total_pending}")

    for task in args.tasks:
        samples = load_longbench_samples(task, num_examples=num_examples, seed=args.seed)
        print(f"\n=== Task {task}: {len(samples)} examples ===")
        for exp_cfg in configs:
            attach_sampler(model, exp_cfg, upstream)
            pending = [s for s in samples if (exp_cfg.name, task, s["idx"]) not in done]
            if not pending:
                print(f"  {exp_cfg.name}: cached")
                continue
            print(f"  RUN {exp_cfg.name} ({len(pending)} ex)")
            for sample in pending:
                t0 = time.time()
                record = run_one(model, tokenizer, sample, exp_cfg, model_max_tokens=model_max)
                append_jsonl(jsonl_path, record)
                done.add((exp_cfg.name, task, sample["idx"]))
                dt = time.time() - t0
                print(
                    f"    ex{sample['idx']} score={record['score']:.3f} "
                    f"cov={record['avg_topk_coverage']} prompt={record['prompt_tokens']} "
                    f"gen={record['gen_tokens']} {dt:.1f}s"
                )

    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    analysis = summarize_sweep(records)
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    write_sweep_report(
        analysis,
        out_dir / "report.md",
        report_topks,
        report_pcts,
        handoff=handoff,
    )
    print(f"\nDone → {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run and report the recovery grid: score, coverage, resident bytes per cell.

Grid, per the study design:

  GSM8K / MATH-500   dense bf16 ceiling + b4 x k in {32, 16}
  LongBench          dense bf16 ceiling + b4 x rho in {2.5, 1}%

Budgets are swept with ``eval.quality --topk/--topk-percent``, so one config
file covers every cell and the cells cannot drift apart. Training happens at a
single budget (``train_recovery.py --train-topk``), which is what makes the
other cells a generalization test rather than a fit.

LongBench tasks are split two ways: tasks the adapter trained on are scored from
``--example-offset`` onward so the score is never measured on training examples,
and tasks it never saw are scored in full.

  python scripts/run_recovery_grid.py run --branch D --adapter runs/D/adapter \
      --device cuda:0 --out results/grid
  python scripts/run_recovery_grid.py report --out results/grid
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MATH_TASKS = ("gsm8k", "math500")
MATH_BUDGETS = (32, 16)                 # fixed token budgets
LONGBENCH_TRAINED = ("2wikimqa", "hotpotqa")
LONGBENCH_HELDOUT = ("musique", "narrativeqa")
LONGBENCH_BUDGETS = (2.5, 1.0)          # percent of the live prefix

DENSE_CONFIG = "configs/official_dense_bf16.yaml"
SPARSE_CONFIG = "configs/proposed_a_k4v4_p5.yaml"


def cells(args) -> list[dict]:
    out: list[dict] = []
    for task in MATH_TASKS:
        out.append(dict(task=task, cell="dense_bf16", config=DENSE_CONFIG, offset=0))
        for k in MATH_BUDGETS:
            out.append(dict(task=task, cell=f"b4_k{k}", config=SPARSE_CONFIG, topk=k, offset=0))
    for task in LONGBENCH_TRAINED + LONGBENCH_HELDOUT:
        off = args.example_offset if task in LONGBENCH_TRAINED else 0
        out.append(dict(task=task, cell="dense_bf16", config=DENSE_CONFIG, offset=off))
        for pct in LONGBENCH_BUDGETS:
            tag = f"b4_p{pct:g}".replace(".", "p")
            out.append(dict(task=task, cell=tag, config=SPARSE_CONFIG, topk_pct=pct, offset=off))
    return out


def run(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cells(args):
        dest = out_dir / f"{args.branch}__{c['task']}__{c['cell']}.jsonl"
        if dest.exists() and not args.overwrite:
            print(f"skip (exists) {dest.name}")
            continue
        cmd = [
            sys.executable, "-m", "bitsieve_fastdllm.eval.quality",
            "--config", str(ROOT / c["config"]),
            "--benchmark", c["task"],
            "--device", args.device,
            "--limit", str(args.limit),
            "--output", str(dest),
        ]
        if c["offset"]:
            cmd += ["--example-offset", str(c["offset"])]
        if "topk" in c:
            cmd += ["--topk", str(c["topk"])]
        if "topk_pct" in c:
            cmd += ["--topk-percent", str(c["topk_pct"])]
        if c["cell"] != "dense_bf16":
            cmd += ["--coverage"]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        print(f"\n== {args.branch} / {c['task']} / {c['cell']}")
        env = {"PYTHONPATH": str(ROOT / "src")}
        r = subprocess.run(cmd, cwd=ROOT, env={**dict(__import__("os").environ), **env})
        if r.returncode:
            print(f"FAILED: {' '.join(cmd)}")
            return r.returncode
    return 0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def report(args) -> int:
    rows = defaultdict(list)
    for path in sorted(Path(args.out).glob("*.jsonl")):
        if path.name.endswith(".partial"):
            continue
        branch, task, cell = path.stem.split("__")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows[(branch, task, cell)].append(json.loads(line))

    print(f"\n{'branch':12s} {'task':13s} {'cell':11s} {'n':>3s} {'score':>7s} "
          f"{'coverage':>9s} {'resident MB':>12s} {'vs bf16':>8s}")
    print("-" * 82)
    for (branch, task, cell), rs in sorted(rows.items()):
        rt = [r.get("runtime") or {} for r in rs]
        cov = _mean([
            (m.get("coverage") or {}).get("mass_abs_mean")
            if isinstance(m.get("coverage"), dict) else None
            for m in rt
        ])
        res = _mean([m.get("resident_cache_bytes") for m in rt])
        dense = _mean([m.get("dense_cache_equivalent_bytes") for m in rt])
        ratio = (res / dense) if (res and dense) else None
        print(f"{branch:12s} {task:13s} {cell:11s} {len(rs):>3d} "
              f"{_mean([r.get('score') for r in rs]):>7.4f} "
              f"{(f'{cov:.4f}' if cov is not None else '—'):>9s} "
              f"{(f'{res / 2**20:.1f}' if res else '—'):>12s} "
              f"{(f'{ratio:.3f}' if ratio else '—'):>8s}")
    print("\ncoverage = absolute share of the fp16 reference attention mass the "
          "selected prefix entries carry (mass_abs_mean)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--branch", required=True, help="label, e.g. A_none or D_sft_noise")
    r.add_argument("--adapter", help="LoRA adapter to merge; omit for branch A")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--limit", type=int, default=60)
    r.add_argument("--example-offset", type=int, default=200,
                   help="examples to skip on LongBench tasks the adapter trained on")
    r.add_argument("--out", required=True)
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(fn=run)

    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=report)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

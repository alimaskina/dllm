#!/usr/bin/env python3
"""Eviction study: what the cache keeps, at two storage widths.

Selection decides what a block *reads* and saves compute; eviction decides what
the cache *keeps* and is the only one of the two that saves memory. This runs
both retention policies against the do-nothing baseline, at bf16 and at 4 bits,
so the two effects can be read apart.

  bf16   no scale metadata at all, so both policies cost exactly the same per
         kept entry -- the cleanest quality-at-equal-memory comparison.
  k4v4   keys are quantized per 32-token group and a group survives as long as
         any one of its entries does, so a policy that keeps a scatter pays more
         metadata per entry than one that keeps a contiguous tail.

  python scripts/run_eviction_grid.py run --device cuda:0 --out results/eviction
  python scripts/run_eviction_grid.py report --out results/eviction
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A FIXED selector budget, not a percent one. 5% of a MATH-500 prefix is 19-52
# tokens while the problem statement alone runs 63-388, so a percent budget
# hides the question from the model and collapses the score for reasons that
# have nothing to do with eviction (the same trap scripts/run_suite.py warns
# about). At this budget selection barely bites on these lengths, which is what
# makes eviction the variable under test rather than a confound.
SPARSE_CONFIG = "configs/proposed_a_k4v4_k512.yaml"
DEFAULT_TOPK = 512
DENSE_CONFIG = "configs/official_dense_bf16.yaml"
POLICIES = ("none", "recent", "ema_recent")
WIDTHS = (("bf16", 16, 16), ("k4v4", 4, 4))


def cells(args) -> list[dict]:
    wanted = getattr(args, "arms", None)
    out = [dict(arm="dense_bf16", config=DENSE_CONFIG)]
    for tag, kb, vb in WIDTHS:
        for policy in POLICIES:
            out.append(
                dict(
                    arm=f"{tag}__{policy}",
                    config=SPARSE_CONFIG,
                    k_bits=kb,
                    v_bits=vb,
                    policy=policy,
                )
            )
    if wanted:
        keep = {a.strip() for a in wanted.split(",") if a.strip()}
        unknown = keep - {c["arm"] for c in out}
        if unknown:
            raise SystemExit(f"unknown arms: {sorted(unknown)}")
        out = [c for c in out if c["arm"] in keep]
    return out


def run(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cells(args):
        dest = out_dir / f"{c['arm']}.jsonl"
        if dest.exists() and not args.overwrite:
            have = sum(1 for l in dest.read_text(encoding="utf-8").splitlines() if l.strip())
            if have >= args.limit:
                print(f"skip (complete, {have} rows) {dest.name}")
                continue
            print(f"resuming {dest.name}: {have}/{args.limit}")
        cmd = [
            sys.executable, "-m", "bitsieve_fastdllm.eval.quality",
            "--config", str(ROOT / c["config"]),
            "--benchmark", args.benchmark,
            "--device", args.device,
            "--limit", str(args.limit),
            "--max-new-tokens", str(args.max_new_tokens),
            "--output", str(dest),
        ]
        if "k_bits" in c:
            cmd += ["--k-bits", str(c["k_bits"]), "--v-bits", str(c["v_bits"]),
                    "--eviction-policy", c["policy"],
                    "--eviction-capacity-floor", str(args.capacity_floor),
                    "--eviction-capacity-percent", str(args.capacity_percent),
                    "--eviction-window", str(args.window),
                    "--eviction-decay", str(args.decay),
                    "--eviction-interval", str(args.interval),
                    "--topk", str(args.topk),
                    "--coverage"]
        print(f"\n== {c['arm']}")
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        if r.returncode:
            print(f"FAILED: {' '.join(cmd)}")
            return r.returncode
    return 0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def report(args) -> int:
    rows = {}
    for path in sorted(Path(args.out).glob("*.jsonl")):
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if recs:
            rows[path.stem] = recs

    print(f"\n{'arm':22s} {'n':>3s} {'score':>7s} {'coverage':>9s} {'mem/full':>9s} "
          f"{'live':>7s} {'seen':>7s} {'evicted%':>9s}")
    print("-" * 82)
    for arm, recs in sorted(rows.items()):
        rt = [r.get("runtime") or {} for r in recs]
        cov = _mean([
            (m.get("coverage") or {}).get("mass_abs_mean")
            if isinstance(m.get("coverage"), dict) else None for m in rt
        ])
        mem = _mean([m.get("eviction_bytes_vs_unevicted") for m in rt])
        live = _mean([m.get("eviction_live_entries") for m in rt])
        seen = _mean([m.get("eviction_tokens_seen") for m in rt])
        # How often the policy actually had anything to do: an example whose
        # generation never outgrew C is identical to the baseline, and averaging
        # those in would hide the effect being measured.
        fired = [m for m in rt if (m.get("eviction_tokens_seen") or 0) > (m.get("eviction_capacity") or 1e9)]
        pct = 100.0 * len(fired) / len(rt) if rt else 0.0
        print(f"{arm:22s} {len(recs):>3d} {_mean([r.get('score') for r in recs]):>7.4f} "
              f"{(f'{cov:.4f}' if cov is not None else '—'):>9s} "
              f"{(f'{mem:.4f}' if mem is not None else '—'):>9s} "
              f"{(f'{live:.0f}' if live is not None else '—'):>7s} "
              f"{(f'{seen:.0f}' if seen is not None else '—'):>7s} "
              f"{pct:>8.0f}%")
    print("\nmem/full = live entries + policy state, over the same run keeping everything")
    print("evicted% = share of examples whose generation outgrew C at all; the rest")
    print("           are identical to the no-eviction arm by construction")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--benchmark", default="math500")
    r.add_argument("--limit", type=int, default=60)
    r.add_argument("--max-new-tokens", type=int, default=4096)
    r.add_argument("--capacity-floor", type=int, default=256)
    r.add_argument("--capacity-percent", type=float, default=5.0)
    r.add_argument("--window", type=int, default=128)
    r.add_argument("--decay", type=float, default=0.9)
    r.add_argument("--interval", type=int, default=4)
    r.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                   help="fixed selector budget; a percent budget is the wrong "
                        "axis on short math prompts")
    r.add_argument("--out", required=True)
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--arms", help="comma-separated subset, to split across GPUs")
    r.set_defaults(fn=run)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=report)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

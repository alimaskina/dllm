#!/usr/bin/env python3
"""Gold NLL on fixed probe set — buckets (k, t, whole/partial)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

ML = Path(__file__).resolve().parents[1]
if str(ML) not in sys.path:
    sys.path.insert(0, str(ML))

from model_loader import load_llada
from nll_analysis import assign_t_bins
from tokenization_audit import TOKENIZER_PRESETS
from trajectory_utils import confidence_trajectory

MASK_ID = 126336
T_BINS = (0.2, 0.3, 0.4)
T_TOL = 0.08


def k_bucket(k: int) -> str:
    if k == 2:
        return "k2"
    if k == 3:
        return "k3"
    if k >= 4:
        return "k4plus"
    return "k1"


def t_bucket(t: float) -> str | None:
    best = min(T_BINS, key=lambda tb: abs(tb - t))
    if abs(best - t) <= T_TOL:
        return f"t{best:.1f}"
    return None


def aggregate_buckets(records: list[dict]) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["k"] < 2:
            continue
        tb = t_bucket(r["t"])
        if tb is None:
            continue
        kb = k_bucket(r["k"])
        wp = "whole" if r.get("whole_word_unresolved") else "partial"
        key = f"{kb}|{tb}|{wp}"
        groups[key].append(r["nll"])

    out = {}
    for key, vals in sorted(groups.items()):
        out[key] = {"mean_nll": statistics.mean(vals), "n": len(vals)}
    return out


@torch.no_grad()
def eval_probe(
    model,
    probe: list[dict],
    *,
    device: str,
    steps: int,
) -> list[dict]:
    all_nll: list[dict] = []
    for sent in tqdm(probe, desc="probe", leave=False):
        ids = sent["token_ids"]
        wps = sent["word_positions"]
        metas = sent.get("word_metas")
        multi = [wp for wp in wps if len(wp) >= 2]
        if len(ids) < 8 or not multi:
            continue
        _, nll_records, _ = confidence_trajectory(
            model,
            ids,
            multi,
            oracle=True,
            mask_id=MASK_ID,
            steps=steps,
            remasking="low_confidence",
            device=device,
            collect_nll=True,
            word_metas=metas,
        )
        if nll_records:
            all_nll.extend(nll_records)
    assign_t_bins(all_nll, list(T_BINS))
    return all_nll


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="e.g. base, iid, word, span")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="HF checkpoint dir; default: code-root (base model)",
    )
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path("/home/alimaskina/dllm/model/LLaDA-8B-Base"),
        help="HF dir with configuration_llada.py / modeling_llada.py",
    )
    parser.add_argument(
        "--probe-json",
        type=Path,
        default=Path(__file__).resolve().parent / "probe_set_en.json",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    probe_data = json.loads(args.probe_json.read_text())
    probe = probe_data["sentences"]

    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = args.code_root
    code_root = args.code_root
    if not (code_root / "configuration_llada.py").exists():
        raise FileNotFoundError(f"Missing LLaDA code files under {code_root}")

    print(f"Loading {ckpt} ({args.run_name}) [code={code_root}]...")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    model = load_llada(str(code_root), args.device)
    weights_path = ckpt / "model.safetensors"
    if ckpt.resolve() != code_root.resolve() and weights_path.exists():
        from safetensors.torch import load_file

        model.load_state_dict(load_file(weights_path), strict=True)
    model.eval()

    records = eval_probe(model, probe, device=args.device, steps=args.steps)
    buckets = aggregate_buckets(records)

    out_path = args.out or (
        Path(__file__).resolve().parent / "results" / f"nll_probe_{args.run_name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": args.run_name,
        "checkpoint": str(ckpt),
        "n_probe_sentences": len(probe),
        "n_nll_records": len(records),
        "buckets": buckets,
        "t_bins": list(T_BINS),
        "t_tol": T_TOL,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(records)} records, {len(buckets)} buckets)")


if __name__ == "__main__":
    main()

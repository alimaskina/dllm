#!/usr/bin/env python3
"""Smoke-check coverage metrics on a few GSM8K examples."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from config import ExperimentConfig  # noqa: E402
from eval_utils import attach_sampler, build_chat_prompt, load_gsm8k_samples, set_seed  # noqa: E402
from generation import batch_sample_sparse_kv  # noqa: E402
from longbench_sweep_configs import longbench_sweep_grid  # noqa: E402
from model_utils import configure_block_size, default_small_block_size  # noqa: E402

_UPSTREAM = _ROOT.parent / "_fast_dllm_upstream" / "v2"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))
import generation_functions as upstream_gen  # noqa: E402


def aggregate_block_metrics(blocks: list[dict]) -> dict:
    cov, self_cov, overlap, qcov = [], [], [], []
    n_layers = 0
    for blk in blocks:
        for layer in blk.get("layers", {}).values():
            n_layers += 1
            if layer.get("attention_mass_captured") is not None:
                cov.append(layer["attention_mass_captured"])
            if layer.get("selector_self_coverage") is not None:
                self_cov.append(layer["selector_self_coverage"])
            if layer.get("rank_overlap_at_k") is not None:
                overlap.append(layer["rank_overlap_at_k"])
            if layer.get("quant_probe_coverage") is not None:
                qcov.append(layer["quant_probe_coverage"])
    out = {
        "n_blocks_with_selection": sum(1 for b in blocks if b.get("layers")),
        "n_layer_records": n_layers,
        "coverage_vs_ref_mean": sum(cov) / len(cov) if cov else None,
        "selector_self_cov_mean": sum(self_cov) / len(self_cov) if self_cov else None,
        "rank_overlap_mean": sum(overlap) / len(overlap) if overlap else None,
        "quant_probe_cov_mean": sum(qcov) / len(qcov) if qcov else None,
    }
    if blocks:
        first = next((b for b in blocks if b.get("layers")), None)
        if first:
            lid = next(iter(first["layers"]))
            out["first_block_sample"] = first["layers"][lid]
    return out


@torch.no_grad()
def run_one(model, tokenizer, sample: dict, exp_cfg: ExperimentConfig) -> dict:
    if exp_cfg.baseline == "original":
        model.mdm_sample = types.MethodType(upstream_gen.Fast_dLLM_QwenForCausalLM.batch_sample, model)
    else:
        model.mdm_sample = types.MethodType(batch_sample_sparse_kv, model)

    prompt = build_chat_prompt(tokenizer, sample["question"])
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = int(input_ids.shape[1])
    run_log: list = []
    seq_len = torch.tensor([prompt_len], device=model.device)
    small = exp_cfg.small_block_size or default_small_block_size(exp_cfg.block_size)
    out = model.mdm_sample(
        input_ids=input_ids,
        tokenizer=tokenizer,
        block_size=exp_cfg.block_size,
        small_block_size=small,
        max_new_tokens=256,
        min_len=prompt_len,
        seq_len=seq_len,
        threshold=exp_cfg.threshold,
        exp_config=exp_cfg,
        experiment_log=run_log,
    )
    gen_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    blocks = run_log[-1]["blocks"] if run_log else []
    return {
        "example_id": sample["idx"],
        "prompt_tokens": prompt_len,
        "gen_tokens": int(out[0].shape[0] - prompt_len),
        "answer_preview": gen_text[:120].replace("\n", " "),
        **aggregate_block_metrics(blocks),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--num-examples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    grid = longbench_sweep_grid((64,))
    configs = [
        grid["middle_k64"],
        grid["extreme_k64_k2v2"],
        grid["extreme_k64_k4v4"],
    ]

    device = torch.device(args.device)
    print(f"Loading model on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        "Efficient-Large-Model/Fast_dLLM_v2_7B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "Efficient-Large-Model/Fast_dLLM_v2_7B", trust_remote_code=True
    )
    configure_block_size(model, configs[0].block_size)

    samples = load_gsm8k_samples(args.num_examples, args.seed)
    print(f"Examples: {args.num_examples}, configs: {[c.name for c in configs]}\n")

    for cfg in configs:
        print("=" * 72)
        print(f"CONFIG: {cfg.name}  selector={cfg.selector.mode}  exec_k={cfg.exec_precision.k_bits}")
        for sample in samples:
            row = run_one(model, tokenizer, sample, cfg)
            print(f"\n  ex{row['example_id']}  prompt={row['prompt_tokens']} gen={row['gen_tokens']}")
            print(f"    blocks w/ selection: {row['n_blocks_with_selection']}  layer records: {row['n_layer_records']}")
            print(f"    coverage_vs_ref:     {row['coverage_vs_ref_mean']:.4f}" if row['coverage_vs_ref_mean'] is not None else "    coverage_vs_ref:     n/a")
            print(f"    selector_self_cov:   {row['selector_self_cov_mean']:.4f}" if row['selector_self_cov_mean'] is not None else "    selector_self_cov:   n/a")
            print(f"    rank_overlap@k:      {row['rank_overlap_mean']:.4f}" if row['rank_overlap_mean'] is not None else "    rank_overlap@k:      n/a")
            q = row["quant_probe_cov_mean"]
            print(f"    quant_probe_cov:     {q:.4f}" if q is not None else "    quant_probe_cov:     n/a (fp16 probe)")
            fs = row.get("first_block_sample") or {}
            print(
                f"    first block layer0: cov={fs.get('attention_mass_captured')} "
                f"self={fs.get('selector_self_coverage')} overlap={fs.get('rank_overlap_at_k')} "
                f"quant={fs.get('quant_probe_coverage')} masked_q={fs.get('masked_query_count')}"
            )
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze int8 KV cache quantization error (keys vs values, by layer/block/token)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from lm_eval import tasks
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_quant_metrics import analyze_cache_snapshot
from model_utils import configure_block_size, default_small_block_size
from run_gsm8k_kv_quant import build_chat_prompt, set_seed

MODEL_PATH = "Efficient-Large-Model/Fast_dLLM_v2_7B"
MASK_ID = 151665
STOP_TOKEN = 151645


def load_gsm8k_questions(n: int, seed: int) -> list[str]:
    task_dict = tasks.get_task_dict(["gsm8k"])
    task = task_dict["gsm8k"]
    task._config.num_fewshot = 0
    task.set_fewshot_seed(seed=seed)
    task.build_all_requests(limit=n, rank=0, world_size=1)
    return [inst.doc["question"] for inst in task._instances]


@torch.no_grad()
def collect_kv_snapshots(
    model,
    input_ids: torch.Tensor,
    *,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    threshold: float,
) -> list[dict]:
    """Run generation and snapshot KV cache after each completed block."""
    device = input_ids.device
    seq_len = torch.tensor([input_ids.shape[1]], device=device)
    min_len = input_ids.shape[1]
    prompt_len = input_ids.shape[1]

    snapshots: list[dict] = []
    num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
    num_small_blocks = block_size // small_block_size

    if min_len > block_size:
        output = model.forward(
            input_ids=input_ids[:, : (min_len // block_size * block_size)],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )
        past_key_values = output.past_key_values
        _save_snapshot(snapshots, past_key_values, block_idx=-1, label="prompt_warmstart")
    else:
        past_key_values = None

    seq_block_idx = seq_len // block_size
    start_block_idx = min_len // block_size

    for block_idx in range(start_block_idx, num_blocks):
        if STOP_TOKEN in input_ids[:, prompt_len:]:
            break

        if (seq_block_idx == block_idx).all():
            pad = MASK_ID * torch.ones(
                (1, block_size - input_ids.shape[1] % block_size), device=device, dtype=torch.long
            )
            input_ids = torch.cat([input_ids, pad], dim=1)
            x_init = input_ids
        else:
            x_init = input_ids[:, : (block_idx + 1) * block_size]

        x_t = x_init.clone()

        while True:
            mask_idx = x_t[:, -block_size:] == MASK_ID
            if mask_idx.sum() == 0:
                output = model.forward(
                    input_ids=x_t[:, -block_size:],
                    use_cache=True,
                    past_key_values=past_key_values,
                    update_past_key_values=True,
                    block_size=block_size,
                )
                past_key_values = output.past_key_values
                _save_snapshot(
                    snapshots,
                    past_key_values,
                    block_idx=block_idx,
                    label=f"gen_block_{block_idx - start_block_idx}",
                )
                next_token = output.logits[:, -1:, :].argmax(dim=-1)
                x_t = torch.cat([x_t, next_token], dim=1)
                break

            for sb in range(num_small_blocks):
                s0 = sb * small_block_size
                s1 = s0 + small_block_size
                start = -block_size + s0
                end = None if s1 == block_size else -block_size + s1
                while True:
                    mask_idx = x_t[:, -block_size:] == MASK_ID
                    if mask_idx[:, start:end].sum() == 0:
                        break
                    logits = model.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=False,
                    ).logits
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                    logits = logits[:, start:end]
                    x_1, p_1t = model.sample_with_top_p(logits, top_p=0.95, temperature=0.0)
                    x1_p = torch.squeeze(
                        torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1
                    )
                    x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)
                    unmask_idx = x1_p > threshold
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                    unmask_idx = unmask_idx & mask_idx[:, start:end]
                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                    if STOP_TOKEN in x_t[:, prompt_len:]:
                        stop_rel = (x_t[:, prompt_len:] == STOP_TOKEN).nonzero()
                        if stop_rel.numel() > 0 and (x_t[:, prompt_len : prompt_len + int(stop_rel[0, 1])] == MASK_ID).sum() == 0:
                            break

        if input_ids.shape[1] == x_t.shape[1]:
            input_ids = x_t
        else:
            input_ids[:, : (block_idx + 1) * block_size] = x_t[:, :-1]
            if (seq_block_idx == block_idx).all():
                input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
            else:
                if input_ids.shape[1] <= (block_idx + 1) * block_size:
                    input_ids = x_t
                else:
                    sel = seq_block_idx == block_idx
                    input_ids[sel, (block_idx + 1) * block_size] = x_t[sel, (block_idx + 1) * block_size]
        seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

        if STOP_TOKEN in input_ids[:, prompt_len:]:
            break

    return snapshots


def _save_snapshot(snapshots, past_key_values, block_idx: int, label: str) -> None:
    keys = [k.detach().clone() for k in past_key_values.key_cache]
    values = [v.detach().clone() for v in past_key_values.value_cache]
    snapshots.append(
        {
            "block_idx": block_idx,
            "label": label,
            "seq_len": keys[0].shape[-2],
            "keys": keys,
            "values": values,
        }
    )


def aggregate_snapshots(all_analyses: list[dict]) -> dict:
    """Pool analyses from multiple samples/snapshots."""
    n = len(all_analyses)
    if n == 0:
        return {}

    num_layers = all_analyses[0]["num_layers"]

    def _mean_path(items, path_fn):
        vals = [path_fn(x) for x in items]
        return sum(vals) / len(vals)

    per_layer_key = []
    per_layer_value = []
    for layer in range(num_layers):
        per_layer_key.append(
            {
                "mse": _mean_path(all_analyses, lambda a: a["per_layer_key"][layer]["mse"]),
                "cosine_mean": _mean_path(all_analyses, lambda a: a["per_layer_key"][layer]["cosine_mean"]),
                "rel_l2": _mean_path(all_analyses, lambda a: a["per_layer_key"][layer]["rel_l2"]),
                "max_abs": _mean_path(all_analyses, lambda a: a["per_layer_key"][layer]["max_abs"]),
            }
        )
        per_layer_value.append(
            {
                "mse": _mean_path(all_analyses, lambda a: a["per_layer_value"][layer]["mse"]),
                "cosine_mean": _mean_path(all_analyses, lambda a: a["per_layer_value"][layer]["cosine_mean"]),
                "rel_l2": _mean_path(all_analyses, lambda a: a["per_layer_value"][layer]["rel_l2"]),
                "max_abs": _mean_path(all_analyses, lambda a: a["per_layer_value"][layer]["max_abs"]),
            }
        )

    # Per token position within block
    block_size = len(all_analyses[0]["per_token_pos_key_cosine"])
    per_token_key_cos = [
        _mean_path(all_analyses, lambda a, t=t: a["per_token_pos_key_cosine"][t]) for t in range(block_size)
    ]
    per_token_value_cos = [
        _mean_path(all_analyses, lambda a, t=t: a["per_token_pos_value_cosine"][t]) for t in range(block_size)
    ]

    return {
        "n_snapshots": n,
        "global_key": {
            "mse": _mean_path(all_analyses, lambda a: a["global_key"]["mse"]),
            "cosine_mean": _mean_path(all_analyses, lambda a: a["global_key"]["cosine_mean"]),
            "rel_l2": _mean_path(all_analyses, lambda a: a["global_key"]["rel_l2"]),
        },
        "global_value": {
            "mse": _mean_path(all_analyses, lambda a: a["global_value"]["mse"]),
            "cosine_mean": _mean_path(all_analyses, lambda a: a["global_value"]["cosine_mean"]),
            "rel_l2": _mean_path(all_analyses, lambda a: a["global_value"]["rel_l2"]),
        },
        "per_layer_key": per_layer_key,
        "per_layer_value": per_layer_value,
        "per_token_pos_key_cosine": per_token_key_cos,
        "per_token_pos_value_cosine": per_token_value_cos,
    }


def render_report(agg: dict, block_size: int) -> str:
    lines = [
        "# KV cache int8 quantization error analysis",
        "",
        f"Block size: {block_size}, snapshots pooled: {agg['n_snapshots']}",
        "",
        "## Global (avg over layers & blocks)",
        "",
        "| tensor | MSE | RMSE≈√MSE | cosine | rel L2 |",
        "|--------|-----|-----------|--------|--------|",
    ]
    for name, g in [("Keys", agg["global_key"]), ("Values", agg["global_value"])]:
        lines.append(
            f"| {name} | {g['mse']:.2e} | {g['mse']**0.5:.2e} | {g['cosine_mean']:.6f} | {g['rel_l2']:.2e} |"
        )

    lines += [
        "",
        "## Keys vs Values",
        "",
        f"- Keys cosine: **{agg['global_key']['cosine_mean']:.6f}**, MSE: **{agg['global_key']['mse']:.2e}**",
        f"- Values cosine: **{agg['global_value']['cosine_mean']:.6f}**, MSE: **{agg['global_value']['mse']:.2e}**",
        f"- Values have {'higher' if agg['global_value']['mse'] > agg['global_key']['mse'] else 'lower'} MSE than keys",
        "",
        "## Per layer (first 5 / middle / last 5)",
        "",
        "| layer | K MSE | K cos | V MSE | V cos |",
        "|-------|-------|-------|-------|-------|",
    ]
    nl = len(agg["per_layer_key"])
    show = list(range(min(5, nl)))
    if nl > 10:
        show += list(range(nl // 2 - 2, nl // 2 + 3))
    show += list(range(max(nl - 5, 5), nl))
    show = sorted(set(show))

    for i in show:
        k = agg["per_layer_key"][i]
        v = agg["per_layer_value"][i]
        lines.append(
            f"| {i} | {k['mse']:.2e} | {k['cosine_mean']:.6f} | {v['mse']:.2e} | {v['cosine_mean']:.6f} |"
        )

    lines += [
        "",
        "## Cosine by token position within block (avg over layers & blocks)",
        "",
        "| pos | K cosine | V cosine |",
        "|-----|----------|----------|",
    ]
    for t in range(block_size):
        lines.append(
            f"| {t} | {agg['per_token_pos_key_cosine'][t]:.6f} | {agg['per_token_pos_value_cosine'][t]:.6f} |"
        )

    # Layer trend summary
    k_cos_by_layer = [x["cosine_mean"] for x in agg["per_layer_key"]]
    v_cos_by_layer = [x["cosine_mean"] for x in agg["per_layer_value"]]
    lines += [
        "",
        "## Trends",
        "",
        f"- Key cosine: min={min(k_cos_by_layer):.6f} (L{k_cos_by_layer.index(min(k_cos_by_layer))}), "
        f"max={max(k_cos_by_layer):.6f} (L{k_cos_by_layer.index(max(k_cos_by_layer))})",
        f"- Value cosine: min={min(v_cos_by_layer):.6f} (L{v_cos_by_layer.index(min(v_cos_by_layer))}), "
        f"max={max(v_cos_by_layer):.6f} (L{v_cos_by_layer.index(max(v_cos_by_layer))})",
        f"- Early layers (0-7) K cos avg: {np.mean(k_cos_by_layer[:8]):.6f}, V cos avg: {np.mean(v_cos_by_layer[:8]):.6f}",
        f"- Late layers (-8:) K cos avg: {np.mean(k_cos_by_layer[-8:]):.6f}, V cos avg: {np.mean(v_cos_by_layer[-8:]):.6f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="GSM8K samples")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bd-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out-dir", default="checkpoints/kv_quant_analysis")
    args = parser.parse_args()

    set_seed(args.seed)
    sbs = default_small_block_size(args.bd_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    configure_block_size(model, args.bd_size)

    questions = load_gsm8k_questions(args.n, args.seed)
    all_analyses = []

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] collecting KV snapshots...")
        prompt = build_chat_prompt(tokenizer, q)
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
        snapshots = collect_kv_snapshots(
            model,
            input_ids,
            block_size=args.bd_size,
            small_block_size=sbs,
            max_new_tokens=args.max_new_tokens,
            threshold=1.0,
        )
        for snap in snapshots:
            analysis = analyze_cache_snapshot(snap["keys"], snap["values"], args.bd_size)
            analysis["label"] = snap["label"]
            analysis["sample_idx"] = i
            all_analyses.append(analysis)
        print(f"  → {len(snapshots)} snapshots, seq_len up to {snapshots[-1]['seq_len'] if snapshots else 0}")

    agg = aggregate_snapshots(all_analyses)
    report = render_report(agg, args.bd_size)

    (out_dir / "analysis.json").write_text(
        json.dumps({"aggregated": agg, "per_snapshot": [{k: v for k, v in a.items() if k not in ("layer_block_key_mse", "layer_block_value_mse", "layer_block_key_cosine", "layer_block_value_cosine")} for a in all_analyses]}, indent=2)
    )
    (out_dir / "kv_quant_report.md").write_text(report)
    print(report)
    print(f"\nSaved → {out_dir}/")


if __name__ == "__main__":
    main()

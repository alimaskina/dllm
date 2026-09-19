"""Metrics for int8 KV cache quantization error analysis."""

from __future__ import annotations

import torch

from kv_cache_quant import dequantize_int8_per_block, quantize_int8_per_block


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    a_f = a.float()
    b_f = b.float()
    dot = (a_f * b_f).sum(dim=dim)
    na = a_f.norm(dim=dim).clamp(min=eps)
    nb = b_f.norm(dim=dim).clamp(min=eps)
    return dot / (na * nb)


def block_quant_error(
    tensor: torch.Tensor,
    block_size: int,
    block_idx: int,
) -> dict[str, float]:
    """Quantization error for one block in [B, H, S, D] tensor."""
    start = block_idx * block_size
    end = start + block_size
    block = tensor[..., start:end, :]
    q, scale = quantize_int8_per_block(block, block_size, 1)
    deq = dequantize_int8_per_block(q, scale, block_size)

    diff = block - deq
    fp_norm = block.float().norm()
    diff_norm = diff.float().norm()

    # Per-token cosine: [B, H, block_size]
    cos = _cosine_sim(block, deq, dim=-1)

    return {
        "mse": float(diff.pow(2).mean().item()),
        "rmse": float(diff.pow(2).mean().sqrt().item()),
        "max_abs": float(diff.abs().max().item()),
        "rel_l2": float((diff_norm / fp_norm.clamp(min=1e-8)).item()),
        "cosine_mean": float(cos.mean().item()),
        "cosine_min": float(cos.min().item()),
        "fp_norm_mean": float(block.float().norm(dim=-1).mean().item()),
        "scale_mean": float(scale.mean().item()),
        "scale_max": float(scale.max().item()),
    }


def per_token_cosine(
    tensor: torch.Tensor,
    block_size: int,
    block_idx: int,
) -> list[float]:
    """Cosine similarity per token position within one block, averaged over batch+heads."""
    start = block_idx * block_size
    end = start + block_size
    block = tensor[..., start:end, :]
    q, scale = quantize_int8_per_block(block, block_size, 1)
    deq = dequantize_int8_per_block(q, scale, block_size)
    cos = _cosine_sim(block, deq, dim=-1)  # [B, H, block_size]
    return cos.mean(dim=(0, 1)).tolist()


def per_token_mse(
    tensor: torch.Tensor,
    block_size: int,
    block_idx: int,
) -> list[float]:
    start = block_idx * block_size
    end = start + block_size
    block = tensor[..., start:end, :]
    q, scale = quantize_int8_per_block(block, block_size, 1)
    deq = dequantize_int8_per_block(q, scale, block_size)
    mse = (block - deq).pow(2).mean(dim=(-1, -3, -4))  # mean over head_dim, batch, heads -> [block_size]
    return mse.tolist()


def analyze_cache_snapshot(
    key_cache: list[torch.Tensor],
    value_cache: list[torch.Tensor],
    block_size: int,
) -> dict:
    """Full error breakdown for one KV cache snapshot."""
    num_layers = len(key_cache)
    seq_len = key_cache[0].shape[-2]
    num_blocks = seq_len // block_size

    per_layer_key = []
    per_layer_value = []
    # [layer][block_idx]
    layer_block_key_mse = [[None] * num_blocks for _ in range(num_layers)]
    layer_block_value_mse = [[None] * num_blocks for _ in range(num_layers)]
    layer_block_key_cos = [[None] * num_blocks for _ in range(num_layers)]
    layer_block_value_cos = [[None] * num_blocks for _ in range(num_layers)]

    token_cos_key = [[[] for _ in range(num_blocks)] for _ in range(num_layers)]
    token_cos_value = [[[] for _ in range(num_blocks)] for _ in range(num_layers)]

    for layer in range(num_layers):
        k_stats = []
        v_stats = []
        for b in range(num_blocks):
            ke = block_quant_error(key_cache[layer], block_size, b)
            ve = block_quant_error(value_cache[layer], block_size, b)
            k_stats.append(ke)
            v_stats.append(ve)
            layer_block_key_mse[layer][b] = ke["mse"]
            layer_block_value_mse[layer][b] = ve["mse"]
            layer_block_key_cos[layer][b] = ke["cosine_mean"]
            layer_block_value_cos[layer][b] = ve["cosine_mean"]
            token_cos_key[layer][b] = per_token_cosine(key_cache[layer], block_size, b)
            token_cos_value[layer][b] = per_token_cosine(value_cache[layer], block_size, b)

        def _agg(stats: list[dict], key: str) -> float:
            return sum(s[key] for s in stats) / len(stats)

        per_layer_key.append(
            {
                "mse": _agg(k_stats, "mse"),
                "rmse": _agg(k_stats, "rmse"),
                "max_abs": max(s["max_abs"] for s in k_stats),
                "rel_l2": _agg(k_stats, "rel_l2"),
                "cosine_mean": _agg(k_stats, "cosine_mean"),
                "cosine_min": min(s["cosine_min"] for s in k_stats),
            }
        )
        per_layer_value.append(
            {
                "mse": _agg(v_stats, "mse"),
                "rmse": _agg(v_stats, "rmse"),
                "max_abs": max(s["max_abs"] for s in v_stats),
                "rel_l2": _agg(v_stats, "rel_l2"),
                "cosine_mean": _agg(v_stats, "cosine_mean"),
                "cosine_min": min(s["cosine_min"] for s in v_stats),
            }
        )

    # Aggregate by block index across layers
    per_block_key = []
    per_block_value = []
    for b in range(num_blocks):
        per_block_key.append(
            {
                "mse": sum(layer_block_key_mse[l][b] for l in range(num_layers)) / num_layers,
                "cosine_mean": sum(layer_block_key_cos[l][b] for l in range(num_layers)) / num_layers,
            }
        )
        per_block_value.append(
            {
                "mse": sum(layer_block_value_mse[l][b] for l in range(num_layers)) / num_layers,
                "cosine_mean": sum(layer_block_value_cos[l][b] for l in range(num_layers)) / num_layers,
            }
        )

    # Per-token position within block (avg over layers and all blocks in snapshot)
    per_token_pos_key_cos = []
    per_token_pos_value_cos = []
    for t in range(block_size):
        vals_k = [
            token_cos_key[l][b][t]
            for l in range(num_layers)
            for b in range(num_blocks)
        ]
        vals_v = [
            token_cos_value[l][b][t]
            for l in range(num_layers)
            for b in range(num_blocks)
        ]
        per_token_pos_key_cos.append(sum(vals_k) / len(vals_k))
        per_token_pos_value_cos.append(sum(vals_v) / len(vals_v))

    def _global_agg(per_layer: list[dict]) -> dict:
        return {
            "mse": sum(x["mse"] for x in per_layer) / len(per_layer),
            "rmse": sum(x["rmse"] for x in per_layer) / len(per_layer),
            "max_abs": max(x["max_abs"] for x in per_layer),
            "rel_l2": sum(x["rel_l2"] for x in per_layer) / len(per_layer),
            "cosine_mean": sum(x["cosine_mean"] for x in per_layer) / len(per_layer),
            "cosine_min": min(x["cosine_min"] for x in per_layer),
        }

    return {
        "num_layers": num_layers,
        "num_blocks": num_blocks,
        "seq_len": seq_len,
        "global_key": _global_agg(per_layer_key),
        "global_value": _global_agg(per_layer_value),
        "per_layer_key": per_layer_key,
        "per_layer_value": per_layer_value,
        "per_block_key": per_block_key,
        "per_block_value": per_block_value,
        "per_token_pos_key_cosine": per_token_pos_key_cos,
        "per_token_pos_value_cosine": per_token_pos_value_cos,
        "layer_block_key_mse": layer_block_key_mse,
        "layer_block_value_mse": layer_block_value_mse,
        "layer_block_key_cosine": layer_block_key_cos,
        "layer_block_value_cosine": layer_block_value_cos,
    }

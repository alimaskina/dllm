from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

from ..cache import PackedKVCache
from ..config import QuantizationConfig
from ..kernels.ops import dense_packed_attention, gather_packed_kv, selector_topk
from ..reference import sdpa_compact


def _ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _time(fn: Callable[[], object], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='CUDA microbenchmarks for packed attention, selector, gather, and compact SDPA.')
    p.add_argument("--contexts", default="2048,8192,16384,32768")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--q-heads", type=int, default=28)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--selector-queries", type=int, default=5)
    p.add_argument("--topk", type=int, default=512)
    p.add_argument("--k-bits", type=int, default=4)
    p.add_argument("--v-bits", type=int, default=4)
    p.add_argument("--residual", type=int, default=32)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--output", required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("microbench requires CUDA")
    device = torch.device("cuda")
    rows = []
    for n in _ints(args.contexts):
        quant = QuantizationConfig(
            k_bits=args.k_bits,
            v_bits=args.v_bits,
            residual_tokens=args.residual,
        )
        cache = PackedKVCache(
            num_layers=1,
            batch_size=args.batch,
            num_kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            max_tokens=n,
            quant=quant,
            device=device,
            compute_dtype=torch.bfloat16,
            backend="triton",
        )
        cache.begin_append(n)
        key = torch.randn(
            args.batch, args.kv_heads, n, args.head_dim, device=device, dtype=torch.bfloat16
        )
        value = torch.randn_like(key)
        cache.stage_layer(0, key, value)
        cache.commit_append()
        del key, value
        view = cache.layer_view(0)
        query = torch.randn(
            args.batch,
            args.q_heads,
            args.block_size,
            args.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        current_k = torch.randn(
            args.batch,
            args.kv_heads,
            args.block_size,
            args.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        current_v = torch.randn_like(current_k)
        if args.selector_queries == 1:
            qidx = [args.block_size // 2]
        else:
            qidx = [
                round(i * (args.block_size - 1) / (args.selector_queries - 1))
                for i in range(args.selector_queries)
            ]
        selected = selector_topk(
            query,
            view,
            query_indices=qidx,
            topk=min(args.topk, n),
            current_key=current_k,
            backend="triton",
        )
        compact_k, compact_v = gather_packed_kv(
            view, selected.indices, dtype=torch.bfloat16, backend="triton"
        )
        full_k = torch.cat([compact_k, current_k], dim=2)
        full_v = torch.cat([compact_v, current_v], dim=2)

        funcs = {
            "dense_packed_attention": lambda: dense_packed_attention(
                query, view, current_k, current_v, backend="triton"
            ),
            "selector": lambda: selector_topk(
                query,
                view,
                query_indices=qidx,
                topk=min(args.topk, n),
                current_key=current_k,
                backend="triton",
            ),
            "gather_dequant": lambda: gather_packed_kv(
                view,
                selected.indices,
                dtype=torch.bfloat16,
                backend="triton",
                out_key=compact_k,
                out_value=compact_v,
            ),
            "compact_sdpa": lambda: sdpa_compact(query, full_k, full_v),
        }
        for name, fn in funcs.items():
            times = _time(fn, args.warmup, args.repeats)
            rows.append(
                {
                    "operation": name,
                    "context_tokens": n,
                    "batch": args.batch,
                    "q_heads": args.q_heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": args.head_dim,
                    "selector_queries": args.selector_queries,
                    "topk": min(args.topk, n),
                    "k_bits": args.k_bits,
                    "v_bits": args.v_bits,
                    "residual": args.residual,
                    "median_ms": statistics.median(times),
                    "p10_ms": sorted(times)[max(0, int(0.1 * len(times)) - 1)],
                    "p90_ms": sorted(times)[min(len(times) - 1, int(0.9 * len(times)))],
                }
            )
        print(json.dumps(rows[-4:], indent=2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

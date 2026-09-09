#!/usr/bin/env python3
"""Speed of the packed path at residual=0 vs the upstream residual=32.

Dropping the fp16 tail moves those 32 tokens from a small fp16 side-pass into
the packed kernel, so the fast path should be no slower - this measures it
rather than assuming it.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.cache import PackedKVCache  # noqa: E402
from bitsieve_fastdllm.config import QuantizationConfig  # noqa: E402
from bitsieve_fastdllm.kernels import ops  # noqa: E402

DEV = torch.device("cuda")
DTYPE = torch.bfloat16
B, HKV, GQA, D, BLOCK = 1, 4, 7, 128, 32
HQ = HKV * GQA


def timed(fn, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    os.environ.setdefault("BITSIEVE_CUDA_STRICT", "1")
    print(f"{'prefix':>8}  {'op':<10} {'r=32 (ms)':>10} {'r=0 (ms)':>10} {'delta':>9}")
    for prefix_len in (2048, 8192, 28672):
        results = {}
        for residual in (32, 0):
            quant = QuantizationConfig(
                k_bits=4, v_bits=4, key_token_group=32,
                value_channel_group=32, residual_tokens=residual,
            )
            cache = PackedKVCache(
                num_layers=1, batch_size=B, num_kv_heads=HKV, head_dim=D,
                max_tokens=prefix_len + 4 * BLOCK, quant=quant, device=DEV,
                compute_dtype=DTYPE, backend="auto",
            )
            torch.manual_seed(0)
            cache.begin_append(prefix_len)
            cache.stage_layer(
                0,
                torch.randn(B, HKV, prefix_len, D, device=DEV, dtype=DTYPE),
                torch.randn(B, HKV, prefix_len, D, device=DEV, dtype=DTYPE),
            )
            cache.commit_append()
            view = cache.layer_view(0)

            q = torch.randn(B, HQ, BLOCK, D, device=DEV, dtype=DTYPE)
            ck = torch.randn(B, HKV, BLOCK, D, device=DEV, dtype=DTYPE)
            cv = torch.randn(B, HKV, BLOCK, D, device=DEV, dtype=DTYPE)
            qi = list(range(0, BLOCK, 8))

            results[(residual, "dense")] = timed(
                lambda: ops.dense_packed_attention(
                    q, view, ck, cv, backend="triton", kernel_variant="blocked"
                )
            )
            results[(residual, "selector")] = timed(
                lambda: ops.selector_topk(
                    q, view, query_indices=qi, topk=512, current_key=ck,
                    domain="prefix", score_kind="softmax", scaling=D ** -0.5,
                    backend="triton", kernel_variant="blocked",
                )
            )
            del cache
            torch.cuda.empty_cache()

        for op in ("dense", "selector"):
            a, b = results[(32, op)], results[(0, op)]
            pct = (b - a) / a * 100.0
            print(f"{prefix_len:>8}  {op:<10} {a:>10.3f} {b:>10.3f} {pct:>+8.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Is the k_bits=16 ("no quantization") cache path as accurate as it claims?

sparse_fp16_all and sparse_k4v4_all are meant to differ ONLY by quantization, so
k4v4 should never beat fp16 by more than noise. In the first task-selection
results it beat it on 19 of the 23 examples where they differed (sign test
p<0.01), which that story does not explain.

They do not only differ by quantization: with k_bits=v_bits=16 the cache stores
k_fp/v_fp and `quantized_length` stays 0, so cuda_fast's `_can_fast` sees no
packed groups and falls back to ops.py's base Triton kernel, while the 4-bit
config runs the optimized packed kernel. Two different implementations.

This measures each path's error against an exact float32 attention reference on
the same inputs. If the fp16 path's error is materially worse than 4-bit's, the
"fp16" arm of the comparison is handicapped by its kernel rather than by having
more precision, and the k4v4-vs-mage comparison in the paper is not measuring
quantization at all.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.cache import PackedKVCache  # noqa: E402
from bitsieve_fastdllm.config import QuantizationConfig  # noqa: E402
from bitsieve_fastdllm.kernels import ops  # noqa: E402
from bitsieve_fastdllm.reference import dense_attention_reference  # noqa: E402

DEV = torch.device("cuda")
DTYPE = torch.bfloat16
B, HKV, GQA, D, BLOCK = 1, 4, 7, 128, 32
HQ = HKV * GQA
PREFIX = 2048


def build(k_bits: int, v_bits: int, key: torch.Tensor, value: torch.Tensor) -> PackedKVCache:
    quant = QuantizationConfig(
        k_bits=k_bits, v_bits=v_bits, key_token_group=32,
        value_channel_group=32, residual_tokens=0,
    )
    cache = PackedKVCache(
        num_layers=1, batch_size=B, num_kv_heads=HKV, head_dim=D,
        max_tokens=PREFIX + 4 * BLOCK, quant=quant, device=DEV,
        compute_dtype=DTYPE, backend="auto",
    )
    cache.begin_append(PREFIX)
    cache.stage_layer(0, key, value)
    cache.commit_append()
    return cache


def main() -> int:
    torch.manual_seed(0)
    key = torch.randn(B, HKV, PREFIX, D, device=DEV, dtype=DTYPE)
    value = torch.randn(B, HKV, PREFIX, D, device=DEV, dtype=DTYPE)
    torch.manual_seed(1)
    q = torch.randn(B, HQ, BLOCK, D, device=DEV, dtype=DTYPE)
    ck = torch.randn(B, HKV, BLOCK, D, device=DEV, dtype=DTYPE)
    cv = torch.randn(B, HKV, BLOCK, D, device=DEV, dtype=DTYPE)

    # Exact reference: float32 attention over the ORIGINAL bf16 tensors.
    exact = dense_attention_reference(
        q.float(), key.float(), value.float(), ck.float(), cv.float()
    )

    print(f"{'cache':<14}{'quantized_len':>14}{'residual':>10}{'max|err|':>12}{'mean|err|':>12}")
    results = {}
    for label, (kb, vb) in {"fp16 (16/16)": (16, 16), "k4v4": (4, 4), "k2v2": (2, 2)}.items():
        cache = build(kb, vb, key, value)
        view = cache.layer_view(0)
        out = ops.dense_packed_attention(
            q, view, ck, cv, backend="triton", kernel_variant="blocked"
        )
        err = (out.float() - exact).abs()
        results[label] = (err.max().item(), err.mean().item())
        print(
            f"{label:<14}{cache.quantized_length:>14}{cache.residual_length:>10}"
            f"{err.max().item():>12.5f}{err.mean().item():>12.6f}"
        )
        del cache
        torch.cuda.empty_cache()

    fp16_max, fp16_mean = results["fp16 (16/16)"]
    k4_max, k4_mean = results["k4v4"]
    print()
    ok = fp16_mean <= k4_mean
    print(
        f"  [{'PASS' if ok else 'FAIL'}] the fp16 cache path is at least as accurate as 4-bit "
        f"(mean |err| {fp16_mean:.6f} vs {k4_mean:.6f})"
    )
    if not ok:
        print(
            "    -> the 'no quantization' arm is LESS accurate than the quantized one, so\n"
            "       sparse_fp16_all vs sparse_k4v4_all is comparing kernels, not precision."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("BITSIEVE_CUDA_STRICT", "0")
    raise SystemExit(main())

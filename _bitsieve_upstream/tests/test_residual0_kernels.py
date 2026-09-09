#!/usr/bin/env python3
"""Does residual_tokens=0 keep the packed CUDA/Triton path alive and correct?

Removing the fp16 tail must not quietly push attention onto the slow torch
reference: with no residual, MORE of the prefix is packed, so the fast kernel
should cover strictly more work than before, not less. Run with
BITSIEVE_CUDA_STRICT=1 so an unavailable fast path raises instead of silently
falling back.

Synthetic tensors only - no 7B checkpoint needed.
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
from bitsieve_fastdllm.reference import (  # noqa: E402
    dense_attention_reference,
    select_topk_reference,
)

DEV = torch.device("cuda")
DTYPE = torch.bfloat16
B, HKV, GQA, D = 1, 4, 7, 128
HQ = HKV * GQA
BLOCK = 32


def build_cache(residual_tokens: int, prefix_len: int, k_bits: int, v_bits: int):
    quant = QuantizationConfig(
        k_bits=k_bits, v_bits=v_bits, key_token_group=32,
        value_channel_group=32, residual_tokens=residual_tokens,
    )
    cache = PackedKVCache(
        num_layers=1, batch_size=B, num_kv_heads=HKV, head_dim=D,
        max_tokens=prefix_len + 4 * BLOCK, quant=quant, device=DEV,
        compute_dtype=DTYPE, backend="auto",
    )
    torch.manual_seed(0)
    key = torch.randn(B, HKV, prefix_len, D, device=DEV, dtype=DTYPE)
    value = torch.randn(B, HKV, prefix_len, D, device=DEV, dtype=DTYPE)
    cache.begin_append(prefix_len)
    cache.stage_layer(0, key, value)
    cache.commit_append()
    return cache, key, value


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    os.environ.setdefault("BITSIEVE_CUDA_STRICT", "1")
    prefix_len = 2048
    failures = 0

    for k_bits, v_bits in ((4, 4), (2, 2)):
        for residual in (0, 32):
            tag = f"K{k_bits}/V{v_bits} residual={residual}"
            print(f"\n=== {tag} ===")
            cache, key_fp, value_fp = build_cache(residual, prefix_len, k_bits, v_bits)
            view = cache.layer_view(0)

            failures += not check(
                "cache length invariant",
                cache.quantized_length + cache.residual_length == prefix_len,
                f"packed={cache.quantized_length} residual={cache.residual_length}",
            )
            # With residual=0 every token must be packed - that is the point:
            # nothing is held in fp16, and the packed kernel owns the whole prefix.
            if residual == 0:
                failures += not check(
                    "whole prefix is packed (no fp16 tail)",
                    cache.residual_length == 0 and cache.quantized_length == prefix_len,
                )

            torch.manual_seed(1)
            q = torch.randn(B, HQ, BLOCK, D, device=DEV, dtype=DTYPE)
            cur_k = torch.randn(B, HKV, BLOCK, D, device=DEV, dtype=DTYPE)
            cur_v = torch.randn(B, HKV, BLOCK, D, device=DEV, dtype=DTYPE)

            # --- dense packed attention: fast kernel vs dequantized reference ---
            try:
                fast = ops.dense_packed_attention(
                    q, view, cur_k, cur_v, backend="triton", kernel_variant="blocked"
                )
                fast_ok = True
            except Exception as exc:  # strict mode makes a missing fast path loud
                fast_ok = False
                failures += not check("dense fast kernel ran", False, repr(exc)[:120])

            if fast_ok:
                deq_k, deq_v = cache.dequantize_layer(0, dtype=torch.float32)
                ref = dense_attention_reference(
                    q.float(), deq_k, deq_v, cur_k.float(), cur_v.float()
                )
                err = (fast.float() - ref).abs().max().item()
                rel = err / ref.abs().max().item()
                failures += not check(
                    "dense kernel matches dequantized reference",
                    rel < 5e-2, f"max|d|={err:.4g} rel={rel:.4g}"
                )

            # --- selector: fast kernel vs reference ranking on the same keys ---
            qi = list(range(0, BLOCK, 8))
            try:
                sel = ops.selector_topk(
                    q, view, query_indices=qi, topk=256, current_key=cur_k,
                    domain="prefix", score_kind="softmax", scaling=D ** -0.5,
                    backend="triton", kernel_variant="blocked",
                )
                sel_ok = True
            except Exception as exc:
                sel_ok = False
                failures += not check("selector fast kernel ran", False, repr(exc)[:120])

            if sel_ok:
                deq_k, _ = cache.dequantize_layer(0, dtype=torch.float32)
                ref_idx, _ = select_topk_reference(
                    q.float(), deq_k, query_indices=qi, topk=256,
                    current_key=cur_k.float(), domain="prefix", score_kind="softmax",
                )
                inter = [
                    len(set(sel.indices[0, h].tolist()) & set(ref_idx[0, h].tolist())) / 256.0
                    for h in range(HKV)
                ]
                worst = min(inter)
                failures += not check(
                    "selector top-k agrees with reference",
                    worst > 0.9, f"worst head overlap={worst:.3f}"
                )

            # --- gather must survive a residual-free view ---
            try:
                gk, gv = ops.gather_packed_kv(
                    view, sel.indices if sel_ok else None, dtype=DTYPE, backend="triton"
                ) if sel_ok else (None, None)
                failures += not check(
                    "gather returned selected K/V",
                    gk is not None and gk.shape[2] == 256,
                    f"shape={tuple(gk.shape) if gk is not None else None}",
                )
            except Exception as exc:
                failures += not check("gather fast kernel ran", False, repr(exc)[:120])

            del cache
            torch.cuda.empty_cache()

    print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

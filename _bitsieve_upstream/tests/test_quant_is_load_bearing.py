#!/usr/bin/env python3
"""Is the quantized store actually what selection and attention read?

Reading the code is not enough - a selector that ranks on a full-precision copy
while claiming to rank on packed keys looks identical in the source. So this
perturbs the packed payload in place and requires everything downstream to move:

  * no full-precision K/V store exists at all in a quantized cache
  * perturbing packed K changes the selected indices    (selection reads packed K)
  * perturbing packed K changes the attention output    (pass 1 reads packed K)
  * perturbing packed V leaves selection alone but changes the output
    (V cannot influence a ranking, but must influence the result)
  * the gathered compact buffer carries quantization error, i.e. it is not a
    copy of the original bf16 tensors
  * K2 differs from K4 differs from bf16 - the bit width is load-bearing, not
    a label

Synthetic tensors; no checkpoint needed.
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

DEV = torch.device("cuda")
DTYPE = torch.bfloat16
B, HKV, GQA, D, BLOCK = 1, 4, 7, 128, 32
HQ = HKV * GQA
PREFIX = 2048
TOPK = 256

FAILURES: list[str] = []


def chk(ok: bool, msg: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        FAILURES.append(msg)


def make_cache(k_bits: int, v_bits: int, key: torch.Tensor, value: torch.Tensor):
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


def select_and_attend(cache, q, ck, cv):
    view = cache.layer_view(0)
    sel = ops.selector_topk(
        q, view, query_indices=list(range(0, BLOCK, 8)), topk=TOPK,
        current_key=ck, domain="prefix", score_kind="softmax",
        scaling=D ** -0.5, backend="triton", kernel_variant="blocked",
    )
    out = ops.dense_packed_attention(
        q, view, ck, cv, backend="triton", kernel_variant="blocked"
    )
    gk, gv = ops.gather_packed_kv(view, sel.indices, dtype=DTYPE, backend="triton")
    return sel.indices.clone(), out.clone(), gk.clone(), gv.clone()


def main() -> int:
    os.environ.setdefault("BITSIEVE_CUDA_STRICT", "1")
    torch.manual_seed(0)
    key = torch.randn(B, HKV, PREFIX, D, device=DEV, dtype=DTYPE)
    value = torch.randn(B, HKV, PREFIX, D, device=DEV, dtype=DTYPE)
    torch.manual_seed(1)
    q = torch.randn(B, HQ, BLOCK, D, device=DEV, dtype=DTYPE)
    ck = torch.randn(B, HKV, BLOCK, D, device=DEV, dtype=DTYPE)
    cv = torch.randn(B, HKV, BLOCK, D, device=DEV, dtype=DTYPE)

    print("=== a quantized cache holds no full-precision copy ===")
    cache = make_cache(4, 4, key, value)
    chk(cache.k_fp is None, "cache.k_fp is None (no fp16 key store to leak from)")
    chk(cache.v_fp is None, "cache.v_fp is None (no fp16 value store to leak from)")
    view = cache.layer_view(0)
    chk(view.k_fp is None and view.v_fp is None, "the layer view exposes no fp16 K/V")
    chk(
        view.k_q is not None and view.k_q.dtype == torch.uint8,
        f"K lives in a uint8 payload (dtype={None if view.k_q is None else view.k_q.dtype})",
    )
    chk(
        cache.residual_length == 0,
        f"no float residual either (residual_length={cache.residual_length})",
    )

    idx0, out0, gk0, gv0 = select_and_attend(cache, q, ck, cv)

    print("\n=== the gathered compact buffer carries quantization error ===")
    # It is dequantized from the packed store, so it must NOT equal the original
    # bf16 keys at the selected positions.
    orig_at_sel = torch.gather(
        key, 2, idx0.unsqueeze(-1).expand(*idx0.shape, D)
    )
    err = (gk0.float() - orig_at_sel.float()).abs().max().item()
    chk(err > 1e-3, f"gathered K differs from the original bf16 K (max|d|={err:.4g})")

    print("\n=== perturbing packed K moves selection AND the first pass ===")
    assert cache.k_q is not None
    saved = cache.k_q.clone()
    # Flip the low nibble of every packed byte: a real change to the stored
    # codes, leaving shapes, scales and zero-points untouched.
    cache.k_q ^= 0x0F
    idx1, out1, _, _ = select_and_attend(cache, q, ck, cv)
    changed_frac = (idx1 != idx0).float().mean().item()
    chk(changed_frac > 0.1, f"selected indices changed ({changed_frac:.1%} of slots)")
    dout = (out1.float() - out0.float()).abs().max().item()
    chk(dout > 1e-3, f"attention output changed (max|d|={dout:.4g})")
    cache.k_q.copy_(saved)

    print("\n=== perturbing packed V leaves the ranking alone, moves the output ===")
    assert cache.v_q is not None
    saved_v = cache.v_q.clone()
    cache.v_q ^= 0x0F
    idx2, out2, _, gv2 = select_and_attend(cache, q, ck, cv)
    chk(
        torch.equal(idx2, idx0),
        "selected indices are unchanged (V cannot affect a QK ranking)",
    )
    dout_v = (out2.float() - out0.float()).abs().max().item()
    chk(dout_v > 1e-3, f"attention output changed (max|d|={dout_v:.4g})")
    dgv = (gv2.float() - gv0.float()).abs().max().item()
    chk(dgv > 1e-3, f"gathered V changed (max|d|={dgv:.4g})")
    cache.v_q.copy_(saved_v)
    del cache
    torch.cuda.empty_cache()

    print("\n=== the bit width is load-bearing: bf16 vs K4 vs K2 all differ ===")
    runs = {}
    for bits in (16, 4, 2):
        c = make_cache(bits, bits, key, value)
        runs[bits] = select_and_attend(c, q, ck, cv)
        del c
        torch.cuda.empty_cache()
    for a, b in ((16, 4), (16, 2), (4, 2)):
        ia, oa = runs[a][0], runs[a][1]
        ib, ob = runs[b][0], runs[b][1]
        frac = (ia != ib).float().mean().item()
        d = (oa.float() - ob.float()).abs().max().item()
        chk(
            frac > 0.0 and d > 1e-3,
            f"K{a}/V{a} vs K{b}/V{b}: indices differ in {frac:.1%} of slots, "
            f"output max|d|={d:.4g}",
        )
    # And lower precision must drift further from bf16, not less.
    d4 = (runs[16][1].float() - runs[4][1].float()).abs().mean().item()
    d2 = (runs[16][1].float() - runs[2][1].float()).abs().mean().item()
    chk(d2 > d4, f"2-bit drifts further from bf16 than 4-bit ({d2:.4g} > {d4:.4g})")

    print(f"\n{'ALL PASS' if not FAILURES else str(len(FAILURES)) + ' FAILURE(S)'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())

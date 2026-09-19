#!/usr/bin/env python3
"""On the real 7B model: does the training forward reproduce the decoder?

The training forward is hand-written (see training/blockdiff.py for why), so
the claim that "what we train on is what we run" needs checking against the
actual decoder, not asserting.

  1. Parity      - with an exact cache and no selector, the training forward
                   must reproduce the logits the real Fast-dLLM decoding path
                   produces for the same partially-masked block on the same
                   clean prefix.
  2. Localized   - cache degradation must move only blocks that read a prefix.
                   Block 0 has none, so it must come back bit-identical.
  3. Selection   - same, for the top-k budget.
  4. Gradients   - LoRA parameters get finite, non-zero gradients through the
                   degraded path with gradient checkpointing on.

Run: python tests/training_parity_e2e.py --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import SelectorConfig  # noqa: E402
from bitsieve_fastdllm.eval.common import load_fast_dllm  # noqa: E402
from bitsieve_fastdllm.training.blockdiff import (  # noqa: E402
    blockdiff_logits,
    build_masks,
)
from bitsieve_fastdllm.training.data import (  # noqa: E402
    MASK_TOKEN_ID,
    encode_example,
    load_math_train,
)
from bitsieve_fastdllm.training.degrade import CacheDegradation  # noqa: E402


def reference_block_logits(model, ids, xt_block, block_start, block_size):
    """Raw (unshifted) logits for one block through the real decoding path."""
    with torch.no_grad():
        past = None
        if block_start:
            past = model.forward(
                input_ids=ids[:, :block_start], use_cache=True,
                update_past_key_values=True, block_size=block_size,
            ).past_key_values
        return model.forward(
            input_ids=xt_block, use_cache=True, past_key_values=past,
            update_past_key_values=False, block_size=block_size,
        ).logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--mask-frac", type=float, default=0.8)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_fast_dllm(args.model, dtype=torch.bfloat16, device=args.device)
    model.eval()
    device = next(model.parameters()).device
    model.model.bd_size = args.block_size
    model.config.bd_size = args.block_size
    bs = args.block_size

    ex = None
    for row in load_math_train(limit=16, seed=0, exclude_math500=False):
        ex = encode_example(
            tokenizer, row["problem"], row["solution"],
            block_size=bs, max_len=args.max_len, device=device,
        )
        if ex is not None and ex.answer_len > 2 * bs:
            break
    if ex is None:
        print("no usable example")
        return 1

    ids = ex.input_ids.unsqueeze(0)
    labels = ex.labels.unsqueeze(0)
    seq = ids.shape[1]
    n_blocks = seq // bs
    print(f"\nexample: seq={seq} ({n_blocks} blocks) prompt={ex.prompt_len} answer={ex.answer_len}")

    answer = labels[0] != -100
    cand = [j for j in range(1, n_blocks) if bool(answer[j * bs : (j + 1) * bs].all())]
    if not cand:
        print("no fully-answer block")
        return 1
    j = cand[len(cand) // 2]
    start = j * bs
    print(f"probing block {j} (positions {start}..{start + bs})")

    g = torch.Generator(device=device).manual_seed(args.seed)
    sel = torch.rand(bs, device=device, generator=g) < args.mask_frac
    xt = ids.clone()
    xt[0, start : start + bs][sel] = MASK_TOKEN_ID

    doubled = torch.cat([xt, ids], dim=1)
    is_masked = torch.cat(
        [xt[0] == MASK_TOKEN_ID, torch.zeros(seq, dtype=torch.bool, device=device)]
    )
    masks = build_masks(seq, bs, device)
    exact = CacheDegradation(mode="exact")

    ok = []

    # ---- 1. parity ------------------------------------------------------
    ref = reference_block_logits(model, ids, xt[:, start : start + bs], start, bs).float()
    with torch.no_grad():
        mine = blockdiff_logits(
            model, doubled, masks=masks, degrade=exact, selector=None,
            is_masked_token=is_masked, gradient_checkpointing=False,
        ).float()

    # Global position p is predicted from the logit at p-1; compare that span.
    a = mine[0, start : start + bs - 1]
    b = ref[0, 0 : bs - 1]
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    kl = F.kl_div(
        F.log_softmax(a, -1), F.log_softmax(b, -1), log_target=True, reduction="batchmean"
    ).item()
    rel = ((a - b).abs().max() / b.abs().max()).item()
    print("\n[1] parity vs the real decoding path")
    print(f"    max|d| / max|ref|  = {rel:.3e}")
    print(f"    top-1 agreement    = {agree:.3f}")
    print(f"    KL(mine || ref)    = {kl:.3e} nats")
    ok.append(("parity", agree > 0.95 and kl < 1e-2))

    # Manual-kernel clean baseline, so 2 and 3 isolate the degradation rather
    # than the SDPA -> manual kernel switch.
    quant0 = CacheDegradation(mode="quant", k_bits=16, v_bits=16)
    nosel = SelectorConfig(mode="uniform", uniform_queries=5, topk=10**9, dense_prefix_layers=0)
    with torch.no_grad():
        base = blockdiff_logits(
            model, doubled, masks=masks, degrade=quant0, selector=nosel,
            is_masked_token=is_masked, gradient_checkpointing=False,
        ).float()
    print(f"\n[1b] manual kernel vs fused SDPA: "
          f"max|d|/max = {((base - mine).abs().max() / mine.abs().max()).item():.3e}, "
          f"top-1 {(base.argmax(-1) == mine.argmax(-1)).float().mean().item():.3f}")

    # ---- 2. degradation is localized ------------------------------------
    deg = CacheDegradation(mode="quant", k_bits=args.bits, v_bits=args.bits)
    with torch.no_grad():
        noisy = blockdiff_logits(
            model, doubled, masks=masks, degrade=deg, selector=None,
            is_masked_token=is_masked, gradient_checkpointing=False,
        ).float()
    d_probe = (noisy[0, start : start + bs] - base[0, start : start + bs]).abs().max().item()
    d_first = (noisy[0, :bs] - base[0, :bs]).abs().max().item()
    print(f"\n[2] packed-cache quantization (k{args.bits}/v{args.bits})")
    print(f"    max|d| on block {j}  = {d_probe:.4f}")
    print(f"    max|d| on block 0   = {d_first:.4f}  (no prefix -> must be 0)")
    ok.append(("degradation is localized", d_probe > 1e-3 and d_first < 1e-5))

    # ---- 3. selection is localized --------------------------------------
    selector = SelectorConfig(
        mode="uniform", uniform_queries=5, topk=args.topk, dense_prefix_layers=0
    )
    with torch.no_grad():
        sparse = blockdiff_logits(
            model, doubled, masks=masks, degrade=quant0, selector=selector,
            is_masked_token=is_masked, gradient_checkpointing=False,
        ).float()
    s_probe = (sparse[0, start : start + bs] - base[0, start : start + bs]).abs().max().item()
    s_first = (sparse[0, :bs] - base[0, :bs]).abs().max().item()
    print(f"\n[3] per-KV-head top-k over the prefix (k={args.topk})")
    print(f"    max|d| on block {j}  = {s_probe:.4f}")
    print(f"    max|d| on block 0   = {s_first:.4f}  (no prefix -> must be 0)")
    ok.append(("selection is localized", s_probe > 1e-3 and s_first < 1e-5))

    # ---- 4. gradients ----------------------------------------------------
    from peft import LoraConfig, get_peft_model

    pm = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    pm.eval()
    out = blockdiff_logits(
        pm, doubled, masks=masks, degrade=deg, selector=selector,
        is_masked_token=is_masked, gradient_checkpointing=True,
    )
    loss = F.cross_entropy(
        out[0, start : start + bs - 1].float(), labels[0, start + 1 : start + bs]
    )
    loss.backward()
    grads = [
        p.grad.abs().mean().item()
        for p in pm.parameters()
        if p.requires_grad and p.grad is not None
    ]
    finite = all(torch.isfinite(torch.tensor(v)) for v in grads)
    print(f"\n[4] gradients through the degraded path (LoRA + checkpointing)")
    print(f"    loss = {loss.item():.4f}, tensors with grad = {len(grads)}, "
          f"non-zero = {sum(1 for v in grads if v > 0)}, all finite = {finite}")
    ok.append(("gradients", finite and len(grads) > 0 and any(v > 0 for v in grads)))

    print("\n--- assertions ---")
    for name, passed in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    fails = sum(1 for _, p in ok if not p)
    print(("\nALL PASS" if not fails else f"\n{fails} FAILURE(S)"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the hand-written training forward against the real Fast-dLLM-v2 decoder.

Checks, in order:

1. **Parity** — with corruption off, ``blockdiff_logits`` must reproduce the
   logits the actual ``batch_sample`` decoding path produces for the same
   partially-masked block on top of the same clean KV cache.  This is the test
   that says "what we train on is what we run at inference".
2. **Corruption is real** — turning on the KIVI noise surrogate must actually
   move the logits, and only for the blocks that read an old cache.
3. **Top-k is real** — the selector mask must drop old-cache mass.
4. **Gradients** — LoRA parameters receive finite, non-zero gradients through
   the corrupted path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent / "sparse_kv_exp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blockdiff import CorruptionConfig, blockdiff_logits, build_masks  # noqa: E402
from data import FAST_DLLM_MASK_ID, encode_example, load_math_train  # noqa: E402
from model_registry import load_model_and_tokenizer  # noqa: E402


def reference_block_logits(model, ids: torch.Tensor, xt_block: torch.Tensor, block_start: int, block_size: int):
    """Raw (unshifted) logits for one block, via the real inference forward path."""
    with torch.no_grad():
        past = None
        if block_start > 0:
            out = model.forward(
                input_ids=ids[:, :block_start],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            past = out.past_key_values
        out = model.forward(
            input_ids=xt_block,
            use_cache=True,
            past_key_values=past,
            update_past_key_values=False,
            block_size=block_size,
        )
    return out.logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--mask-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model, tokenizer, _ = load_model_and_tokenizer("fast_dllm_v2_7b", device)
    model.eval()
    model.model.bd_size = args.block_size
    model.config.bd_size = args.block_size

    rows = load_math_train(num_examples=8, seed=0, exclude_math500=False)
    ex = None
    for r in rows:
        ex = encode_example(
            tokenizer, r["problem"], r["solution"],
            block_size=args.block_size, max_len=args.max_len, device=device,
        )
        if ex is not None and ex.answer_len > 2 * args.block_size:
            break
    assert ex is not None, "no usable example"

    ids = ex.input_ids.unsqueeze(0)
    labels = ex.labels.unsqueeze(0)
    seq = ids.shape[1]
    n_blocks = seq // args.block_size
    print(f"\nexample: seq={seq} ({n_blocks} blocks) prompt={ex.prompt_len} answer={ex.answer_len}")

    # Pick a block that lies entirely inside the answer and has an old cache.
    answer = labels[0] != -100
    cand = [
        j for j in range(1, n_blocks)
        if bool(answer[j * args.block_size:(j + 1) * args.block_size].all())
    ]
    assert cand, "no fully-answer block"
    j = cand[len(cand) // 2]
    bs = args.block_size
    start = j * bs
    print(f"probing block {j} (positions {start}..{start + bs})")

    # Mask a subset of that block; everything else stays clean.
    g = torch.Generator(device=device).manual_seed(args.seed)
    mask_sel = torch.rand(bs, device=device, generator=g) < args.mask_frac
    xt = ids.clone()
    xt[0, start:start + bs][mask_sel] = FAST_DLLM_MASK_ID

    # ---- 1. parity -------------------------------------------------------
    ref = reference_block_logits(model, ids, xt[:, start:start + bs], start, bs).float()

    doubled = torch.cat([xt, ids], dim=1)
    is_masked = torch.cat([xt[0] == FAST_DLLM_MASK_ID, torch.zeros(seq, dtype=torch.bool, device=device)])
    masks = build_masks(seq, bs, device)
    clean_cfg = CorruptionConfig()
    with torch.no_grad():
        mine = blockdiff_logits(
            model, doubled, masks=masks, cfg=clean_cfg, is_masked_token=is_masked
        ).float()

    # Global position p is predicted from logit p-1; compare the overlapping span.
    a = mine[0, start:start + bs - 1]
    b = ref[0, 0:bs - 1]
    rel = (a - b).abs().max() / b.abs().max()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean()
    kl = F.kl_div(F.log_softmax(a, -1), F.log_softmax(b, -1), log_target=True, reduction="batchmean")
    print("\n[1] parity vs real inference forward")
    print(f"    max|Δ| / max|ref|   = {rel:.3e}")
    print(f"    top-1 agreement     = {agree:.3f}")
    print(f"    KL(mine || ref)     = {kl:.3e} nats")
    ok1 = bool(agree > 0.95 and kl < 1e-2)
    print(f"    => {'PASS' if ok1 else 'FAIL'}")

    # Manual-kernel clean baseline: tests 2/3 must isolate the *corruption*,
    # not the SDPA -> manual kernel switch, so compare against this.
    with torch.no_grad():
        mine_manual = blockdiff_logits(
            model, doubled, masks=masks, cfg=CorruptionConfig(force_manual=True),
            is_masked_token=is_masked,
        ).float()
    kern = (mine_manual - mine).abs().max() / mine.abs().max()
    kern_agree = (mine_manual.argmax(-1) == mine.argmax(-1)).float().mean()
    print(f"\n[1b] manual kernel vs fused SDPA (clean, whole forward)")
    print(f"    max|Δ| / max|logit|  = {kern:.3e}")
    print(f"    top-1 agreement      = {kern_agree:.3f}")

    # ---- 2. corruption actually changes the old-cache reads ---------------
    noisy_cfg = CorruptionConfig(k_bits=4, v_bits=4)
    with torch.no_grad():
        noisy = blockdiff_logits(
            model, doubled, masks=masks, cfg=noisy_cfg, is_masked_token=is_masked
        ).float()
    d_probe = (noisy[0, start:start + bs] - mine_manual[0, start:start + bs]).abs().max()
    d_first = (noisy[0, :bs] - mine_manual[0, :bs]).abs().max()   # block 0 has no old cache
    print("\n[2] KIVI noise surrogate (k4/v4)")
    print(f"    max|Δ| on probed block {j} = {d_probe:.4f}")
    print(f"    max|Δ| on block 0        = {d_first:.4f}  (no old cache -> must be 0)")
    ok2 = bool(d_probe > 1e-3 and d_first < 1e-5)
    print(f"    => {'PASS' if ok2 else 'FAIL'}")

    # ---- 3. top-k selection drops mass -----------------------------------
    topk_cfg = CorruptionConfig(topk=32)
    with torch.no_grad():
        sparse = blockdiff_logits(
            model, doubled, masks=masks, cfg=topk_cfg, is_masked_token=is_masked
        ).float()
    d_sparse = (sparse[0, start:start + bs] - mine_manual[0, start:start + bs]).abs().max()
    d_sparse0 = (sparse[0, :bs] - mine_manual[0, :bs]).abs().max()
    print("\n[3] per-head top-k over the old cache (k=32)")
    print(f"    max|Δ| on probed block {j} = {d_sparse:.4f}")
    print(f"    max|Δ| on block 0        = {d_sparse0:.4f}  (no old cache -> must be 0)")
    ok3 = bool(d_sparse > 1e-3 and d_sparse0 < 1e-5)
    print(f"    => {'PASS' if ok3 else 'FAIL'}")

    # ---- 4. gradients ----------------------------------------------------
    from peft import LoraConfig, get_peft_model

    lcfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    pm = get_peft_model(model, lcfg)
    pm.eval()
    full_cfg = CorruptionConfig(k_bits=4, v_bits=4, topk=32)
    out = blockdiff_logits(
        pm, doubled, masks=masks, cfg=full_cfg, is_masked_token=is_masked,
        gradient_checkpointing=True,
    )
    tgt = labels[0, start + 1:start + bs]
    loss = F.cross_entropy(out[0, start:start + bs - 1].float(), tgt)
    loss.backward()
    grads = [
        (n, p.grad.abs().mean().item())
        for n, p in pm.named_parameters()
        if p.requires_grad and p.grad is not None
    ]
    finite = all(torch.isfinite(torch.tensor(v)) for _, v in grads)
    nonzero = sum(1 for _, v in grads if v > 0)
    print("\n[4] gradients through the corrupted path (LoRA + grad checkpointing)")
    print(f"    loss                = {loss.item():.4f}")
    print(f"    LoRA tensors w/ grad= {len(grads)}, non-zero = {nonzero}, all finite = {finite}")
    ok4 = bool(finite and nonzero > 0 and len(grads) > 0)
    print(f"    => {'PASS' if ok4 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("OVERALL:", "PASS" if all([ok1, ok2, ok3, ok4]) else "FAIL")
    print("=" * 60)
    sys.exit(0 if all([ok1, ok2, ok3, ok4]) else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LoRA recovery training for Fast-dLLM-v2 under a sparse / low-bit KV cache.

Branches (A is the untrained baseline and needs no run here):

  B  --branch sft         plain LoRA SFT on MATH train, exact cache
  C  --branch gkd         on-policy GKD, JSD(beta), student samples densely
  D  --branch sft_noise   SFT + Gaussian KV noise matched to the measured
                          quantization error at the target bit width
  E  --branch gkd_noise   on-policy GKD where the student samples through the
                          *real* sparse+quantized decoding loop, and the loss
                          forward carries the same KV noise and top-k selection

Teacher is the same model with the LoRA adapters disabled (self-distillation),
always dense bf16 with an exact cache.  Student and teacher score the identical
noised sequence, so the JSD is over the same masking pattern.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent), str(_HERE.parent / "sparse_kv_exp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blockdiff import CorruptionConfig, blockdiff_logits, build_masks  # noqa: E402
from data import encode_example, load_math_train, make_noisy_batch  # noqa: E402
from losses import masked_cross_entropy, masked_jsd  # noqa: E402
from model_registry import load_model_and_tokenizer  # noqa: E402
from onpolicy import generate_batch, sparse_gen_config  # noqa: E402

BRANCHES = {
    "sft": dict(loss="ce", onpolicy=False, noise=False, topk=False),
    "gkd": dict(loss="jsd", onpolicy=True, noise=False, topk=False),
    "sft_noise": dict(loss="ce", onpolicy=False, noise=True, topk=False),
    "gkd_noise": dict(loss="jsd", onpolicy=True, noise=True, topk=True),
}

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",       # attention
    "gate_proj", "up_proj", "down_proj",          # MLP
]


def build_corruption(args, spec: dict, *, student_cache: str = "auto",
                     student_topk: str = "auto") -> CorruptionConfig:
    """Corruption applied inside the student's loss forward for this branch.

    ``student_cache``:
      ``exact`` — student reads the true bf16 cache (branches B and, as literally
                  specified, C);
      ``noise`` — Gaussian surrogate at the measured quantization variance (D, E);
      ``quant`` — the real KIVI quantizer with a straight-through estimator.

    NOTE: with ``exact`` + a JSD loss the student and the (adapter-disabled)
    teacher are the *same* function, so the loss is identically zero and nothing
    trains.  See README "Branch C is degenerate".
    """
    if student_cache == "auto":
        student_cache = "noise" if spec["noise"] else "exact"
    use_topk = spec["topk"] if student_topk == "auto" else (student_topk == "on")

    if student_cache == "exact" and not use_topk:
        return CorruptionConfig()

    k_scale = v_scale = 1.0
    if args.noise_calibration:
        cal = json.loads(Path(args.noise_calibration).read_text())
        rec = cal.get("recommended_noise_scale", {}).get(str(args.k_bits))
        if rec:
            k_scale, v_scale = float(rec["k"]), float(rec["v"])
            print(f"[train] calibrated noise scale: k={k_scale:.4f} v={v_scale:.4f}")
        else:
            print(f"[train] WARNING: no calibration entry for {args.k_bits} bits; using 1.0")

    degraded = student_cache in ("noise", "quant")
    return CorruptionConfig(
        mode="quant" if student_cache == "quant" else "noise",
        k_bits=args.k_bits if degraded else None,
        v_bits=args.v_bits if degraded else None,
        kivi_group_size=args.kivi_group_size,
        residual_k_blocks=args.kivi_residual_length // args.block_size,
        residual_v_blocks=0,
        topk=args.topk if use_topk and args.topk_pct is None else None,
        topk_pct=args.topk_pct if use_topk else None,
        propagate_to_x0=args.propagate_cache_errors,
        k_noise_scale=k_scale,
        v_noise_scale=v_scale,
    )


def encode_pairs(tokenizer, pairs, args, device):
    out = []
    for problem, solution in pairs:
        ex = encode_example(
            tokenizer, problem, solution,
            block_size=args.block_size, max_len=args.max_len, device=device,
        )
        if ex is not None and ex.answer_len >= args.min_answer_tokens:
            out.append(ex)
    return out


def forward_loss(
    student, peft_model, ex, args, spec, corrupt_cfg, clean_cfg, generator,
):
    """Loss for one example, summed over the complementary masking pair.

    Rows are run one at a time: the top-k selector is per-sequence, and one
    doubled row already fills the attention score buffers.
    """
    batch = make_noisy_batch(
        ex.input_ids.unsqueeze(0), ex.labels.unsqueeze(0),
        block_size=args.block_size, generator=generator,
        complementary=not args.no_complementary,
    )
    seq = ex.input_ids.shape[0]
    masks = build_masks(
        seq, args.block_size, ex.input_ids.device,
        residual_k_blocks=corrupt_cfg.residual_k_blocks,
        residual_v_blocks=corrupt_cfg.residual_v_blocks,
        propagate_to_x0=corrupt_cfg.propagate_to_x0,
    )

    total = None
    n_tok = 0
    rows = batch.input_ids.shape[0]
    for r in range(rows):
        ids = batch.input_ids[r : r + 1]
        lab = batch.labels[r : r + 1]
        im = batch.is_masked_token[r]
        if int((lab != -100).sum()) == 0:
            continue

        s_logits = blockdiff_logits(
            student, ids, masks=masks, cfg=corrupt_cfg, is_masked_token=im,
            gradient_checkpointing=args.grad_checkpointing, generator=generator,
        )

        if spec["loss"] == "ce":
            loss, n = masked_cross_entropy(
                s_logits, lab, p_mask=batch.p_mask[r : r + 1] if args.p_mask_weighting else None
            )
        else:
            with torch.no_grad(), peft_model.disable_adapter():
                t_logits = blockdiff_logits(
                    student, ids, masks=masks, cfg=clean_cfg, is_masked_token=im,
                )
            loss, n = masked_jsd(s_logits, t_logits, lab, beta=args.beta)
            del t_logits
        del s_logits

        if n == 0:
            continue
        (loss / rows).backward()
        total = loss.detach() if total is None else total + loss.detach()
        n_tok += n
    return (total / max(1, rows)) if total is not None else None, n_tok


@torch.no_grad()
def heldout_ce(student, examples, args, cfg, *, seed: int = 999) -> float:
    """Mean masked CE on held-out MATH problems under the given cache regime.

    The generator is re-seeded on every call so each evaluation sees the *same*
    masking pattern and the same noise draw — otherwise the curve is dominated
    by which tokens happened to be masked.
    """
    generator = torch.Generator(device=examples[0].input_ids.device).manual_seed(seed)
    tot, n = 0.0, 0
    for ex in examples:
        batch = make_noisy_batch(
            ex.input_ids.unsqueeze(0), ex.labels.unsqueeze(0),
            block_size=args.block_size, generator=generator, complementary=False,
        )
        masks = build_masks(
            ex.input_ids.shape[0], args.block_size, ex.input_ids.device,
            residual_k_blocks=cfg.residual_k_blocks,
            residual_v_blocks=cfg.residual_v_blocks,
            propagate_to_x0=cfg.propagate_to_x0,
        )
        logits = blockdiff_logits(
            student, batch.input_ids, masks=masks, cfg=cfg,
            is_masked_token=batch.is_masked_token[0], generator=generator,
        )
        loss, k = masked_cross_entropy(logits, batch.labels)
        if k:
            tot += loss.item() * k
            n += k
    return tot / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=sorted(BRANCHES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--grad-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")

    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--min-answer-tokens", type=int, default=16)
    ap.add_argument("--train-examples", type=int, default=4000)
    ap.add_argument("--heldout-examples", type=int, default=32)
    ap.add_argument("--no-complementary", action="store_true")
    ap.add_argument("--p-mask-weighting", action="store_true")

    # cache regime the student is trained against (branches D, E)
    ap.add_argument("--k-bits", type=int, default=4)
    ap.add_argument("--v-bits", type=int, default=4)
    ap.add_argument("--kivi-group-size", type=int, default=32)
    ap.add_argument("--kivi-residual-length", type=int, default=32)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--topk-pct", type=float, default=None)
    ap.add_argument("--propagate-cache-errors", action="store_true")
    ap.add_argument("--student-cache", choices=("auto", "exact", "noise", "quant"),
                    default="auto",
                    help="what the student's loss forward reads from the old cache")
    ap.add_argument("--student-topk", choices=("auto", "on", "off"), default="auto",
                    help="apply the top-k selector inside the student's loss forward")
    ap.add_argument("--noise-calibration", default=str(_HERE / "noise_calibration.json"))

    # distillation
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lmbda", type=float, default=1.0, help="fraction of on-policy data")
    ap.add_argument("--gen-every", type=int, default=25)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--gen-max-new-tokens", type=int, default=512)

    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()

    spec = BRANCHES[args.branch]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    model, tokenizer, upstream_sample = load_model_and_tokenizer("fast_dllm_v2_7b", device)
    model.eval()                      # keeps the SDPA path; LoRA has no dropout
    model.model.bd_size = args.block_size
    model.config.bd_size = args.block_size

    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
        ),
    )
    peft_model.eval()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()       # fp32 LoRA weights for a stable Adam state
    n_train = sum(p.numel() for p in trainable)
    print(f"[train] branch={args.branch} trainable LoRA params: {n_train/1e6:.1f}M")

    corrupt_cfg = build_corruption(
        args, spec, student_cache=args.student_cache, student_topk=args.student_topk
    )
    clean_cfg = CorruptionConfig()
    if spec["loss"] == "jsd" and not corrupt_cfg.active:
        print(
            "\n[train] WARNING: JSD loss with an exact student cache. The teacher is\n"
            "        this same model with the adapters disabled, so student and teacher\n"
            "        are the identical function and the loss is identically 0 — nothing\n"
            "        will train. Re-run with --student-cache quant (or noise) to make\n"
            "        the student read the degraded cache the teacher does not.\n"
        )
    # The evaluation regime is the same for every branch — noise + top-k at the
    # target width — so the held-out curves of A/B/C/D/E are directly comparable.
    eval_cfg = build_corruption(args, {"noise": True, "topk": True})
    print(f"[train] student cache regime: {asdict(corrupt_cfg)}")
    print(f"[train] held-out eval regime: {asdict(eval_cfg)}")

    rows = load_math_train(num_examples=None, seed=args.seed, exclude_math500=True)
    heldout_rows = rows[: args.heldout_examples]
    train_rows = rows[args.heldout_examples : args.heldout_examples + args.train_examples]
    heldout = encode_pairs(
        tokenizer, [(r["problem"], r["solution"]) for r in heldout_rows], args, device
    )
    train_pool = [(r["problem"], r["solution"]) for r in train_rows]
    train_examples = encode_pairs(tokenizer, train_pool, args, device)
    print(f"[train] usable train examples: {len(train_examples)} / {len(train_pool)}"
          f" (max_len={args.max_len}); held-out: {len(heldout)}")
    if not train_examples:
        raise SystemExit("no training examples fit in --max-len")

    gen_cfg = None
    if spec["onpolicy"] and args.branch == "gkd_noise":
        gen_cfg = sparse_gen_config(
            k_bits=str(args.k_bits), v_bits=str(args.v_bits),
            topk=args.topk, topk_pct=args.topk_pct,
            block_size=args.block_size, max_new_tokens=args.gen_max_new_tokens,
        )
        print(f"[train] on-policy sampling through the sparse loop: {gen_cfg.name}")
    elif spec["onpolicy"]:
        print("[train] on-policy sampling with dense decoding")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    rng = random.Random(args.seed)
    log_path = out_dir / "train_log.jsonl"
    onpolicy_pool: list = []
    t0 = time.time()

    ce0_clean = heldout_ce(peft_model, heldout, args, clean_cfg)
    ce0_corrupt = heldout_ce(peft_model, heldout, args, eval_cfg)
    print(f"[train] step 0 held-out CE  clean={ce0_clean:.4f}  "
          f"under-target-cache={ce0_corrupt:.4f}")

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        if spec["onpolicy"] and (step % args.gen_every == 0):
            probs = [p for p, _ in rng.sample(train_pool, args.gen_batch)]
            gt = time.time()
            samples = generate_batch(
                model, tokenizer, probs,
                exp_cfg=gen_cfg, upstream_batch_sample=upstream_sample,
                block_size=args.block_size, max_new_tokens=args.gen_max_new_tokens,
            )
            onpolicy_pool = encode_pairs(
                tokenizer, [(s.problem, s.completion) for s in samples], args, device
            )
            cov = [s.coverage for s in samples if s.coverage is not None]
            print(f"[train] step {step}: sampled {len(samples)} "
                  f"({len(onpolicy_pool)} usable) in {time.time()-gt:.0f}s"
                  + (f", coverage={sum(cov)/len(cov):.3f}" if cov else ""))

        opt.zero_grad(set_to_none=True)
        step_loss, step_tok, n_ex = 0.0, 0, 0
        for _ in range(args.grad_accum):
            if spec["onpolicy"] and onpolicy_pool and rng.random() < args.lmbda:
                ex = rng.choice(onpolicy_pool)
            else:
                ex = rng.choice(train_examples)
            loss, n = forward_loss(
                model, peft_model, ex, args, spec, corrupt_cfg, clean_cfg, generator
            )
            if loss is None:
                continue
            step_loss += float(loss)
            step_tok += n
            n_ex += 1

        gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if step == 0 and float(gnorm) < 1e-6:
            print(
                f"\n[train] WARNING: gradient norm is {float(gnorm):.2e} at step 0 — this "
                "run has no learning signal.\n        See the branch-C note in README.md.\n"
            )
        opt.step()

        if step % args.log_every == 0:
            rec = {
                "step": step,
                "loss": step_loss / max(1, n_ex),
                "tokens": step_tok,
                "lr": lr_at(step),
                "grad_norm": float(gnorm),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                "elapsed_s": round(time.time() - t0, 1),
            }
            print(f"  step {step:4d}  loss {rec['loss']:.4f}  "
                  f"tok {step_tok:5d}  gnorm {rec['grad_norm']:.3f}  "
                  f"lr {rec['lr']:.2e}  mem {rec['peak_mem_gb']:.1f}G  "
                  f"{rec['elapsed_s']:.0f}s")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        if args.eval_every and (step + 1) % args.eval_every == 0:
            ce_c = heldout_ce(peft_model, heldout, args, clean_cfg)
            ce_q = heldout_ce(peft_model, heldout, args, eval_cfg)
            print(f"  [eval] step {step+1}: held-out CE clean={ce_c:.4f} "
                  f"(start {ce0_clean:.4f})  under-target-cache={ce_q:.4f} "
                  f"(start {ce0_corrupt:.4f})")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": step + 1, "eval_ce_clean": ce_c,
                    "eval_ce_target_cache": ce_q,
                }) + "\n")

        if args.save_every and (step + 1) % args.save_every == 0:
            peft_model.save_pretrained(str(out_dir / "adapter"))

    peft_model.save_pretrained(str(out_dir / "adapter"))
    print(f"[train] done in {time.time()-t0:.0f}s -> {out_dir/'adapter'}")


if __name__ == "__main__":
    main()

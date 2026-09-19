#!/usr/bin/env python3
"""Recovery training: can fine-tuning bring the metric back under a packed cache?

Student: Fast-dLLM-v2-7B, LoRA r=32 on attention *and* MLP, bf16.
Teacher: the same model with the adapters disabled -- frozen, dense bf16, exact
prefix (self-distillation).
Train:   MATH train only. Evaluated on GSM8K/MATH-500 (in-domain) and LongBench
         (out-of-domain; not one long prompt appears in training).

Branches (A is the untrained baseline and needs no run here):

  B  --branch sft         LoRA SFT on MATH train, exact prefix
  C  --branch gkd         on-policy GKD, JSD(beta), student samples densely
  D  --branch sft_noise   SFT + Gaussian KV noise at the measured quantization
                          variance, no selection in the loop
  E  --branch gkd_noise   on-policy GKD where the student samples through the
                          real packed-cache decoder, and the loss forward carries
                          the same degradation and the same top-k selection

The cache regime comes from the *evaluation* config file, so training cannot
drift from what is reported: pass the same --config the eval uses.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bitsieve_fastdllm.config import ExperimentConfig  # noqa: E402
from bitsieve_fastdllm.eval.common import load_fast_dllm  # noqa: E402
from bitsieve_fastdllm.training.blockdiff import blockdiff_logits, build_masks  # noqa: E402
from bitsieve_fastdllm.training.data import (  # noqa: E402
    encode_example,
    load_math_train,
    make_noisy_batch,
)
from bitsieve_fastdllm.training.degrade import CacheDegradation  # noqa: E402
from bitsieve_fastdllm.training.losses import masked_cross_entropy, masked_jsd  # noqa: E402
from bitsieve_fastdllm.training.onpolicy import generate_batch, sampling_config  # noqa: E402

BRANCHES = {
    "sft": dict(loss="ce", onpolicy=False, degrade=False, select=False, dense_sampling=True),
    "gkd": dict(loss="jsd", onpolicy=True, degrade=False, select=False, dense_sampling=True),
    "sft_noise": dict(loss="ce", onpolicy=False, degrade=True, select=False, dense_sampling=True),
    "gkd_noise": dict(loss="jsd", onpolicy=True, degrade=True, select=True, dense_sampling=False),
}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_degradation(cfg: ExperimentConfig, args, *, degrade: bool) -> CacheDegradation:
    """The cache regime the student's loss forward reads, taken from the eval config."""
    if not degrade or args.student_cache == "exact":
        return CacheDegradation(mode="exact")

    k_scale = v_scale = 1.0
    if args.student_cache == "noise" and args.noise_calibration:
        path = Path(args.noise_calibration)
        if path.exists():
            rec = json.loads(path.read_text()).get("noise_scale", {}).get(str(cfg.quant.k_bits))
            if rec:
                k_scale, v_scale = float(rec["k"]), float(rec["v"])
                print(f"[train] calibrated noise scale k={k_scale:.4f} v={v_scale:.4f}")
            else:
                print(f"[train] WARNING: no calibration for {cfg.quant.k_bits} bits; using 1.0")
        else:
            print(f"[train] WARNING: {path} missing; using an uncalibrated noise scale")

    return CacheDegradation(
        mode="noise" if args.student_cache == "noise" else "quant",
        k_bits=cfg.quant.k_bits,
        v_bits=cfg.quant.v_bits,
        key_token_group=cfg.quant.key_token_group,
        value_channel_group=cfg.quant.value_channel_group,
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


def forward_loss(model, peft_model, ex, args, spec, degrade, selector, exact, generator):
    """Loss for one example, summed over the complementary masking pair.

    Rows run one at a time: selection is per-sequence, and a single doubled row
    already fills the attention score buffers.
    """
    batch = make_noisy_batch(
        ex.input_ids.unsqueeze(0), ex.labels.unsqueeze(0),
        block_size=args.block_size, generator=generator,
        complementary=not args.no_complementary,
    )
    masks = build_masks(
        ex.input_ids.shape[0], args.block_size, ex.input_ids.device,
        propagate_to_x0=not args.no_error_compounding,
    )

    total, n_tok, rows = None, 0, batch.input_ids.shape[0]
    for r in range(rows):
        ids = batch.input_ids[r : r + 1]
        lab = batch.labels[r : r + 1]
        if int((lab != -100).sum()) == 0:
            continue
        im = batch.is_masked_token[r]

        s_logits = blockdiff_logits(
            model, ids, masks=masks, degrade=degrade, selector=selector,
            is_masked_token=im, gradient_checkpointing=args.grad_checkpointing,
            generator=generator,
        )
        if spec["loss"] == "ce":
            loss, n = masked_cross_entropy(
                s_logits, lab,
                p_mask=batch.p_mask[r : r + 1] if args.p_mask_weighting else None,
            )
        else:
            with torch.no_grad(), peft_model.disable_adapter():
                t_logits = blockdiff_logits(
                    model, ids, masks=masks, degrade=exact, selector=None,
                    is_masked_token=im,
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
def heldout_ce(model, examples, args, degrade, selector, *, seed: int = 999) -> float:
    """Masked CE on held-out MATH under a given cache regime.

    Re-seeded every call so each evaluation sees the same masking pattern and
    the same noise draw; otherwise the curve is dominated by which tokens
    happened to be masked.
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
            propagate_to_x0=not args.no_error_compounding,
        )
        logits = blockdiff_logits(
            model, batch.input_ids, masks=masks, degrade=degrade, selector=selector,
            is_masked_token=batch.is_masked_token[0], generator=generator,
        )
        loss, k = masked_cross_entropy(logits, batch.labels)
        if k:
            tot += loss.item() * k
            n += k
    return tot / max(1, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=sorted(BRANCHES), required=True)
    ap.add_argument("--config", default="configs/proposed_a_k4v4_p5.yaml",
                    help="the evaluation config whose cache regime training targets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
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

    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--min-answer-tokens", type=int, default=16)
    ap.add_argument("--train-examples", type=int, default=6000)
    ap.add_argument("--heldout-examples", type=int, default=32)
    ap.add_argument("--no-complementary", action="store_true")
    ap.add_argument("--p-mask-weighting", action="store_true")
    ap.add_argument("--no-error-compounding", action="store_true",
                    help="stop degradation propagating into the cache; "
                         "isolates the single-block effect but no longer matches the decoder")

    ap.add_argument("--student-cache", choices=("auto", "exact", "noise", "quant"), default="auto")
    ap.add_argument("--student-select", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--noise-calibration", default="results/training_noise_calibration.json")

    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lmbda", type=float, default=1.0, help="fraction of on-policy data")
    ap.add_argument("--gen-every", type=int, default=25)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--gen-max-new-tokens", type=int, default=768)

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

    cfg = ExperimentConfig.load(args.config)
    args.block_size = cfg.generation.block_size
    print(f"[train] target regime from {args.config}: "
          f"k{cfg.quant.k_bits}v{cfg.quant.v_bits} "
          f"groups {cfg.quant.key_token_group}/{cfg.quant.value_channel_group} "
          f"residual={cfg.quant.residual_tokens} | selector {cfg.selector.mode}/"
          f"{cfg.selector.uniform_queries} topk={cfg.selector.topk} "
          f"pct={cfg.selector.topk_percent}")
    if cfg.quant.residual_tokens:
        print(f"[train] WARNING: residual_tokens={cfg.quant.residual_tokens} keeps a bf16 tail "
              "the training forward does not model; the shipped configs use 0")

    model, tokenizer = load_fast_dllm(args.model, dtype=torch.bfloat16, device=args.device)
    model.eval()                       # keeps the SDPA path; LoRA has no dropout
    model.model.bd_size = args.block_size
    model.config.bd_size = args.block_size
    device = next(model.parameters()).device

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
        p.data = p.data.float()        # fp32 LoRA weights for a stable Adam state
    print(f"[train] branch={args.branch} trainable LoRA params: "
          f"{sum(p.numel() for p in trainable) / 1e6:.1f}M")

    if args.student_cache == "auto":
        args.student_cache = "noise" if spec["degrade"] else "exact"
    use_select = spec["select"] if args.student_select == "auto" else (args.student_select == "on")

    degrade = build_degradation(cfg, args, degrade=args.student_cache != "exact")
    selector = cfg.selector if use_select else None
    exact = CacheDegradation(mode="exact")
    # Every branch is scored under the same regime -- the full target cache --
    # so the held-out curves of A/B/C/D/E are directly comparable.
    eval_degrade = build_degradation(
        cfg, argparse.Namespace(**{**vars(args), "student_cache": "quant"}), degrade=True
    )
    print(f"[train] student loss forward: {asdict(degrade)}, selection={'on' if use_select else 'off'}")

    if spec["loss"] == "jsd" and not degrade.active and not use_select:
        print(
            "\n[train] WARNING: JSD loss with an exact student cache and no selection.\n"
            "        The teacher is this same model with the adapters disabled, so student\n"
            "        and teacher are the identical function, the loss is identically 0 and\n"
            "        nothing will train. Re-run with --student-cache quant to make the\n"
            "        student read the degraded cache the teacher does not.\n"
        )

    rows = load_math_train(limit=None, seed=args.seed, exclude_math500=True)
    heldout = encode_pairs(
        tokenizer, [(r["problem"], r["solution"]) for r in rows[: args.heldout_examples]],
        args, device,
    )
    train_pool = [
        (r["problem"], r["solution"])
        for r in rows[args.heldout_examples : args.heldout_examples + args.train_examples]
    ]
    train_examples = encode_pairs(tokenizer, train_pool, args, device)
    print(f"[train] usable train examples {len(train_examples)}/{len(train_pool)} "
          f"(max_len={args.max_len}); held-out {len(heldout)}")
    if not train_examples:
        print("no training examples fit in --max-len")
        return 1

    gen_cfg = None
    if spec["onpolicy"]:
        gen_cfg = sampling_config(
            cfg, dense=spec["dense_sampling"], max_new_tokens=args.gen_max_new_tokens
        )
        print(f"[train] on-policy sampling: "
              f"{'dense bf16' if spec['dense_sampling'] else 'packed cache + selection'}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    rng = random.Random(args.seed)
    log_path = out_dir / "train_log.jsonl"
    pool: list = []
    t0 = time.time()

    ce0_clean = heldout_ce(peft_model, heldout, args, exact, None)
    ce0_target = heldout_ce(peft_model, heldout, args, eval_degrade, cfg.selector)
    print(f"[train] step 0 held-out CE  exact={ce0_clean:.4f}  target-cache={ce0_target:.4f}")

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        if spec["onpolicy"] and step % args.gen_every == 0:
            probs = [p for p, _ in rng.sample(train_pool, args.gen_batch)]
            gt = time.time()
            samples = generate_batch(model, tokenizer, probs, config=gen_cfg)
            pool = encode_pairs(
                tokenizer, [(s.problem, s.completion) for s in samples], args, device
            )
            sp = sum(s.blocks_sparse or 0 for s in samples)
            by = sum(s.blocks_dense_bypass or 0 for s in samples)
            print(f"[train] step {step}: sampled {len(samples)} ({len(pool)} usable) in "
                  f"{time.time() - gt:.0f}s, blocks sparse={sp} bypass={by}")

        opt.zero_grad(set_to_none=True)
        step_loss, step_tok, n_ex = 0.0, 0, 0
        for _ in range(args.grad_accum):
            if spec["onpolicy"] and pool and rng.random() < args.lmbda:
                ex = rng.choice(pool)
            else:
                ex = rng.choice(train_examples)
            loss, n = forward_loss(
                model, peft_model, ex, args, spec, degrade, selector, exact, generator
            )
            if loss is None:
                continue
            step_loss += float(loss)
            step_tok += n
            n_ex += 1

        gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if step == 0 and float(gnorm) < 1e-6:
            print(f"\n[train] WARNING: gradient norm {float(gnorm):.2e} at step 0 -- "
                  "this run has no learning signal. See docs/recovery_training.md.\n")
        opt.step()

        if step % args.log_every == 0:
            rec = {
                "step": step, "loss": step_loss / max(1, n_ex), "tokens": step_tok,
                "lr": lr_at(step), "grad_norm": float(gnorm),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                "elapsed_s": round(time.time() - t0, 1),
            }
            print(f"  step {step:4d}  loss {rec['loss']:.4f}  tok {step_tok:5d}  "
                  f"gnorm {rec['grad_norm']:.3f}  lr {rec['lr']:.2e}  "
                  f"mem {rec['peak_mem_gb']:.1f}G  {rec['elapsed_s']:.0f}s")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        if args.eval_every and (step + 1) % args.eval_every == 0:
            ce_c = heldout_ce(peft_model, heldout, args, exact, None)
            ce_q = heldout_ce(peft_model, heldout, args, eval_degrade, cfg.selector)
            print(f"  [eval] step {step + 1}: held-out CE exact={ce_c:.4f} "
                  f"(start {ce0_clean:.4f})  target-cache={ce_q:.4f} (start {ce0_target:.4f})")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step": step + 1, "eval_ce_exact": ce_c, "eval_ce_target_cache": ce_q
                }) + "\n")

        if args.save_every and (step + 1) % args.save_every == 0:
            peft_model.save_pretrained(str(out_dir / "adapter"))

    peft_model.save_pretrained(str(out_dir / "adapter"))
    print(f"[train] done in {time.time() - t0:.0f}s -> {out_dir / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

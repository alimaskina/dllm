# Recovery training

Does fine-tuning bring the metric back under a packed cache and a sparse budget?

The training-free configurations lose accuracy on some tasks. This asks whether
a LoRA adapter can recover it, and measures the answer with the *same*
evaluation the rest of the repository reports, so training and eval cannot drift
apart.

- **Student**: Fast-dLLM-v2-7B, LoRA $r=32$ on attention **and** MLP, bf16.
- **Teacher**: the same model with the adapters disabled — frozen, dense bf16,
  exact prefix. Self-distillation, so no second checkpoint is needed.
- **Train**: Hendrycks MATH *train* only, deduplicated against MATH-500.
- **Eval**: GSM8K and MATH-500 (in-domain); LongBench (out-of-domain — not one
  long prompt appears in training).

## Branches

| | `--branch` | student's loss forward reads | loss | on-policy sampling |
|---|---|---|---|---|
| **A** | *(no run)* | — | — | — |
| **B** | `sft` | exact prefix | CE on answers | — |
| **C** | `gkd` | exact prefix | JSD($\beta$) | dense bf16 |
| **D** | `sft_noise` | Gaussian noise at the measured quantization variance | CE on answers | — |
| **E** | `gkd_noise` | same noise **and** the config's top-k selection | JSD($\beta$) | the real packed-cache decoder |

The cache regime is read out of the evaluation config file, not restated:

```bash
python scripts/calibrate_training_noise.py --device cuda:0        # once
python tests/training_parity_e2e.py --device cuda:0               # once

python scripts/train_recovery.py --branch sft_noise \
    --config configs/proposed_a_k4v4_p5.yaml --out runs/D --device cuda:0

python -m bitsieve_fastdllm.eval.quality \
    --config configs/proposed_a_k4v4_p5.yaml --benchmark gsm8k \
    --adapter runs/D/adapter --output results/D_gsm8k.jsonl
```

`--adapter` is available on `eval.quality`, `eval.performance` and
`scripts/run_experiments.py`; the adapter is merged into the base weights before
the runtime patches attention, so decoding sees plain `Linear` layers.
`scripts/run_recovery_matrix.sh` runs the whole matrix.

## Why the training forward is hand-written

Fast-dLLM-v2 offers two forwards and neither is usable. With
`model.training == True` it applies its *own* random masking — teacher and
student would score different noise — and runs `flex_attention` under
`torch.compile(fullgraph=True)`, which cannot be intercepted. With
`model.training == False` it is a block-causal pass over a single sequence,
where block $j$'s context is the *noisy* tokens of blocks $< j$; at inference
those blocks are already decoded and clean.

`training/blockdiff.py` therefore reimplements the doubled $[x_t ; x_0]$
formulation on the model's own submodules. An $x_t$ query in block $j$ attends
to its own noisy block plus the **clean** $x_0$ of blocks $< j$ — exactly the
tokens that live in the packed cache at decode time, which is what makes them
interceptable, and it supervises every block in one forward.

## What makes it the same corruption the decoder produces

Training only means something if the degradation it teaches tolerance for is the
one inference actually applies. Four things are matched, and each is tested
rather than asserted (`tests/test_training_matches_inference.py`,
`tests/training_parity_e2e.py`):

1. **The quantization grid.** `mode="quant"` routes through the repository's own
   `simulate_key_quantization` / `simulate_value_quantization`, which are
   bit-exact with the packed path — keys per channel inside a 32-*token* group,
   values per token inside a 32-*channel* group, scale and zero rounded through
   `param_dtype`. A straight-through estimator carries the gradient; the forward
   value is bit-exactly the quantized one.
2. **No residual.** The shipped configs use `residual_tokens: 0`, so nothing is
   held in bf16 and *every* visible prefix read is degraded. There is no clean
   tail to carve out.
3. **The selection rule.** Per **KV head**, not per query head: importance is
   the softmax mass over the prefix from the selector's query representatives,
   averaged over the $H_q/H_{kv}$ query heads sharing a KV head — this is
   `reference.selector_importance_reference` with `domain="prefix"`, and the
   test compares against that function directly.
4. **Error compounding.** At inference `_sparse_forward` calls `session.attend`
   (a degraded read) *before* `stage_if_committing`, so a block's K/V are
   computed from hidden states that already passed through a degraded prefix.
   $x_0 \to x_0$ reads are degraded too, by default;
   `--no-error-compounding` isolates the single-block effect but no longer
   matches the decoder.

End-to-end against the real 7B checkpoint, with degradation off, the hand-written
forward reproduces the decoding path at top-1 agreement **1.000** and
KL **1.2e-3** nats, and degradation and selection leave block 0 — which has no
prefix — bit-identical.

## The noise surrogate is calibrated, not guessed

Rounding error of an asymmetric $b$-bit quantizer with step $\Delta$ is uniform
on $[-\Delta/2, \Delta/2]$, std $\Delta/\sqrt{12}$. `training/degrade.py`
computes that std over the *same* grouping the quantizer uses, so the noise is
heteroscedastic the same way the real error is.
`scripts/calibrate_training_noise.py` measures the real error on 40 genuine
prefixes (prompt **and** reference solution, since the cache holds decoded
blocks too) and training applies the measured correction:

| bits | tensor | measured std | measured / analytic | kurtosis |
|---|---|---:|---:|---:|
| 4 | K | 0.0774 | **1.085** | 5.09 |
| 4 | V | 0.0742 | **1.031** | 3.04 |
| 2 | K | 0.3911 | 1.097 | 5.05 |
| 2 | V | 0.3730 | 1.037 | 3.02 |

So the surrogate's variance is right to within 3–10 %. The caveat is stated
rather than hidden: the key error is heavy-tailed (kurtosis 5.1 against a
Gaussian's 3.0), so the Gaussian matches the variance but not the tails. Values
are almost exactly Gaussian (3.0). Where the tails matter, `--student-cache
quant` swaps the surrogate for the real quantizer.

## Branch C is degenerate as literally specified

C is "on-policy GKD, no corruption". But the teacher here *is* the student with
its LoRA switched off, and C's student reads the same exact prefix — so the two
are the identical function, $\mathrm{JSD} \equiv 0$, and the gradient norm is
`4e-08` at step 0. Measured, not predicted:

```
[train] WARNING: gradient norm 4.01e-08 at step 0 -- this run has no learning signal.
  step    0  loss -0.0000  tok   738  gnorm 0.000
```

In GKD the student is a *smaller* model, which is where the signal comes from.
Under self-distillation the only thing that can separate student from teacher is
the cache regime. So either keep C as specified and report it as a null cell
(it is then identical to A), or run `--branch gkd --student-cache quant`, where
the student reads the real quantized prefix and the teacher the exact one — the
natural no-noise counterpart of E. `train_recovery.py` warns and flags the zero
gradient rather than silently burning GPU hours.

## Caveats

- **Step 0 of each block is dense at inference** (semantic A schedules selection
  and attends densely). The training forward models the sparse steps, which are
  the majority; it does not reproduce that one dense step per block.
- **Selection is recomputed** inside the loss forward by the same rule, from the
  current noised state — not replayed index-for-index from the selection the
  sampler froze at step 0.
- **The first token of each block** is predicted from the previous block's
  *noisy* tokens during training and from its *decoded* tokens at inference.
  This is upstream's own formulation, kept unchanged.
- **Training sequences are shorter than evaluation prefixes.** MATH solutions
  give prefixes up to ~1.2 k tokens; LongBench prompts run far longer, so a
  percent budget bites differently there. `--gen-max-new-tokens` and `--max-len`
  control the gap for the on-policy branches.
- **A fixed `topk` above the prefix length never engages**, so training against
  such a config degrades nothing — the same trap `tests/smoke_e2e.py` reports.
  Prefer a percent budget, or check `blocks_sparse` in the sampling line.

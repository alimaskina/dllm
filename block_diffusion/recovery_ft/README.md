# Recovery fine-tuning — does training bring the metric back under a sparse / low-bit KV cache?

Question: the training-free sparse + 4-bit KV cache loses accuracy on some
tasks. Can LoRA fine-tuning or self-distillation recover it, and at what cost in
resident memory?

Student: **Fast-dLLM-v2-7B**, LoRA `r=32` on attention **and** MLP, bf16.
Teacher: the **same** model with the adapters disabled — frozen, dense bf16,
exact cache (self-distillation).
Train: MATH train only. Eval: GSM8K + MATH500 (in-domain) and LongBench
2WikiMQA / HotpotQA / MuSiQue / NarrativeQA (out-of-domain — not one long prompt
appears in training).

## Branches

| | `--branch` | student's loss forward reads | loss | on-policy sampling |
|---|---|---|---|---|
| **A** | *(no run)* | — | — | — |
| **B** | `sft` | exact cache | CE on answers | — |
| **C** | `gkd` | exact cache | JSD(β=0.1) | dense decoding |
| **D** | `sft_noise` | Gaussian noise at the measured 4-bit quant variance | CE on answers | — |
| **E** | `gkd_noise` | same noise **+** per-head top-k selection | JSD(β=0.1) | the real sparse+quantized decoding loop |

```bash
# 0. calibrate the noise surrogate against the real quantizer (writes noise_calibration.json)
python calibrate_noise.py --device cuda:0 --num-prompts 40 --bits 4 2

# 1. verify the training forward reproduces the real decoder
python test_parity.py --device cuda:0

# 2. train
python train.py --branch sft_noise --out runs/D --device cuda:0 --steps 500

# 3. evaluate the grid (omit --adapter for branch A)
python eval_recovery.py --branch D_sft_noise --adapter runs/D/adapter \
    --num-examples 60 --device cuda:0 --out results/D
```

`launch_all.sh` runs the whole matrix across GPUs.

---

## Branch C is degenerate as literally specified

C is "on-policy GKD, no corruption". But the teacher here *is* the student with
its LoRA switched off, and the student's loss forward reads the same exact
cache — so student and teacher are the identical function, `JSD ≡ 0`, and the
gradient norm is `0.000` at every step. Measured, not assumed:

```
  step    0  loss -0.0000  tok  1011  gnorm 0.000
  step    2  loss  0.0000  tok   979  gnorm 0.014
```

In the GKD paper the student is a *smaller* model, which is where the signal
comes from. Under self-distillation the only thing that can make the student
differ from the teacher is the cache regime. Two honest options:

* keep C as specified and report it as a null cell (it is then identical to A);
* run `--branch gkd --student-cache quant`, where the student reads the **real**
  KIVI-quantized cache (straight-through estimator) and the teacher the exact
  one. This is the natural no-noise counterpart of E and does produce gradient.

`train.py` prints a warning and flags a zero gradient norm at step 0 rather
than silently burning GPU hours.

A related observation from the smoke runs: with `k4v4` and **no** top-k, the
JSD is ~3e-4 nats — 4-bit quantization alone barely moves this model. Nearly all
of the degradation the eval grid measures comes from the **top-k budget**, not
from the bit width. Branches that omit top-k from the student forward have very
little to learn from.

## What the training forward is, and why it is hand-written

`blockdiff.py` reimplements the doubled `[x_t ; x_0]` block-diffusion forward on
the model's own submodules. The upstream model offers only two paths and neither
is usable here: `model.training=True` applies its *own* random masking (teacher
and student would see different noise) and runs `flex_attention` under
`torch.compile(fullgraph=True)`; `model.training=False` is a plain block-causal
pass over a single sequence, where block *j*'s context would be the *noisy*
tokens of earlier blocks instead of the decoded, clean ones inference sees.

The doubled formulation gives every block supervision in one forward *and*
matches inference: an `x_t` query in block *j* attends to its own noisy block
plus the clean `x_0` of blocks `< j` — exactly the tokens that live in the
quantized cache at decode time, which is what makes them interceptable.

**Corruption geometry.** `block_size == bd_size == KIVI group_size == 32`, so
cache group *i* is exactly block *i*, and with `kivi_residual_length = 32` the
cache block *j* sees has blocks `0..j-2` quantized and block `j-1` still bf16.
Values carry `residual_length = 0` upstream, so every visible old block is
quantized. `build_masks` encodes precisely this.

**`test_parity.py` checks all of it** against the real decoder:

```
[1] parity vs real inference forward
    max|Δ| / max|ref|   = 1.692e-02
    top-1 agreement     = 1.000
    KL(mine || ref)     = 8.429e-04 nats      => PASS
[2] KIVI noise surrogate (k4/v4)
    max|Δ| on probed block 7 = 1.4219
    max|Δ| on block 0        = 0.0000  (no old cache -> must be 0)   => PASS
[3] per-head top-k over the old cache (k=32)
    max|Δ| on probed block 7 = 2.0000
    max|Δ| on block 0        = 0.0000  (no old cache -> must be 0)   => PASS
[4] gradients through the corrupted path (LoRA + grad checkpointing)  => PASS
```

Block 0 has no old cache, so corruption must leave it bit-identical — it does,
which is what proves the corruption lands only on cache reads.

## The noise surrogate is calibrated, not guessed

Rounding error of an asymmetric b-bit quantizer with step `Δ` is uniform on
`[-Δ/2, Δ/2]`, std `Δ/√12`. `quant_noise.py` computes that std from the *same*
grouping the real quantizer uses (K: per-channel inside a 32-token group;
V: per-token over head_dim), so the injected noise is heteroscedastic in the
same way the real error is. `calibrate_noise.py` measures the real error on 40
genuine caches and reports the correction, which training then applies:

| bits | tensor | measured std | measured / analytic | kurtosis |
|---|---|---:|---:|---:|
| 4 | K | 0.0778 | **1.080** | 4.94 |
| 4 | V | 0.0994 | **1.039** | 2.60 |
| 2 | K | 0.3930 | 1.092 | 4.87 |
| 2 | V | 0.4989 | 1.041 | 2.61 |

So the surrogate std is right to within 4–9 %, and `train.py` scales by the
measured ratio. The caveat is honest: the real K error is heavy-tailed
(kurtosis 4.9 against a Gaussian's 3.0), so the Gaussian matches the *variance*
but not the tails. `--student-cache quant` replaces the surrogate with the
actual quantizer (straight-through estimator) when that matters.

## Resident bytes

Reported per cache token, quantization scales and zero-points included.

The sweep grid ranks with an **fp16 selector** over a **4-bit exec** cache. Both
views are read every block — the selector re-ranks at the start of each one — so
both must stay resident, and the 4-bit cache is then *not* a memory saving:

| config | resident / bf16 | coverage |
|---|---:|---:|
| `b4_k64` (fp16 selector, two views) | **1.289** | 0.938 |
| `b4_k64_sel4` (selector reads the quantized cache) | **0.289** | 0.935 |

Ranking on the quantized cache costs ~0.003 coverage and turns a 1.29× memory
*increase* into a 3.5× saving. `eval_recovery.py --selector-precision exec`
runs that variant; the default reproduces the original grid, and the report
marks two-view configurations explicitly.

## Known caveats

* **`k=32/64` on GSM8K is below the problem statement length.** The budget can
  be smaller than the question itself, so part of the drop is the model being
  unable to see the problem, not a selection failure. That makes this a genuine
  test of whether training can compensate — but the cell should not be read as
  "4-bit quantization costs X".
* **Training sequences are shorter than eval caches.** MATH solutions give
  caches up to ~750 tokens; GSM8K decoding reaches 1–2 k. Top-k therefore bites
  harder at eval than in training. Use `--gen-max-new-tokens 1024 --max-len 1280`
  for the on-policy branches to close the gap.
* **The first token of each block** is predicted from the previous block's
  *noisy* tokens during training and from its *decoded* tokens at inference.
  This is upstream's own formulation and is kept unchanged.
* **Top-k inside the loss forward is recomputed**, by the same rule, from the
  current noised state — it is not replayed token-for-token from the selection
  the sampler froze at step 0 of each block.
* **Quantization error does not compound into the cache** by default, matching
  the evaluation harness. `--propagate-cache-errors` turns on the compounding
  (more faithful to a streaming deployment, not comparable to existing numbers).

## Files

| file | |
|---|---|
| `quant_noise.py` | analytic σ of KIVI K/V quantization error |
| `calibrate_noise.py` | measures the real error, validates σ, writes `noise_calibration.json` |
| `blockdiff.py` | doubled block-diffusion forward with cache corruption + top-k |
| `data.py` | MATH train loading, prompt/label encoding, upstream noising |
| `losses.py` | shifted masked CE, generalized JSD(β) |
| `onpolicy.py` | student sampling, dense or through the real sparse loop |
| `train.py` | branches B–E |
| `eval_recovery.py` | the score / coverage / resident-bytes grid |
| `test_parity.py` | the four correctness checks above |

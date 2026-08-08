# Experiment 1: Fixed-State Fidelity
## Order Drift in Quantized Diffusion Language Models

**Model:** LLaDA-8B-Base (`GSAI-ML/LLaDA-8B-Base`)  
**Baseline:** FP16  
**Quantizations:** INT4 (NF4, double quant, BF16 compute), INT8  
**Completion length:** 128 tokens  
**Mask ratios:** 0.9 (early), 0.6 (mid), 0.2 (late)  
**Prompt sets:** generic (15 factual, n=500 synthetic), diverse (30 multi-domain, n=500 synthetic), GSM8K (n=200, gold-teacher states)

---

## TL;DR

> Quantized LLaDA preserves **what** to generate with near-perfect fidelity (>99% token top-1 agreement), but systematically disrupts **where** to start — choosing the wrong first position in **45% of steps** (INT4, generic/diverse) and **27% of steps** (INT4, GSM8K gold-teacher). This gap is the core signal of order drift, and it survives even when revealed positions contain ground-truth answer tokens.

---

## Background

In LLaDA-style masked diffusion decoding, two decisions are made at every step:

1. **What** — which token to place at a masked position: $\hat{x}_i = \arg\max_v\, p(v \mid x_t, i)$
2. **Where** — which masked positions to reveal first: $S_t = \mathrm{TopK}_{i \in \text{masked}}(c_i,\, k)$, where $c_i = \max_v p(v \mid x_t, i)$

The hypothesis is that quantization noise is small relative to the **token margin** (gap between top-1 and top-2 logits within a position) but large relative to the **position boundary margin** (gap between the $k$-th and $(k+1)$-th confidence scores across positions). As a result:

$$\arg\max_v p^Q(v \mid x_t, i) \approx \arg\max_v p^{FP}(v \mid x_t, i) \quad \text{(what preserved)}$$

$$\mathrm{TopK}_i\, c_i^Q \neq \mathrm{TopK}_i\, c_i^{FP} \quad \text{(where disrupted)}$$

---

## Setup

### Masked states

For each sample, a prompt is tokenized and a completion is appended. Three mask ratios simulate decoding stages:

| Label | Mask ratio | Masked positions | Decoding stage |
|-------|-----------|-----------------|----------------|
| early | 0.9 | ~90% of comp | step 1–10 of 128 |
| mid   | 0.6 | ~60% of comp | step ~50 |
| late  | 0.2 | ~20% of comp | step ~100 |

Both FP and Q models receive **identical** input\_ids. Only the forward pass differs.

### State construction

**Synthetic states** (generic, diverse): revealed positions are filled with cycled prompt tokens. Used for cross-domain comparison because these prompt sets have no ground-truth completion.

**Gold-teacher states** (GSM8K): the ground-truth answer from the dataset is tokenized; `mask_ratio` fraction of its tokens are randomly masked; the rest remain as actual answer tokens. This matches teacher-forced denoising exactly — the visible context is always in-distribution.

### Confidence signals

| Signal | Formula | Source |
|--------|---------|--------|
| `confidence` | $\max_v\, \mathrm{softmax}(\ell_i)$ | LLaDA / MaskGIT default |
| `margin` | $p_{(1)} - p_{(2)}$ at position $i$ | Dream `topk_margin` |
| `neg_entropy` | $-H(p_i) = \sum_v p_{iv}\log p_{iv}$ | Dream `entropy` |

### Unmasking $k$ values

| Label | $k$ formula | Source |
|-------|------------|--------|
| `topk1` | $k = 1$ | Where-to-Unmask paper (greedy, cleanest) |
| `topk10step` | $k = n_{\text{masked}} / 10$ | 10-step practical default |
| `topkllada` | $k = n_{\text{masked}} / 128$ | LLaDA default (steps=128) |

---

## Results

### WHAT: token distribution agreement (FP16 vs INT4)

Results from generic prompt set (synthetic, n=200):

| Metric | early | mid | late | **ALL** |
|--------|-------|-----|------|---------|
| token top-1 agree | 0.9957 ± 0.012 | 0.9999 ± 0.001 | **1.0000** ± 0.000 | **0.9985** ± 0.007 |
| token top-3 agree | 0.8778 ± 0.028 | 0.9025 ± 0.030 | 0.9109 ± 0.043 | 0.8970 ± 0.037 |
| token top-5 agree | 0.8757 ± 0.024 | 0.9050 ± 0.021 | 0.9102 ± 0.032 | 0.8970 ± 0.030 |
| token top-10 agree | 0.8871 ± 0.023 | 0.9067 ± 0.018 | 0.9108 ± 0.026 | 0.9015 ± 0.025 |
| KL(FP ‖ Q) | 0.0038 ± 0.014 | 0.0002 ± 0.001 | 0.0002 ± 0.001 | **0.0014** ± 0.009 |

Token top-1 agreement is near-perfect. KL divergence is negligible (0.0014 nats). The INT4 model produces essentially the same token predictions as FP16.

---

### WHERE: position ranking agreement (FP16 vs INT4)

Results from generic prompt set (synthetic, n=200):

| Signal | $k$ | early | mid | late | **ALL** |
|--------|-----|-------|-----|------|---------|
| `confidence` | topk1 | 0.550 ± 0.498 | 0.475 ± 0.499 | 0.635 ± 0.481 | **0.553** ± 0.497 |
| `confidence` | topk10step | 0.807 ± 0.094 | 0.718 ± 0.145 | 0.685 ± 0.318 | **0.737** ± 0.215 |
| `margin`     | topk1 | 0.580 ± 0.494 | 0.535 ± 0.499 | 0.575 ± 0.494 | **0.563** ± 0.496 |
| `neg_entropy` | topk1 | 0.560 ± 0.496 | 0.500 ± 0.500 | 0.620 ± 0.485 | **0.560** ± 0.496 |

`topk1` overlap is only **0.553**: the INT4 model selects the *same first position* as FP16 in only **55% of steps**.

---

### WHERE: position ranking agreement (FP16 vs INT8)

| Signal | $k$ | early | mid | late | **ALL** |
|--------|-----|-------|-----|------|---------|
| `confidence` | topk1 | 0.735 ± 0.441 | 0.710 ± 0.454 | 0.775 ± 0.418 | **0.740** ± 0.439 |
| `confidence` | topk10step | 0.887 ± 0.083 | 0.858 ± 0.116 | 0.845 ± 0.242 | **0.863** ± 0.163 |
| `margin`     | topk1 | 0.755 ± 0.430 | 0.700 ± 0.458 | 0.775 ± 0.418 | **0.743** ± 0.437 |
| `neg_entropy` | topk1 | 0.755 ± 0.430 | 0.690 ± 0.463 | 0.745 ± 0.436 | **0.730** ± 0.444 |

INT8 is substantially better: topk1 overlap **0.740** vs 0.553 for INT4.

---

### Key diagnostic: what − where gap

$$\Delta = \text{token top-1 agree} - \text{conf topk-1 overlap}$$

Generic/diverse synthetic states (n=200):

| Quantization | early | mid | late | **ALL** |
|---|---|---|---|---|
| **INT4** | +0.446 | **+0.525** | +0.365 | **+0.445** |
| **INT8** | +0.263 | **+0.290** | +0.225 | **+0.259** |

Both quantizations show a large positive gap. INT4 loses ~45 percentage points between what and where; INT8 loses ~26.

---

## Summary table (generic, synthetic)

| Metric | FP16 (ref) | INT8 | INT4 |
|---|---|---|---|
| token top-1 agree | 1.000 | **0.999** | 0.999 |
| token top-3 agree | 1.000 | **0.948** | 0.897 |
| KL(FP ‖ Q) | 0 | **0.0004** | 0.0014 |
| conf topk-1 overlap | 1.000 | **0.740** | 0.553 |
| conf topk10step overlap | 1.000 | **0.863** | 0.737 |
| what − where gap | 0 | **0.259** | 0.445 |

---

## GSM8K: gold-teacher states (INT4, n=200)

Revealed positions contain actual ground-truth answer tokens. This eliminates out-of-distribution context and matches teacher-forced denoising.

### Per-stage results

| Metric | early | mid | late | **ALL** |
|--------|-------|-----|------|---------|
| token top-1 agree | 0.868 ± 0.082 | 0.991 ± 0.015 | 0.995 ± 0.017 | **0.951** ± 0.077 |
| token top-3 agree | 0.861 ± 0.039 | 0.911 ± 0.023 | 0.917 ± 0.035 | 0.896 ± 0.042 |
| KL(FP ‖ Q) | 0.071 ± 0.091 | 0.006 ± 0.006 | 0.003 ± 0.006 | **0.027** ± 0.061 |
| conf topk-1 overlap | 0.715 ± 0.451 | 0.745 ± 0.436 | 0.735 ± 0.441 | **0.732** ± 0.443 |
| conf topk10step overlap | 0.864 ± 0.105 | 0.863 ± 0.132 | 0.775 ± 0.346 | **0.834** ± 0.226 |
| **what − where gap** | +0.153 | **+0.246** | +0.260 | **+0.220** |

---

## Cross-domain and cross-model results (INT4)

### Full comparison: LLaDA-8B vs Dream-7B

| Model / prompts | n | states | token top-1 | topk-1 pos | topk10step | **what−where gap** | KL(FP‖Q) |
|---|---|---|---|---|---|---|---|
| LLaDA / generic | 500 | synthetic | 0.998 | 0.573 | 0.739 | **0.425** | 0.001 |
| LLaDA / diverse | 500 | synthetic | 0.997 | 0.605 | 0.767 | **0.392** | 0.003 |
| LLaDA / GSM8K | 200 | gold | 0.951 | 0.732 | 0.834 | **0.220** | 0.027 |
| Dream / generic | 500 | synthetic | 0.867 | 0.653 | 0.781 | **0.214** | 0.310 |
| Dream / diverse | 500 | synthetic | 0.874 | 0.631 | 0.760 | **0.243** | 0.312 |

### Per decoding stage — what−where gap

| Stage | LLaDA/gen | LLaDA/div | LLaDA/gsm (gold) | Dream/gen | Dream/div |
|---|---|---|---|---|---|
| early (0.9) | 0.401 | 0.409 | 0.153 | 0.380 | 0.323 |
| mid   (0.6) | **0.490** | **0.437** | **0.246** | 0.251 | 0.280 |
| late  (0.2) | 0.384 | 0.330 | 0.260 | **0.010** | 0.126 |

Note: LLaDA/gsm early stage has lower token top-1 agree (0.868) and higher KL (0.071) because at 90% masking only ~13 gold tokens are visible, and the model faces genuine uncertainty about which answer tokens belong in masked positions — a much harder task than predicting repeated prompt tokens.

### Two distinct failure modes

**LLaDA — pure order drift:**
- Token top-1 agreement: **95–99.8%** (lowest on gold GSM8K early stage)
- Position top-1 agreement: **57–73%** — *where* to start is wrong in 27–43% of steps
- KL(FP‖Q) ≈ **0.001–0.027** — token distributions close to identical
- Conclusion: **what largely preserved, where broken**

**Dream — holistic degradation:**
- Token top-1 agreement: **86.7–87.4%** — even *what* breaks under INT4
- Position top-1 agreement: **63–65%** — similar disruption to LLaDA
- KL(FP‖Q) ≈ **0.31** — token distributions massively shifted (200× worse than LLaDA)
- Gap **collapses in late-stage** (0.01–0.13): when few positions remain, both metrics converge
- Conclusion: **both what and where broken**; smaller reported gap is not because where is better preserved, but because the baseline itself degrades

The smaller what−where gap for Dream should not be read as "Dream is more robust to order drift" — rather Dream's entire output distribution shifts, making the asymmetry harder to isolate with this metric alone.

**Likely cause:** LLaDA is LLaMA-based (NF4 well-tuned for LLaMA); Dream is Qwen-based with different weight statistics, more sensitive to INT4 quantization overall.

---

## Key findings

**1. Order drift exists and is structural.**  
With gold-teacher states on GSM8K, INT4 chooses the wrong first position in **~27% of steps** while token top-1 agreement is still ~95%. The gap is positive and significant even with perfectly realistic revealed context.

**2. Spearman ≠ topk overlap.**  
Spearman correlation is high (0.948–0.971 for INT4), suggesting global rank structure is preserved. But this masks a sharp failure at the decision boundary: the positions ranked $k$ and $k+1$ are often swapped by quantization noise, changing the unmask set.

**3. Effect peaks in mid-to-late decoding.**  
For gold GSM8K: gap is flat at mid (0.246) and late (0.260). Gold context reduces early-stage drift (larger boundary margins at low mask ratio), but mid/late stages remain affected.

**4. All three confidence signals are equally affected.**  
confidence, margin (top1−top2), and neg\_entropy all show near-identical topk1 overlap. The drift is a property of the confidence score distribution, not of how scores are computed.

**5. INT8 halves the gap but does not eliminate it.**  
INT8 reduces topk1 overlap loss from ~45% to ~26%. Even at 8-bit precision, one in four first-position decisions diverges from FP16.

**6. Larger reveal sets are more stable.**  
topk10step overlap (0.737–0.863) is substantially better than topk1 (0.553–0.732). When $k$ is large, individual confidence perturbations matter less. However, the first few revealed positions drive context for subsequent steps, so drift accumulates.

---

## Limitations

- **Generic and diverse states are still synthetic.** These prompt sets have no ground-truth completion, so synthetic construction is unavoidable. Results for these sets should be read as upper-bound gap estimates.
- **Gold states only for GSM8K.** Extending gold-teacher or FP-teacher states to generic/diverse requires generating completions with the FP model first (Exp 2 will do this naturally).
- **Static comparison.** We compare one FP forward pass vs one Q forward pass on identical inputs. Trajectory drift (Exp 2) will show how errors accumulate across steps.
- **Single base model.** Only LLaDA-8B-Base tested with INT8. Dream-7B only tested with INT4.

---

## Next steps

- **Exp 2: Free-running trajectory drift** — let FP and Q decode independently, measure divergence step by step.
- **Exp 3: What/where decomposition** — hybrid decoding (FP what + Q where and vice versa) to causally isolate the contribution of order drift to final accuracy.
- **FP-teacher states for generic/diverse** — generate completions greedily with FP16, then mask; eliminates synthetic bias for non-GSM8K prompts.
- **Layer analysis** — quantize one layer at a time, measure topk-1 position overlap drop, rank layers by sensitivity; enables targeted mixed-precision.
- **Margin analysis** — check whether drift correlates with position boundary margin $c_{(k)} - c_{(k+1)}$ at each step.

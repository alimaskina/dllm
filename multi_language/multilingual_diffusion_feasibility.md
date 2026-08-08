# Multilingual Diffusion: feasibility plan

## Core hypothesis

Different languages induce different token-fragmentation patterns. Under token-wise diffusion training, highly fragmented lexical units may receive a systematically different denoising curriculum: the model often reconstructs parts of a word with sibling subtokens visible, while states where the whole multi-token unit is unresolved inside otherwise clean context are rare.

At inference, confidence/margin decoding may create exactly these low-noise “unresolved lexical island” states. If so:

**fragmentation → objective mismatch → inference OOD states → errors**

A strong project requires evidence for every link.

## 1. Check that the setup is real

### Tokenization audit
For EN + several structurally different languages (e.g. DE/RU/TR/FI/ZH/KO), measure:
- distribution of tokens per linguistic word: `P(k=1), P(k=2), P(k=3), P(k>=4)`;
- control for character length and word frequency;
- optionally content/function words and morphological complexity.

Goal: verify that useful languages actually have substantially different fragmentation under the model tokenizer.

### Language competence
Use a multilingual multiple-choice task with output only `A/B/C/D`, plus a small free-generation sanity check.

Compare where possible:
- Dream ↔ Qwen2.5;
- Fast-dLLM v2 ↔ Qwen2.5;
- LLaDA as independent replication.

Goal: distinguish “model does not know the language” from “generation/denoising is bad in that language”.

## 2. Audit the training objective

For the exact corruption/loss used by each model, estimate per language:

- fraction of target-token loss where all sibling subtokens of its word are also masked;
- fraction where at least one sibling is visible;
- distribution of noise level `t` conditional on the whole lexical unit being unresolved;
- exposure to fully unresolved multi-token words at low noise.

For standard independent masking, a `k`-token word is fully masked with probability `t^k`, so low-noise whole-word holes become rapidly rarer as `k` grows.

Important: run this for the real objectives (e.g. LLaDA-style masking, Dream/CART), not just an abstract MDLM.

Goal: establish whether different languages really receive measurably different lexical denoising curricula.

## 3. Check train–inference mismatch directly

Run **oracle reverse trajectories** on full target sentences:
- scheduler chooses which positions to reveal normally;
- revealed values are replaced by gold tokens, so errors never accumulate.

Compare schedulers:
- random (negative control);
- confidence;
- margin.

At each global mask ratio `t`, measure:

`P_traj(whole word unresolved | k, t)`

against the training expectation:

`P_train(whole word unresolved | k, t)`.

Key statistic:

`Mismatch(k,t) = P_traj / P_train`.

Goal: test whether realistic decoding leaves difficult multi-token lexical units unresolved much more often than the training corruption would.

## 4. Check whether the mismatch is harmful

On the same oracle trajectories, measure:
- gold NLL on remaining masked tokens;
- whole-word/group NLL;
- sentence-level remaining-mask NLL / expected errors.

Control for:
- global mask ratio;
- token count `k`;
- frequency;
- character length;
- language.

Goal: see whether states that are most underrepresented during training are also unusually hard for the model.

This is evidence, not yet causality.

## 5. Independent diffusion-specific fragmentation check

Use a controlled task with random output codebooks:
- same underlying decision;
- answer represented by 1, 2, or 4 model tokens;
- many random codebooks.

Compare DLM vs matched AR model.

Goal: test whether diffusion has extra sensitivity to output segmentation beyond ordinary AR tokenization effects.

Keep this as supporting evidence, not the main result.

## 6. Check whether decoding policy interacts with fragmentation

At fixed NFE/reveal budget, compare confidence, margin, random, L2R.

Test:
- `Policy × token_count`;
- then `Policy × Language` after controlling for token_count/fragmentation.

Interpretation:
- only `Policy × fragmentation` remains → languages probably do not need separate policies; local structure explains the difference;
- `Policy × Language` remains → investigate genuine language/morphology-specific decoding effects.

## 7. Causal test: does fixing the objective help?

Only if steps 1–4 show a clear signal.

Train small matched DLMs with identical initialization/data/compute/tokenizer:

1. **Independent token masking** — standard baseline.
2. **Lexical/group-correlated masking** — whole lexical units are masked together while keeping the same marginal mask rate.
3. **Random correlated groups** — same group-size distribution, but boundaries are not linguistically aligned.

The third model controls for the possibility that generic span correlation, rather than lexical structure, is what helps.

Evaluate on:
- full-sentence denoising with several whole units masked;
- multiword infilling;
- full multilingual generation/translation.

Strong prediction:

languages with larger measured objective mismatch should benefit more from the corrected objective.

## 8. Full vs block diffusion

Do not make this part of the first proof-of-concept, but replicate the mechanism later in:
- full-sequence diffusion;
- block diffusion.

Question: does the clean causal prefix in block diffusion reduce the fragmentation-induced mismatch, or does the same problem remain inside each block?

## Go / no-go criteria

The story is promising only if most of this chain holds:

1. languages have meaningfully different fragmentation;
2. the objective converts that into different denoising exposure;
3. realistic inference creates lexical mask states that training under-samples;
4. those states are genuinely harder;
5. the effect is stronger in DLMs than in matched AR models;
6. changing the corruption process reduces the problem and improves non-toy generation.

If one link fails, revise the mechanism instead of adding more downstream benchmarks.

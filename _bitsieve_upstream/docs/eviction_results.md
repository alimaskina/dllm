# KV-cache eviction: results

Selection decides what a block *reads*; eviction decides what the cache *keeps*.
Only the second saves memory. This measures what it costs in quality.

## What was run

Two models, both block-diffusion decoders, MATH-500, n=60, greedy, block size 32,
4096-token budget. Selector: `uniform`-5 representatives, **fixed** top-512
prefix budget — a percent budget on a ~100-token math prompt collapses to ~20
tokens and hides the question from the model, which reads as a quality collapse
unrelated to eviction (see `scripts/run_suite.py`).

| | Fast-dLLM-v2-7B | DreamReasoner-8B |
|---|---|---|
| layers × KV heads | 28 × 4 | 36 × 8 |
| unmasking threshold | 0.95 | 0.9 |
| transformers | 4.53.1 | 5.x (its remote code requires it) |
| coverage measured | yes | no (skipped) |

Eviction state, per (layer, KV head) per live entry: `g` (EMA, fp32),
`n` (age in blocks, int16), `pos` (int32) = 10 B. At step 0 of each block the
selector's own α updates `g ← λg + (1−λ)α`, `n += 1`, `ĝ = g/(1−λⁿ)`. The
bias correction is not optional: without it a fresh entry receiving identical
attention reads as >5× worse purely for having been observed once.

Every 4 blocks, if live > C:

- **`recent`** — keep the C newest by position.
- **`ema+recent`** — keep the W=128 newest, plus the top C−W by `ĝ` among the
  rest (the window is excluded from that contest, so C slots hold C entries).

λ = 0.9, W = 128, interval = 4 blocks throughout.

## Fast-dLLM-v2

`C = max(5%·S, 256)`. Median context 576 tokens, so the floor always won and
C was 256 everywhere.

| arm | score | coverage | B/entry |
|---|---:|---:|---:|
| dense bf16 (ceiling) | 0.667 | — | — |
| bf16, no eviction | 0.683 | 0.988 | — |
| bf16 `recent` | 0.500 | 0.627 | 522 |
| bf16 `ema+recent` | **0.650** | 0.968 | 522 |
| k4v4, no eviction | 0.667 | 0.981 | — |
| k4v4 `recent` | 0.433 | 0.659 | 170 |
| k4v4 `ema+recent` | **0.617** | 0.965 | 186 |

Paired sign tests over the same 60 problems:

| comparison | wins/losses | p |
|---|---:|---:|
| `ema` vs `recent`, bf16 | 10 / 1 | **0.012** |
| `ema` vs `recent`, k4v4 | 12 / 1 | **0.003** |
| no eviction vs `ema`, bf16 | 5 / 3 | 0.73 |
| no eviction vs `ema`, k4v4 | 5 / 2 | 0.45 |
| selection vs dense | 3 / 2 | 1.00 |
| k4v4 vs bf16 | 4 / 5 | 1.00 |

`ema+recent` beats `recent` decisively and is indistinguishable from keeping
everything, at ~45% of the cache. Coverage explains the mechanism directly:
0.97 vs 0.63 — recency alone discards the entries that carry the attention mass.

Selection at top-512 and 4-bit quantization are both free here.

## DreamReasoner-8B

Median context 640 tokens with a tail to 4320, so `max(5%·S, 256)` again
collapsed to 256 — and at that capacity only **7% of the cache survives on the
longest quarter**. Six capacity points were swept to separate "the policy is
wrong" from "the budget is too tight".

Scores, policy `ema+recent`; memory is the summed resident bytes over all 60
requests, with the share of the unevicted run in brackets.

| capacity | bf16 score | bf16 memory | k4v4 score | k4v4 memory |
|---|---:|---:|---:|---:|
| no eviction | 0.700 | 11106 MB | 0.733 | 2949 MB |
| C = 256 | 0.600 | 2546 (0.23) | 0.600 | 960 (0.33) |
| C = 512 | 0.700 | 4391 (0.40) | 0.683 | 1473 (0.50) |
| max(25%·S, 256) | 0.650 | 4344 (0.39) | 0.567 | 1447 (0.49) |
| max(25%·S, 512) | 0.700 | 4807 (0.43) | 0.700 | 1663 (0.56) |
| max(50%·S, 256) | 0.700 | 6542 (0.59) | **0.717** | 2125 (0.72) |
| max(50%·S, 512) | **0.767** | 6074 (0.55) | 0.717 | 2117 (0.72) |

`recent` scored lower than `ema+recent` in **12 cells out of 12** — both widths,
all six capacities, no exception. Individually only bf16 `max(50%·S, 512)`
reaches significance (8/0, p = 0.008); unanimity across twelve is itself
p ≈ 2e-4.

Against keeping everything, `ema+recent` is significantly worse only in the two
harshest cells (k4v4 C=256, p = 0.039; k4v4 max(25%·S, 256), p = 0.006).
`recent` is significantly worse in six of twelve and worse in direction in all
of them.

## Analysis

**The floor matters more than the percent.** `max(25%·S, 256)` and
`max(25%·S, 512)` give the same capacity on long problems and differ only on the
median — 0.567 vs 0.700 on k4v4. Cutting short contexts is what hurts; long ones
tolerate it.

**Eviction stops hurting at roughly half the cache.** At C=256 (7% surviving on
the longest quarter) both policies break and the choice between them stops
mattering — which is why the first Dream run showed no policy difference and the
capacity sweep was needed to see one.

**Quantization metadata costs more than the policy does.** One k4v4 entry is
160 B, of which **20% is scale/zero** (16 B for keys, 16 B for values); the
policy state is 10 B, +6%. Key scales amortize over a 32-token group, value
scales are paid per token. At 4 bits the payload shrank 8× and the fp16 metadata
did not, which is why its share is so visible. On bf16 there is no metadata at
all — 512 B payload, +2% state.

**A scattered survivor set is not free.** `ema+recent` costs 172 B/entry against
`recent`'s 160 B on k4v4: it keeps a window plus a scatter, leaving 2444 key
groups alive against 1271 for the same number of entries, each still carrying
its scale vector. On bf16 both cost identically, which is where the quality
comparison is clean. Selecting whole 32-token groups instead of individual
entries would recover that 8% — an untested third policy.

**On bf16, `max(50%·S, 512)` scored 0.767 against 0.700 for the full cache** —
dropping 45% of the cache beat keeping it. Plausibly attention dilution on a
4000-token context, but 0/4 at p = 0.125 is a hint, not a result.

## Caveats

- Coverage was measured on Fast-dLLM only. On Dream the mechanism is supported
  by score alone.
- One seed, n = 60. Individual cells sit near the significance boundary; the
  12/12 pattern is the load-bearing evidence, not any single cell.
- `max(5%·S, 256)` from the original specification never engaged its percent
  term at these lengths — the floor decided every capacity. Adaptivity was only
  tested by the swept variants.
- The bf16 result above the full-cache ceiling needs a larger n before it can
  be claimed.

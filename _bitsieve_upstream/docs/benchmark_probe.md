# Which benchmark should the eviction study use?

MATH-500 answers the eviction question badly: the median context is 576-640
tokens, so `C = max(5%·S, 256)` collapses to its floor and the policy barely
engages. The search was for a benchmark with two properties **at once** — the
model reasons long enough that the cache actually grows, and the metric is far
enough from saturation that eviction can move it.

Both properties turn out to be properties of the *decoder*, not the benchmark.
The same benchmark is long on one model and short on the other.

## Method

`scripts/probe_benchmarks.py` runs the same examples under two generation
ceilings, 2048 and 8192. The comparison separates three cases that look alike in
a single-budget run:

- score rises with the ceiling → the extra tokens buy reasoning
- score flat, every run hits the ceiling → the model loops; length is not thought
- score flat, runs stop on their own → the ceiling was never the constraint

Ten examples per cell, greedy, block size 32. Confidence threshold 0.95 on
Fast-dLLM and 0.9 on Dream (each model's own default). We also record how many
generations contain a `\boxed{}` at all, because a grader reading a format the
model does not emit measures the grader rather than the model.

## Results

| model | benchmark | ceiling | score | median context | hit the ceiling | had `\boxed` |
|---|---|---:|---:|---:|---:|---:|
| Dream | AIME-2024 | 2048 | 0.20 | 2186 | 10/10 | — |
| Dream | AIME-2024 | 8192 | **0.50** | 8330 | 10/10 | 7/10 |
| Dream | AIME-2025 | 2048 | 0.40 | 2186 | 10/10 | — |
| Dream | AIME-2025 | 8192 | **0.70** | 8330 | 10/10 | 9/10 |
| Dream | GPQA-Diamond | 2048 | 0.00 | 2227 | 10/10 | — |
| Dream | GPQA-Diamond | 8192 | 0.20 | 8371 | 10/10 | 9/10 |
| Dream | MMLU-Pro | 2048 | 0.50 | 2215 | 10/10 | — |
| Dream | MMLU-Pro | 8192 | 0.50 | 8359 | 10/10 | 10/10 |
| Fast-dLLM | AIME-2024 | 2048 | 0.20 | 1244 | 3/10 | 7/10 |
| Fast-dLLM | AIME-2024 | 8192 | 0.20 | 1244 | 3/10 | 7/10 |
| Fast-dLLM | MMLU-Pro | 2048 | 0.70 | 613 | 0/10 | 1/10 |
| Fast-dLLM | MMLU-Pro | 8192 | 0.70 | 613 | 0/10 | 1/10 |

### Dream reasons long everywhere; only AIME rewards it

Dream spends the entire budget on all four benchmarks at both ceilings. On AIME
that buys something: +0.30 on 2024, +0.30 on 2025. On MMLU-Pro six thousand
extra tokens buy exactly nothing, and GPQA moves by one example out of ten.

So for DreamReasoner, **AIME is the benchmark to use** — long by construction,
and the metric still has room in both directions. GPQA is long but too hard to
be a sensitive instrument at this size; MMLU-Pro is long and flat.

### Fast-dLLM does not reason long at all

Seven of ten AIME generations stop **on their own** between 600 and 1900 tokens,
byte-identically at both ceilings:

```
ceiling 2048:  600  836  912 1050 1180 1307 1886 | 2145 2177 2177
ceiling 8192:  600  836  912 1050 1180 1307 1886 | 8289 8321 8321
```

The three that do not stop consume whatever ceiling they are given and produce
no additional correct answers (0.20 either way). That is looping, not thinking.

This matters for the eviction study directly: at a median AIME context of 1244
tokens, `max(5%·S, 256)` gives C = 256 and the seven well-behaved examples never
reach it. **Long-output benchmarks cannot make Fast-dLLM's cache grow.** For
that model the cache only gets large from a long *input*, so eviction has to be
measured on LongBench-style prompts instead.

## What is deliberately missing

**OlympiadBench.** Scored a flat 0.00 on every run. Its gold answers are
symbolic LaTeX (`$\frac{1}{2n+2}$`, `$\binom{2n}{n}$`) and a string-normalising
grader cannot match them regardless of what the model writes. Usable only behind
a sympy equivalence check, so it is not wired in.

## Provenance and one known grading defect

The table was produced by the predecessor of `scripts/probe_benchmarks.py` —
same prompts, same models, same ceilings, same graders except for one bug found
afterwards and fixed in the committed version.

The bug: the multiple-choice fallback (used only when a generation has no
`\boxed{}`) searched for the **first** `answer|option` in the text. Models
enumerate the choices on the way to a verdict ("Option A. ..."), so the rule
routinely read the enumeration instead of the conclusion and scored correct
answers as misses. Both fallbacks now read from the end of the generation;
`tests/test_probe_grading.py` pins this on the real generations that exposed it.

Which cells this touches, given the `\boxed` column:

- **Fast-dLLM MMLU-Pro** — only 1 of 10 generations had a box, so nine were
  graded by the fallback. The table already shows the corrected 0.70; the
  pre-fix probe reported 0.50. Re-checked against the stored generations, and
  the committed script reproduces the corrected verdicts.
- **Dream GPQA-Diamond** — 9 of 10 had a box, so one example was graded by the
  fallback and that cell carries up to ±0.10 of uncertainty.
- Everything else is unaffected: Dream MMLU-Pro graded 10/10 from `\boxed`, and
  the AIME cells use the numeric grader, which did not change.

To regenerate the whole table under the committed code (Fast-dLLM needs
transformers 4.53, Dream needs 5.x, so they run in separate environments):

```bash
PYTHONPATH=src python scripts/probe_benchmarks.py run --model fastdllm \
    --device cuda:0 --limit 10 --max-new-tokens 2048 --threshold 0.95 --out results/benchprobe
PYTHONPATH=src python scripts/probe_benchmarks.py run --model dream \
    --device cuda:1 --limit 10 --max-new-tokens 8192 --threshold 0.9  --out results/benchprobe
PYTHONPATH=src python scripts/probe_benchmarks.py report --out results/benchprobe
```

Ten examples per cell is a probe, not a measurement: it is enough to tell "the
budget changes the score" from "it does not", and not enough to rank models.

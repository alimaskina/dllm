# Running the official 200-example evaluation

This repository currently reports **60 examples per task**. The official THUDM/LongBench
protocol evaluates the **full 200-example split**, and the published per-model scores everyone
compares against are computed on all 200. Everything else in our harness already follows the
official protocol (verified below); the sample size is the one remaining deviation.

This document is the complete procedure for closing it. Nothing else needs to change.

---

## What to run

Three tasks, four variants each, 200 examples per task:

| task | metric | generation budget | why it is in the table |
|---|---|---|---|
| `gov_report` | ROUGE-L | 512 | continuous metric, all examples differ, largest effects |
| `qasper` | QA-F1 | 128 | different task type, continuous metric |
| `trec` | classification (exact) | 64 | discrete metric, few but unanimous differences |

Variants (these names appear in every output file):

| variant | what it is |
|---|---|
| `dense` | exact bf16 KV cache, no selection — the ceiling |
| `sparse_fp16_all` | selection ranked by ALL masked block queries, fp16 cache |
| `sparse_k4v4_all` | same selection, 4-bit key / 4-bit value cache |
| `sparse_fp16_middle` | selection ranked by a SINGLE middle query, fp16 cache |

Budget: 5% of the live prefix (the default, `TOPK_PCT=5.0`). Do not change it — the reported
numbers are at this budget.

---

## Setup

```bash
git clone <this repo> && cd <repo>
git checkout bitsieve-fastdllm-fixes      # the results below are on this branch
bash scripts/setup.sh --require-gpu       # creates the `bitsieve` conda env
conda activate bitsieve
python -m pytest tests/ -q                # 83 tests, all should pass
```

Requirements: Linux, one or two NVIDIA GPUs with **≥ 40 GB** free each (the model is a 7B in
bf16 and prompts reach 32k tokens), CUDA 11.8+ driver, ~60 GB disk for the model and dataset
caches.

The model and dataset are pulled from HuggingFace on first use and cached:

- model `Efficient-Large-Model/Fast_dLLM_v2_7B`, revision `0661abf5f9f0ee338970d091052a26c8efa51974`
  (pinned by default — do not override, the numbers are revision-specific)
- dataset `zai-org/LongBench`, file `data.zip`

If the machine has no internet, download `data.zip` elsewhere and set
`LONG_BENCH_ARCHIVE_PATH=/path/to/data.zip`.

---

## The commands

Two GPUs (recommended — total wall time is set by `gov_report`):

```bash
# GPU 0: the long one, ~4.5 h
PYTHON=python DEVICE=cuda:0 N=200 TASKS=gov_report \
  OUTPUT_ROOT=results/task_selection \
  bash scripts/run_task_selection.sh

# GPU 1, in a second shell: ~1 h
PYTHON=python DEVICE=cuda:1 N=200 TASKS=qasper,trec \
  OUTPUT_ROOT=results/task_selection \
  bash scripts/run_task_selection.sh
```

One GPU (~6 h, sequential):

```bash
PYTHON=python DEVICE=cuda:0 N=200 TASKS=gov_report,qasper,trec \
  OUTPUT_ROOT=results/task_selection \
  bash scripts/run_task_selection.sh
```

Use `nohup`/`tmux` — these run for hours:

```bash
setsid nohup bash -c 'PYTHON=python DEVICE=cuda:0 N=200 TASKS=gov_report \
  OUTPUT_ROOT=results/task_selection bash scripts/run_task_selection.sh' \
  > run_gov.log 2>&1 < /dev/null &
```

**`N=200` exactly.** The manifest guard permits raising the example count on an existing run but
refuses to lower it below what the manifest records, so a smaller `N` on `gov_report`/`qasper`
aborts with a message about the count shrinking. That guard is what makes the next section safe:
it is what stops a resume from silently mixing runs that are not comparable.

---

## Resuming (this matters — the run is already 30% done)

`results/task_selection/{gov_report,qasper,trec}/` already contain **60 finished examples per
variant** (`qasper`/`dense` has 63), committed to the branch. They were produced by this same code, model revision and
budget, so the commands above **continue from them** and compute only the missing 140. The log
says so on start:

```
extending the existing run in results/task_selection/gov_report: longbench_n 60 -> 200
  [dense/quality] gov_report: running 140/200 (max_new_tokens=512)
```

If you see `running 200/200` instead, the existing rows were not picked up — stop and check that
you are on branch `bitsieve-fastdllm-fixes` with `results/` intact, rather than burning 6 GPU-hours
recomputing them.

The same applies after an interruption: re-run the identical command and it picks up where it
stopped. Rows are flushed to disk after every example, so at most one example is ever lost.

A run that dies mid-example leaves a truncated final JSONL line; the loader drops it with a
warning. That is expected, not corruption.

**Before the run, `qasper` reports verdict `incomplete`** — its `dense` arm has 63 rows against 60
for the other three, from an interrupted start. Those rows are valid and are kept so the 200-run
does not recompute them; the verdict resolves by itself once all four arms reach 200. Do not
delete them, and do not read that `incomplete` as a problem.

---

## Reading the results

```bash
python scripts/rank_task_orderings.py results/task_selection
```

The three tasks should print verdict `ok` and a block like:

```
gov_report   200   0.340   0.290   0.288   0.269   0.10-0.37   ok
             differing examples: dense-mage 200/200, k4v4-mage 200/200, mage-herald 200/200
             sign test (wins:losses): dense-mage 163:37 p=0.000, ...
```

**Read the sign test, not the mean.** On a discrete metric a mean can rest on one flipped example,
and on `trec` the mean halved between n=20 and n=60 while the sign result held. The
`differing examples` line marks with `!` any comparison resting on ≤2 examples.

### What the numbers should look like

The n=60 results on this branch, for comparison. The n=200 run should land near these; the
`dense-mage` and `mage-herald` signs should stay significant and `k4v4-mage` should stay a
coin flip.

| task | dense | mage | k4v4 | herald | dense−mage | k4v4−mage | mage−herald |
|---|---|---|---|---|---|---|---|
| gov_report | 0.340 | 0.290 | 0.288 | 0.269 | 49:11 p<.001 | 29:31 p=.897 | 44:16 p<.001 |
| qasper | 0.368 | 0.296 | 0.300 | 0.244 | 26:9 p=.006 | 16:12 p=.572 | 21:9 p=.043 |
| trec | 0.683 | 0.667 | 0.650 | 0.550 | 2:1 p=1.00 | 0:1 p=1.00 | 7:0 p=.016 |

Sanity bands from LongBench's own results table (2023-era 7B models, so ours sitting at or a
little above the top is expected for a 2024-generation base — this is a check against a broken
protocol, not a target): gov_report 0.10–0.37, qasper 0.17–0.43, trec 0.52–0.79.

**If a dense score lands far outside its band, stop and report it** rather than treating the run as
finished — it means something in the protocol broke, not that the model is unusual.

### What to send back

- `results/task_selection/{gov_report,qasper,trec}/` (the `.jsonl` files — they contain every
  prediction, so anything can be recomputed without re-running the model)
- the full output of `python scripts/rank_task_orderings.py results/task_selection`
- the run logs

---

## What is already official (no action needed)

Verified against THUDM/LongBench's `pred.py` and `eval.py`:

| item | status |
|---|---|
| prompt templates | verbatim from `dataset2prompt.json` for all three tasks |
| generation budgets | from `dataset2maxlen.json` — 512 / 128 / 64 |
| metrics | from `dataset2metric` — `rouge_score` / `qa_f1_score` / `classification_score` |
| first-line-only scoring | applied to `trec` only, as upstream does |
| chat template | applied to `gov_report`/`qasper`, skipped for `trec` — upstream's exemption list |
| truncation | from the middle, `first = max//2`, `last = max - first`, as `pred.py` does |
| context limit | 32768, the model's own; only the 2 longest `gov_report` examples hit it |

Two deviations remain and are deliberate:

1. Truncation is applied **after** the chat template rather than before. Worth ~30 tokens out of
   32768 and it keeps the generation prompt intact, which a block-diffusion model needs.
2. The model is not in the LongBench results table, so the published bands are a
   generation-gap sanity check rather than a comparison target.

---

## Cost

Measured on one A100-class GPU, for the 140 new examples × 4 variants:

| task | s/example | remaining |
|---|---|---|
| gov_report | 28.0 | ~4.4 h |
| qasper | 2.5 | ~0.4 h |
| trec | 3.1 | ~0.5 h |

~5.3 GPU-hours total; ~4.5 h wall on two GPUs. `gov_report` dominates because it generates 512
tokens per example against an 11k-token mean prompt.

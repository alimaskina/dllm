# Running the recovery study

One command trains every branch and produces the grid.

```bash
conda activate bitsieve          # or whatever scripts/setup.sh created
bash scripts/run_recovery_matrix.sh
```

It finds the visible GPUs by itself, deals nine jobs across them, and runs one
7B model per card at a time. To see exactly what it would do without running
anything:

```bash
PLAN_ONLY=1 bash scripts/run_recovery_matrix.sh
```

On the first run it downloads the checkpoint, the Hendrycks MATH train split
and the LongBench archive from Hugging Face, so the machine needs network access
once. Nothing has to be staged by hand. Set `LONG_BENCH_ARCHIVE_PATH` to use a
local copy of the LongBench zip instead.

---

## The question

The training-free packed cache loses accuracy at tight budgets. Can a LoRA
adapter get it back, does an adapter trained at one budget help at a *tighter*
one, and does an adapter trained on some long-context tasks help on tasks it
never saw?

## What runs

Nine jobs. Each is evaluated over the same 18-cell grid, so every question above
is answered from one table.

| job | training data | training budget | loss |
|---|---|---|---|
| `A_none` | — | — | the training-free baseline |
| `M_sft` | MATH train | — | CE on answers, exact prefix |
| `M_gkd` | MATH train | — | JSD(0.1) vs the adapter-disabled teacher |
| `M_sft_noise` | MATH train | — | CE, prefix degraded by calibrated KV noise |
| `M_gkd_noise` | MATH train | `topk=32` | JSD(0.1), degraded prefix **and** top-k selection |
| `L_*` | LongBench `2wikimqa`+`hotpotqa` | `2.5%` | the same four |

Selection inside the training loop belongs to branch E by design ("no selection
in the loop" for D), so `--student-select` defaults to each branch's own
setting. `STUDENT_SELECT=on` puts every branch under the budget instead.

## The grid

18 cells per job:

| benchmark | cells |
|---|---|
| GSM8K, MATH-500 | dense bf16 ceiling, `b4_k32`, `b4_k16` |
| 2WikiMQA, HotpotQA, MuSiQue, NarrativeQA | dense bf16 ceiling, `b4_p2p5`, `b4_p1` |

Each cell reports **score**, **coverage** (the absolute share of the fp16
reference attention mass the selected prefix entries carry) and **resident cache
bytes** with its ratio to bf16 — a quality number is not readable without what
it cost.

Two generalization axes fall out of this:

- **budget** — the math track trains at `k=32` and is scored at 32 *and* 16;
  the LongBench track trains at `2.5%` and is scored at 2.5% *and* 1%.
- **task** — `2wikimqa` and `hotpotqa` are trained on and scored from
  `--example-offset` onward, so never on a training example; `musique` and
  `narrativeqa` are never trained on at all.

LongBench ships **200 examples per task and no train split**, so the same 200
have to cover both ends. The defaults train on the first 100 (plus 8 held out
for the training curve) and evaluate from example 120, leaving 80 — enough for
`LIMIT_LONGBENCH=60`. The script checks this arithmetic against the real split
size *before* touching a GPU, because getting it wrong is silent until the first
trained LongBench cell hours later, and an offset that is too small would
quietly score the adapter on its own training examples.

The offset is applied to **every** job, not just the ones that trained on those
tasks. It costs nothing for `A_none` or the math track, and it means every row
of the table is scored on the same examples — otherwise the trained and
untrained rows would be comparing different subsets.

## Before the branches run

The script refuses to train until two things pass, because a silent failure here
would invalidate everything downstream:

1. `scripts/calibrate_training_noise.py` measures the packed cache's real
   rounding error, so branch D/E noise carries the measured variance rather
   than a guess. Cached in `results/training_noise_calibration.json`.
2. `tests/training_parity_e2e.py` checks the hand-written training forward
   against the real decoder. It must print `ALL PASS`; expect top-1 agreement
   `1.000` and KL `~1e-3` nats on both the short- and long-context paths.

## Cost

Measured on an A100-80GB, per job:

| benchmark | wall clock at 60 examples |
|---|---:|
| GSM8K | ~49 min |
| MATH-500 | ~49 min |
| 2WikiMQA / HotpotQA / MuSiQue | ~24-30 min each |
| NarrativeQA | ~156 min |

≈ **5.6 h of evaluation per job**, plus ~1.5 h of training (2.5 h on the
LongBench track). Nine jobs ≈ **64 GPU-hours**, so ~8 h on eight cards.
NarrativeQA alone is 46 % of it — `LIMIT_NARRATIVEQA=20` cuts the total by
about a third.

## Knobs

| variable | default | |
|---|---|---|
| `DEVICES` | all visible | e.g. `cuda:0,cuda:1` |
| `LIMIT` / `LIMIT_LONGBENCH` / `LIMIT_NARRATIVEQA` | 60 | examples per cell |
| `LONGBENCH_PER_TASK` / `EXAMPLE_OFFSET` | 100 / 120 | the train/eval split of each task's 200 |
| `STEPS` | 500 | training steps per branch |
| `TRACKS` | `math longbench` | drop one to halve the run |
| `BRANCHES` | all four | |
| `GKD_STUDENT_CACHE` | `quant` | `exact` reproduces the null cell on purpose |
| `OUT` / `RESULTS` | `runs/recovery` / `results/recovery` | |

## Resuming

Everything is idempotent: an adapter that already exists is not retrained, and a
grid cell whose `.jsonl` already exists is not re-evaluated. An interrupted run
is restarted with the same command.

## Reading the result

```bash
python scripts/run_recovery_grid.py report --out results/recovery
```

Also written to `results/recovery/report.txt` at the end of the run.

Two things to check before reading anything into the numbers:

- **Branch C.** With `GKD_STUDENT_CACHE=exact` the teacher is this same model
  with its adapters disabled and the student reads the same exact prefix, so the
  JSD is identically zero and nothing trains — but float noise still drifts the
  adapter, so C will differ from A without having learned anything. The gradient
  norm in `train_log.jsonl` is what separates drift from signal. The default
  `quant` avoids this.
- **A fixed budget above the prefix never engages.** `topk=32` bites on GSM8K's
  ~100-400-token prefixes; a budget larger than the whole prefix silently runs
  dense. `blocks_sparse` in each row's `runtime` confirms selection actually ran.

Everything the training forward matches against the decoder, the noise
calibration numbers and the known caveats are in
[recovery training](recovery_training.md).

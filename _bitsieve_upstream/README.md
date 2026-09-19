# BitSieve for Fast-dLLM v2

Packed low-bit KV caching and one-shot sparse selection for block-diffusion inference.
See [methodology and evaluation](docs/methodology.md) for the algorithm, kernel implementation, and evaluation protocol.

## Repository

```text
configs/                   Seven static method configurations
scripts/setup.sh           Conda environment setup
scripts/run_experiments.py Quality, performance, memory, and summary commands
src/bitsieve_fastdllm/     Cache, kernels, runtime, and evaluation modules
docs/methodology.md        Methodology and evaluation reference
results/                   Results and logs
```

## Installation

Use Linux, Conda, an NVIDIA GPU with compute capability 8.0 or newer, and a compatible NVIDIA driver.

From the repository root:

```bash
bash scripts/setup.sh --env bitsieve --require-gpu
conda activate bitsieve
```

The setup defaults are Python 3.11, PyTorch 2.7.1, Transformers 4.53.1, and NumPy<2. Other dependencies are specified in `pyproject.toml`. The script chooses a CUDA wheel family from the visible driver, or defaults to `cu126` when the driver is hidden. You can also omit `--require-gpu` and choose the wheel explicitly:

```bash
bash scripts/setup.sh --env bitsieve --pytorch-cuda cu128
```

The default cache directory is `~/.cache/bitsieve-fastdllm`, which can be changed with `--cache-root`:
```bash
bash scripts/setup.sh --env bitsieve --cache-root /path/to/bitsieve-cache
```

## Configurations

| Config name | Persistent K/V | Selector | Sparse budget |
|---|---|---|---:|
| `official_dense_bf16` | 16-bit native cache | None | Full prefix |
| `dense_kivi4_k4v4_r0` | Packed 4/4-bit, no residual | None | Full prefix |
| `dense_kivi2_k2v2_r0` | Packed 2/2-bit, no residual | None | Full prefix |
| `mage_bf16_all_a_k512` | 16-bit cache | All available masked positions | 512 |
| `herald_middle_bf16_a` | 16-bit cache | One central masked position | 512 |
| `proposed_a_k4v4_k512` | Packed 4/4-bit, no residual | Uniform-5 | 512 |
| `proposed_a_k2v2_k512` | Packed 2/2-bit, no residual | Uniform-5 | 512 |
| `proposed_a_k4v4_p5` | Packed 4/4-bit, no residual | Uniform-5 | 5% of prefix |
| `proposed_a_k2v2_p5` | Packed 2/2-bit, no residual | Uniform-5 | 5% of prefix |

These are MAGE-like and HERALD-middle-proxy baselines, not the original full implementations. Configs can be added/modified in `configs/` folder.

### Nothing is held in full precision

Every quantized config uses `residual_tokens: 0`: the whole prefix is packed, so
no part of the cache sits in bf16 and the memory numbers need no footnote. Every
selector config uses `dense_prefix_layers: 0`, so no layer is exempted from
selection. Both were non-zero in an earlier revision, which meant the reported
compression and the selector's measured cost each excluded a piece of the model.

Neither costs speed. Against the earlier settings at an identical budget, under
the fixed schedule used for work-normalised timing, time per output block is
2.1% lower and the resident cache 12.4% smaller; the selector kernel alone is
7-15% faster, because the former bf16 tail moves out of a `torch.matmul`
side-pass into the packed Triton kernel. `tests/test_residual0_kernels.py` runs
with `BITSIEVE_CUDA_STRICT=1`, so an unavailable fast path raises instead of
silently falling back to the slow reference.

### A fixed budget does not always engage

`effective_topk` is `min(topk, prefix_len)`, so a `k512` config attends densely
until the prefix passes 512 entries - selecting 512 of 400 *is* dense attention.
On short-prompt math the prefix often never gets there: one GSM8K example with a
114-token prompt finishes with a 416-token cache, so `proposed_a_k4v4_k512`
runs exactly like `dense_kivi4_k4v4_r0` and measures nothing about selection.

Two things follow. Every run now reports `blocks_sparse`,
`blocks_dense_bypass`, `sparse_block_fraction`, and `sparse_layer_step_fraction`,
and the summary tables carry `mean_sparse_block_fraction` - a config that never
took the sparse path cannot be read as evidence about the sparse path. And the
`_p5` configs size the budget as a percentage of the live prefix, so selection
engages from the first block regardless of prompt length; prefer them whenever
the claim is about the selector rather than about a specific k.

### Honest coverage

`coverage_diagnostics: true` scores each selection against the attention it is
meant to approximate: the reference top-k is always computed from **exact
bf16/fp16 keys ranked by every masked query in the block**, independent of the
query subset (`selector.mode`) and of the key precision the config under test
uses. A starved or quantized selector therefore cannot grade its own homework -
which it would if the reference reused its own scores. Reported as
`coverage.mass_mean` (share of the reference mass captured, normalised by what
the best selection at that budget captures, so 1.0 means "as good as possible
here") and `coverage.overlap_mean` (`|selected ∩ reference top-k| / k`).
Dense-bypassed layers are excluded rather than counted as a free 1.0.

This keeps a shadow fp16 key cache and recomputes reference attention, so it
costs memory and time: it is off by default and must stay off for performance
and memory runs. The packed kernels are untouched by it.

## One-button suite (quality + coverage + speed, four variants)

```bash
bash scripts/run_suite.sh
```

Runs 15 GSM8K + 10 LongBench examples (5 each of `qmsum`/`repobench-p`
by default) through four variants at a shared 5%-of-prefix budget:

| variant | cache | selector |
|---|---|---|
| `dense` | 16-bit, full attention | none (ceiling) |
| `sparse_fp16_all` | 16-bit | all masked queries (MAGE-style) |
| `sparse_fp16_middle` | 16-bit | one center query (HERALD-proxy) |
| `sparse_k4v4_all` | KIVI K4/V4 | all masked queries |

Every selector variant runs twice: a **quality pass** (`coverage_diagnostics`
off) that quality and speed are read from, and a separate **coverage pass**
(diagnostics on) that only measures coverage - the diagnostic shadow key cache
must never be the thing tokens/s gets measured against. `dense` has no
selector, so it runs once and reports no coverage.

The printed table carries `sparse_frac` (median `sparse_block_fraction`) on
every selector row; anything under `1.00` means the budget failed to engage on
some examples - the run warns about this explicitly, because it is the exact
way an earlier revision's `k512` configs measured nothing about selection on
short prompts (see above). Percent-of-prefix budgets keep this at `1.00` here.

Try `python scripts/run_suite.py --smoke` first (~2 minutes, one example per
benchmark) to confirm the environment works before committing to the full run,
which is on the order of an hour on one A100. Everything is configurable via
env vars (`bash scripts/run_suite.sh --help` after setting `PYTHON=`, or read
the flags in `scripts/run_suite.py`) - GPU, model revision, example counts,
which LongBench tasks, which variants, the budget percentage. Re-running with
the same `--output-root` resumes from what is already in its JSONL files.

`PYTHON` must point at an interpreter with this project's dependencies (the env
`scripts/setup.sh` creates, or your own with `transformers==4.53.1` - anything
newer fails inside the model's remote code with an unrelated-looking
`KeyError` during RoPE init). The script checks this and fails with a clear
message before attempting to load the model if it does not hold.

## Quality evaluation

To reproduce the headline table at the **official 200-example split** (this repository ships
60-example runs), follow [docs/RUN_OFFICIAL_200.md](docs/RUN_OFFICIAL_200.md). It resumes from the
committed results rather than recomputing them, and costs ~5 GPU-hours.

### Independent quality check

For a directly comparable quality run, use the pinned one-button harness:

```bash
bash scripts/run_quality_repro.sh
```

The default button evaluates 20 examples each from GSM8K, QMSum, and
RepoBench-P across the four suite variants under two profiles:
`Long=5% / math k=64` and `Long=20% / math k=128`. It runs the official task
prompts and metrics, saves every prediction/reference pair, and writes a
manifest for each profile with the model revision, dataset revisions, example
IDs, and prompt fingerprints. The quality pass intentionally skips coverage
diagnostics, so its timing and scores are not changed by audit-only work.

Useful overrides are environment variables, for example:

```bash
DEVICE=cuda:1 VARIANTS=dense,sparse_k4v4_all bash scripts/run_quality_repro.sh
```

Re-running with the same `OUTPUT_ROOT` resumes completed examples. Compare
`summary.json` and `manifest.json`; a different manifest means the runs are
not a like-for-like comparison.

Each worker processes one request at a time; multiple GPUs can run independent jobs.

```bash
python scripts/run_experiments.py quality \
  --gpus 1,4 \
  --configs official_dense_bf16,dense_kivi4_k4v4_r0,dense_kivi2_k2v2_r0,mage_bf16_all_a_k512,herald_middle_bf16_a,proposed_a_k4v4_k512,proposed_a_k2v2_k512,proposed_a_k4v4_p5,proposed_a_k2v2_p5 \
  --benchmarks gsm8k,hotpotqa,narrativeqa,qasper,qmsum,lcc,repobench-p,math500,musique \
  --output-root results/quality_main
```

The runner sets output budgets in memory: 2,048 tokens for GSM8K and MATH-500 (a math answer truncated mid-chain never emits its `\boxed{...}`, and the grader then falls back to the last number in the text, scoring by accident), 64 for NIAH, and 512 for the other datasets. The same table is used by a direct `python -m bitsieve_fastdllm.eval.quality` call, so a config default cannot silently shorten it. `--max-new-tokens` overrides that choice for every requested benchmark. `--max-cache-tokens` defaults to 32,768.

Supported dataset names are `gsm8k`, `math500`, `hotpotqa`, `narrativeqa`, `qasper`, `qmsum`, `lcc`, `repobench-p`, `triviaqa`, `2wikimqa`, `musique`, and `niah`. LongBench tasks refer to the LongBench subsets, not the original full standalone datasets.

## Fixed-work performance

Each `(method, context, batch)` runs in a new process. Stop-token termination is disabled, and every output block uses the requested fixed denoising schedule. Tokens/s is aggregate throughput across the request batch; TPOB is time per block round, not divided by batch size.

```bash
python scripts/run_experiments.py performance \
  --gpus 1,4 \
  --configs official_dense_bf16,dense_kivi4_k4v4_r0,dense_kivi2_k2v2_r0,mage_bf16_all_a_k512,herald_middle_bf16_a,proposed_a_k4v4_k512,proposed_a_k2v2_k512,proposed_a_k4v4_p5,proposed_a_k2v2_p5 \
  --contexts 512,2048,8192,16384,28672 \
  --batch-sizes 1,4,8,16 \
  --max-new-tokens 128 \
  --fixed-steps 20 \
  --warmup 2 \
  --repeats 3 \
  --capacity fixed \
  --max-cache-tokens 32768 \
  --output-root results/performance
```

`--capacity fixed` retains a common physical cache capacity. `--capacity workload` sets capacity to `context + output budget`, rounded up to the block size. Do not combine the two policies as one memory experiment.

## Memory and capacity

The `memory` action uses workload-sized cache capacity by default:

```bash
python scripts/run_experiments.py memory \
  --gpus 1,4 \
  --configs official_dense_bf16,dense_kivi4_k4v4_r0,dense_kivi2_k2v2_r0,mage_bf16_all_a_k512,herald_middle_bf16_a,proposed_a_k4v4_k512,proposed_a_k2v2_k512,proposed_a_k4v4_p5,proposed_a_k2v2_p5 \
  --contexts 28672 \
  --batch-sizes 1,4,8,16,32 \
  --max-new-tokens 128 \
  --warmup 1 \
  --repeats 2 \
  --output-root results/memory
```

The existing runtime fields `peak_cuda_allocated_bytes` and `peak_cuda_reserved_bytes` reset after prefill. They are decode-phase metrics, not end-to-end peaks.
One failed job does not stop other jobs, and all requested batch sizes are attempted. OOM and other errors are recorded separately for new runs.

## Recovery training

Can fine-tuning bring the metric back under a packed cache and a sparse budget?
`scripts/train_recovery.py` trains a LoRA adapter against the cache regime a
given evaluation config describes, and `--adapter` on the evaluation entry
points measures it with the same harness everything else is reported with.

```bash
python scripts/calibrate_training_noise.py --device cuda:0     # once
python tests/training_parity_e2e.py --device cuda:0            # once

python scripts/train_recovery.py --branch sft_noise \
    --config configs/proposed_a_k4v4_p5.yaml --out runs/D --device cuda:0

python -m bitsieve_fastdllm.eval.quality \
    --config configs/proposed_a_k4v4_p5.yaml --benchmark gsm8k \
    --adapter runs/D/adapter --output results/D_gsm8k.jsonl
```

`DEVICES=cuda:0,cuda:1 bash scripts/run_recovery_matrix.sh` runs the whole
matrix, one model per card at a time. Branches, what is matched against the
decoder and how it is checked, the noise calibration, and the caveats are in
[recovery training](docs/recovery_training.md).

## GPUs, logging, and resuming

`--gpus` accepts CUDA device selectors passed verbatim into each child's `CUDA_VISIBLE_DEVICES`, for example `1,4` or GPU UUIDs.

Completed jobs are skipped when their status, source/settings fingerprint, and output checksum match. Interrupted quality jobs resume from compatible saved examples through the existing evaluator.

Status and execution metadata are stored beside each result as `.jsonl.status.json`.

## Summaries

```bash
python scripts/run_experiments.py summarize \
  --input results/quality_main \
  --input results/quality_extra \
  --output results/quality_tables

python scripts/run_experiments.py summarize \
  --input results/performance \
  --input results/memory \
  --output results/systems_tables
```

The command produces Markdown/CSV tables for quality, performance/memory, and failures when those result types are present. 

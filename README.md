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
| `dense_kivi4_k4v4_r32` | Packed 4/4-bit, residual 32 | None | Full prefix |
| `dense_kivi2_k2v2_r32` | Packed 2/2-bit, residual 32 | None | Full prefix |
| `mage_bf16_all_a_k512` | 16-bit cache | All available masked positions | 512 |
| `herald_middle_bf16_a` | 16-bit cache | One central masked position | 512 |
| `proposed_a_k4v4_k512` | Packed 4/4-bit, residual 32 | Uniform-5 | 512 |
| `proposed_a_k2v2_k512` | Packed 2/2-bit, residual 32 | Uniform-5 | 512 |

These are MAGE-like and HERALD-middle-proxy baselines, not the original full implementations. Configs can be added/modified in `configs/` folder.

## Quality evaluation

Each worker processes one request at a time; multiple GPUs can run independent jobs.

```bash
python scripts/run_experiments.py quality \
  --gpus 1,4 \
  --configs official_dense_bf16,dense_kivi4_k4v4_r32,dense_kivi2_k2v2_r32,mage_bf16_all_a_k512,herald_middle_bf16_a,proposed_a_k4v4_k512,proposed_a_k2v2_k512 \
  --benchmarks gsm8k,hotpotqa,narrativeqa,qasper,qmsum,lcc,repobench-p,math500,2wikimqa,musique \
  --output-root results/quality_main
```

The runner sets output budgets in memory: 1,024 tokens for MATH-500, 64 for NIAH, and 512 for the other datasets. `--max-new-tokens` overrides that choice for every requested benchmark. `--max-cache-tokens` defaults to 32,768.

Supported dataset names are `gsm8k`, `math500`, `hotpotqa`, `narrativeqa`, `qasper`, `qmsum`, `lcc`, `repobench-p`, `triviaqa`, `2wikimqa`, `musique`, and `niah`. LongBench tasks refer to the LongBench subsets, not the original full standalone datasets.

## Fixed-work performance

Each `(method, context, batch)` runs in a new process. Stop-token termination is disabled, and every output block uses the requested fixed denoising schedule. Tokens/s is aggregate throughput across the request batch; TPOB is time per block round, not divided by batch size.

```bash
python scripts/run_experiments.py performance \
  --gpus 1,4 \
  --configs official_dense_bf16,dense_kivi4_k4v4_r32,dense_kivi2_k2v2_r32,mage_bf16_all_a_k512,herald_middle_bf16_a,proposed_a_k4v4_k512,proposed_a_k2v2_k512 \
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
  --configs official_dense_bf16,dense_kivi4_k4v4_r32,dense_kivi2_k2v2_r32,mage_bf16_all_a_k512,herald_middle_bf16_a,proposed_a_k4v4_k512,proposed_a_k2v2_k512 \
  --contexts 28672 \
  --batch-sizes 1,4,8,16,32 \
  --max-new-tokens 128 \
  --warmup 1 \
  --repeats 2 \
  --output-root results/memory
```

The existing runtime fields `peak_cuda_allocated_bytes` and `peak_cuda_reserved_bytes` reset after prefill. They are decode-phase metrics, not end-to-end peaks.
One failed job does not stop other jobs, and all requested batch sizes are attempted. OOM and other errors are recorded separately for new runs.

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

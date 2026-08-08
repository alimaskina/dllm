# GSM8K quality: FP16 vs INT4 LLaDA-8B-Base

Reproduces the official LLaDA evaluation setup from [EVAL.md](https://github.com/ML-GSAI/LLaDA/blob/main/EVAL.md).

**Where/what unmask analysis (Sudoku, WikiText, GSM8K):** see [sudoku4_fp16_int4_analytics.md](sudoku4_fp16_int4_analytics.md) and [wikitext_g64_multitoken_report.md](wikitext_g64_multitoken_report.md).

## Multi-token word filter (WikiText)

All WikiText multi-token word experiments use **strict lexical** words via `multitoken_word_filters.is_lexical`:

- whitespace `\S+` span has **≥2 tokenizer tokens**
- alphabetic core (≥3 chars), no infobox junk (`target:`, `image;`, …)
- **no glued punctuation** — rejects `present.`, `height;`, `"The`, `(born`
- **no contraction fragments** — rejects standalone `n't`, `'s`, …

Legacy loose tier (`is_lexical_loose`, allowed word+punctuation): pass `--word-tier lexical_loose`.

```bash
# Default in analyze scripts: --word-tier lexical
python analyze_multitoken_words.py --checkpoint checkpoints/results_wikitext_fp16_g64_n256 --report
python generate_wikitext_report.py --out wikitext_g64_multitoken_report.md
```

## Environment

Same as `quant_unmask`:

```bash
conda activate llada_quant
# torch 2.5.1+cu124, transformers 4.46.2, accelerate 0.34.2, bitsandbytes 0.44.1
```

## Settings

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | `GSAI-ML/LLaDA-8B-Base` | EVAL.md |
| Task | `gsm8k` via lm-eval | eval_llada_lm_eval.sh |
| num_fewshot | **5** (default from `gsm8k.yaml`, not overridden in shell script) | lm_eval/tasks/gsm8k/gsm8k.yaml |
| gen_length | 1024 | EVAL.md Tab.1 |
| steps | 1024 | EVAL.md Tab.1 |
| block_length | 1024 | EVAL.md Tab.1 |
| Paper GSM8K | **70.3%** | EVAL.md |

**Important:** reproduction requires the official lm-eval task builder (`build_all_requests` / CLI).
Do not hand-build `Instance(...)` with a bare question string — that drops the 5-shot context.

## Run

```bash
# Smoke test (10 samples) — official CLI with --log_samples
bash run_gsm8k.sh fp16 smoke
bash run_gsm8k.sh int4 smoke

# View generations from official runs
python show_samples.py results_gsm8k_fp16_smoke.json

# Or re-dump FP16+INT4 side-by-side via official task builder
python dump_generations.py --n 10

# Full GSM8K test set (~1319 samples)
bash run_gsm8k.sh fp16
bash run_gsm8k.sh int4

# Faster variants (checkpointed — auto-resume on restart)
# gen_batch_size=8 by default (override: GEN_BATCH_SIZE=4 bash run_gsm8k.sh ...)
bash run_gsm8k.sh fp16 fast n256          # checkpoint: checkpoints/results_gsm8k_fp16_n256_fast/
bash run_gsm8k.sh fp16 fast 2gpu n256      # resume: re-run same command after crash

# Score interim checkpoint without waiting for job end
python score_checkpoints.py checkpoints/results_gsm8k_fp16_n256_fast_2gpu --limit 256

# Start from scratch (delete checkpoint)
FRESH=1 bash run_gsm8k.sh fp16 fast 2gpu n256
```

## Checkpointing & traces

When `checkpoint_dir` is set (automatic via `run_gsm8k.sh`), each sample saves:

| Path | Contents |
|------|----------|
| `checkpoints/{run}/rank{N}.jsonl` | Final response, prompt, pointer to trace |
| `checkpoints/{run}/traces/rank{N}/{idx}.json.gz` | Full per-step trace (gzip, compact JSON) |

Traces are written asynchronously (2 worker threads) so GPU is not blocked on disk I/O.

Each trace step includes:
- `unmasked`: positions/tokens/confidence actually unmasked this step
- `gap`: зазор WHERE — `boundary_margin` = conf(top-1 pos) − conf(top-2 pos) among masked positions
- `completion_tokens`: full completion-region snapshot before unmask (`mask_id` or revealed token_id)
- `mask_ratio`: `n_masked_before / gen_length` (early/mid/late stage)
- `completion`: per-position signals on masked slots — `confidence`, `predicted_token_id`, `token_margin`, `neg_entropy`, `top10_token_id`

Trace metadata also includes `gold_answer` (GSM8K target) and `mask_id`.

Re-run the same command to resume. Use `FRESH=1` to wipe checkpoints and traces.

**Note:** jobs started before trace support only save final results at the end (lm-eval). Restart with `FRESH=1` if you need step-by-step traces.

## Files

- `generate.py` — official LLaDA generation (from ML-GSAI/LLaDA)
- `eval_llada.py` — lm-eval harness + `quant` arg (fp16/bf16/int4/int8)
- `run_gsm8k.sh` — official CLI launcher (`--log_samples`)
- `dump_generations.py` — side-by-side dump via `task.build_all_requests()`
- `show_samples.py` — render `samples_gsm8k_*.jsonl` from official runs

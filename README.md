# dllm-sparse-kv

Sparse-KV attention on block-diffusion LLMs (Dream Reasoner, Fast-dLLM). Selector picks a top-K subset of old-cache tokens; exec runs with sparse mask + optional KIVI 4-bit K/V quant.

## What's here

- **`src/`** — the sparse-KV runtime: SDPA hook, dual-precision cache, KIVI quant, selector.
- **`bench/`** — gsm8k / math500 sweeps + post-hoc regrader.
- **`docs/ALGORITHM.md`** — precise per-block description of selector + exec.
- **`examples/verify_sparse.py`** — probe script that proves the sparse mask actually fires.

## Configs

Three per model / task:

| name | selector K/V | exec K/V | keep-set built from |
|---|---|---|---|
| `fp16_all_kK` | fp16 / fp16 | fp16 / fp16 | mean(all queries × heads) attention row |
| `fp16_middle_kK` | fp16 / fp16 | fp16 / fp16 | attention row of the middle query |
| `k4sel_v4_kK` | **4-bit KIVI** / **4-bit KIVI** | **4-bit KIVI** / **4-bit KIVI** | mean row of the *quantized* selector attention |

Q is always fp16. `kivi_group_size=32`, `kivi_residual_length=32`.

## Quickstart

```bash
pip install -r requirements.txt
# Dense baseline (no sparse):
python bench/run_dream.py --task gsm8k --num-examples 100 --device cuda:0 \
    --k 64 --selector all_mean --output-dir results/dense

# All 4-bit KIVI (K & V in both phases):
python bench/run_dream.py --task gsm8k --num-examples 100 --device cuda:0 \
    --k 64 --selector all_mean \
    --selector-k-bits 4 --selector-v-bits 4 \
    --exec-k-bits 4 --exec-v-bits 4 \
    --output-dir results/k4sel_v4

# Regrade (fixes \$70{,}000, 26.00 vs 26, \sqrt2 vs \sqrt{2}):
python bench/regrade.py results/*/results.jsonl --show-diffs
```

Requires `transformers>=5.3` (Dream Reasoner uses `validate_rope` which lives there).

## Reproducing the numbers

The two models we benchmarked:

- `Dream-org/DreamReasoner-8B` — Dream Reasoner (36 layers, 32 heads, 8 KV heads GQA, block_size=32).
- `Efficient-Large-Model/Fast_dLLM_v2_7B` — Fast-dLLM v2 (needs the `Fast-dLLM/v2/` upstream inference code).

Datasets: `gsm8k` (`gsm8k/main/test[:100]`) and `HuggingFaceH4/MATH-500` (`test[:100]`), with a chat-templated prompt asking for `\boxed{answer}`.

See `docs/ALGORITHM.md` for how the sparse mask, dual-precision cache and per-block selector interact.

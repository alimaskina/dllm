# Run data behind `docs/eviction_results.md`

Per-example records for every arm in the eviction study. The generated text and
the reference answers are stripped (7 MB of predictions for numbers that are
already decided by the `score` field); everything the report is computed from is
here. The per-arm config, identical on every row of a file, is hoisted into a
sibling `<arm>.config.json`.

Every score in `docs/eviction_results.md` is the mean of the `score` column of
the corresponding file — 21 of 21 cells reproduce exactly.

## Layout

`<model>/<run>/<arm>.jsonl`, with the run named for the capacity rule it used:
`p<percent>_f<floor>` means `C = max(percent%·S, floor)`.

| directory | what it is |
|---|---|
| `fastdllm/grid_p5_f256` | the seven-arm grid on Fast-dLLM-v2-7B |
| `dream/grid_p5_f256` | the same seven arms on DreamReasoner-8B |
| `dream/sweep_{bf16,k4v4}_p5_f512` | capacity sweep, C = 512 flat |
| `dream/sweep_{bf16,k4v4}_p25_f256` | C = max(25%·S, 256) |
| `dream/sweep_{bf16,k4v4}_p25_f512` | C = max(25%·S, 512) |
| `dream/sweep_{bf16,k4v4}_p50_f256` | C = max(50%·S, 256) |
| `dream/sweep_{bf16,k4v4}_p50_f512` | C = max(50%·S, 512) — the setting that won, shipped as `configs/evict_ema_*.yaml` |

Arm names: `dense_bf16` is the no-selector ceiling, `*__none` / `*__evict-none`
keeps the whole cache, `*__recent` / `*__evict-recent` is recency-only eviction,
`*__ema_recent` / `*__evict-ema` is the bias-corrected EMA plus the protected
window. The two spellings are the two runners (`run_eviction_grid.py` and
`run_dream_eviction_grid.py`), not two policies.

`p5_f256` is the capacity from the original specification. It is in the data
because it is what was run first, not because it works: on both models the
median context is 576-640 tokens, so the 5% term never beat the floor and C was
256 in every one of those runs. See the report for what that costs.

## Row schema

Common to both runners: `id`, `score`, and a `runtime` block. Fast-dLLM rows
additionally carry `benchmark`, `model_id`, `dtype`, `prompt_tokens` and
`metadata`; Dream rows carry the model in `environment.json` instead (the
Fast-dLLM grid predates that file and records it per row).

Inside `runtime`, the fields the memory accounting uses:

| field | meaning |
|---|---|
| `eviction_cache_bytes` | payload + quantization metadata for the surviving entries |
| `eviction_state_bytes` | the policy's own `g`/`n`/`pos` arrays |
| `eviction_total_bytes` | the two above; this is what "resident" means |
| `eviction_unevicted_bytes` | what the same request would have cost with no eviction |
| `eviction_bytes_vs_unevicted` | their ratio — the number quoted in the report |
| `eviction_live_entries`, `eviction_capacity`, `eviction_tokens_seen` | how far the policy actually engaged |
| `coverage` | share of fp16 attention mass the kept set captures |

A run where `eviction_tokens_seen` never exceeds `eviction_capacity` never
evicted anything and is identical to the no-eviction arm by construction.

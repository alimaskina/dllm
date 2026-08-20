# Sparse Old-Cache + Low-Bit KV/Q Experimental Framework

Transparent research harness for **Fast-dLLM-v2** exploring:

- **Sparse attention** over fixed old KV cache (top-k after step 0)
- **KIVI-like quantization** (K per-channel, V per-token) simulated via quantize→dequantize
- **Independent selector vs execution precision**
- **Idealized hardware cost** accounting (no custom kernels)

## Layout

```
sparse_kv_exp/
  config.py           # YAML + presets A–D
  quantization.py     # K/V/Q quant geometry
  kv_store.py         # Dual-precision old cache views
  selector.py         # Per-layer top-k policies (all_mean | middle | uniform)
  attention_hook.py   # SDPA patch: sparse old + dense current block
  cost_model.py       # MAC / bandwidth / bit-weighted proxies
  generation.py       # batch_sample_sparse_kv (minimal change to upstream loop)
  logging_utils.py    # JSONL + markdown reports
  run_smoke_gsm8k.py  # Smoke test entry point
  configs/smoke.yaml
```

## Baselines

| Mode | Config |
|------|--------|
| original Fast-dLLM-v2 | `baseline: original` |
| dense FP16 old cache | `sparse_old_cache: false`, all fp16 |
| dense + quant KV | `sparse_old_cache: false`, exec K/V low-bit |
| sparse top-k + FP16 | `sparse_old_cache: true`, exec fp16 |
| sparse + quant KV | sparse + exec K/V low-bit |
| sparse + quant KV + Q | sparse + exec K/V/Q low-bit |

## Handoff sweep (для коллеги)

Полная инструкция: **[HANDOFF.md](HANDOFF.md)**

```bash
# smoke
MODEL=fast_dllm_v2_7b DEVICE=cuda:0 bash smoke_handoff.sh

# full sweep на 8 GPU
MODEL=fast_dllm_v2_7b DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  bash launch_handoff.sh
```


Presets **A–D** match the spec (dense FP16, sparse FP16, sparse+K2V2, selector Q4K2 + exec Q4K2V2).

## Kernel replacement points

1. **`attention_hook._sparse_attention`** — fused sparse QK on old cache + dense current block
2. **`quantization.quantize_dequantize`** — in-place low-bit KV storage + dequant on read
3. **`kv_store.DualPrecisionCache`** — pinned INT2/INT4 buffers instead of FP16 clones
4. **`selector.select_per_layer`** — can run on approximate QK without full SDPA materialization

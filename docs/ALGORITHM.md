# Sparse-KV block-diffusion — per-block algorithm

Notation. `T_q = block_length` (32 for Dream). `old_cache_len` = length of the DynamicCache before the current block (prompt-prefill + all previously finalized blocks). `T_k = old_cache_len + block_length`.

The **first denoising step** of a block runs a *selector forward*: it computes attention weights over the full cache and picks a top-K subset. The **remaining denoising steps** run *exec forwards*: attention is restricted to that subset via a boolean mask (`-inf` on masked columns), and old K/V may be replaced with quantized dequant copies.

Everything is per-block; keep-set is re-picked when we enter the next block.

## SDPA hook (`src/attention_hook.py`)

Patches `ALL_ATTENTION_FUNCTIONS["sdpa"]` with `_capturing_sdpa`. For a single forward:

```
Dream / Fast-dLLM attention layer
    ↓ attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
_capturing_sdpa(query, key, value, attn_mask, ...)
    if CTX.enabled:
        Q ← quantize(query, CTX.q_bits)                       # per-token
        K_old ← key[:, :, :cache_len, :] (from cache)
        K_cur ← key[:, :, cache_len:, :]  (freshly computed for this block)
        # If a dual-precision cache is bound, override the OLD half:
        if CTX.kv_store is not None:
            K_old, V_old ← CTX.kv_store.get_layer_old_kv(layer_idx,
                phase=CTX.phase, sparse=CTX.sparse)
        K, V ← cat([K_old, K_cur]), cat([V_old, V_cur])
        # Sparse mask on OLD columns only:
        if CTX.sparse:
            keep = CTX.per_layer_keep[layer_idx]              # list[int] in [0, cache_len)
            attn_mask ← mask_out_columns(attn_mask, {0..cache_len}\keep)
    return _NATIVE_SDPA(module, Q, K, V, attn_mask, ...)      # unwrapped native backend
```

Note: `_NATIVE_SDPA` is snapshot at module import time. This unwrapping is what prevents recursion when `fast_dllm_attn_capture` wraps this hook.

## Dual-precision cache (`src/kv_store.py`)

`DualPrecisionCache`:

- `bind_fp16(dynamic_cache)` — take a reference to the current DynamicCache.
- `build_precision_views(selector_k_bits, selector_v_bits, exec_k_bits, exec_v_bits, kivi_group_size=32, kivi_residual_length=32, k/v_quant_scheme="kivi")` — produce `selector_k[l]`, `selector_v[l]`, `exec_k[l]`, `exec_v[l]` for each layer as **dequantized fp16 tensors with quant noise**. When bits==16, this is just a clone.
- `build_sparse_exec(per_layer_indices, source="exec", ...)` — build `sparse_exec_k[l]`/`sparse_exec_v[l]` = full-length exec tensor with the keep-set columns optionally re-quantized. With `requantize=False, source="exec"`, this is a no-op — the hook reads `exec_k[l]`/`exec_v[l]` directly.
- `get_layer_old_kv(layer_id, phase, sparse)` — return `(K_old, V_old)` from the right view depending on selector/exec phase.

Note: quantization here is **quant → dequant back to fp16 with noise**, not native int4 inference. Model still runs SDPA in fp16; only the K/V values passed in carry ~10% relative error where quantized. A true int4 kernel would save memory too — this repo measures accuracy under KIVI's quant noise, not the memory saving.

## Selector step (per block, first denoising step)

```
old_cache_len = past_key_values.get_seq_length()
if old_cache_len == 0 or sparse_topk >= old_cache_len:
    # first gen block, or budget k >= cache — no sparse this block
    logits = model.forward(cur_x, past_key_values=past_key_values, use_cache=True, store_kv=False)
else:
    if any_quant_bits < 16:
        kv_store.bind_fp16(past_key_values)
        kv_store.build_precision_views(sel_k_bits, sel_v_bits, exec_k_bits, exec_v_bits, ...)

    with attention_experiment(phase="selector",
                              cache_len=old_cache_len,
                              sparse=False,           # no mask in selector — we want full attention
                              kv_store=kv_store,      # will inject sel_k/sel_v (quant-dequant copies)
                              k_bits=sel_k_bits, v_bits=sel_v_bits,
                              enabled=(kv_store is not None)), \
         capture_attention(None, head_mean=True) as captured:
        logits = model.forward(cur_x, past_key_values=past_key_values, use_cache=True, store_kv=False)
        snap = {l: w.clone() for l, w in captured.items()}   # copy INSIDE the with (finally clears)

    # keep-set: per-layer top-K from the selector attention.
    for layer_id, weights in snap.items():
        # weights shape [B, T_q, T_k] (already mean-over-heads via head_mean=True in capture)
        if selector_mode == "middle":
            row = weights[0, T_q//2, :old_cache_len]
        elif selector_mode == "all_mean":
            row = weights[0, :, :old_cache_len].mean(dim=0)
        keep[layer_id] = sorted(topk(row, sparse_topk).indices.tolist())
```

The selector-forward *also* produces `logits` which are used for the **first denoising step**. Because logits depend on V, if you quantize `selector_v_bits=4` you take a small accuracy hit on that first step.

## Exec steps (per block, denoising steps 1..denoising_steps)

```
if kv_store and keep:
    kv_store.build_sparse_exec(keep, source="exec",
        k_bits=exec_k_bits, v_bits=exec_v_bits, ...)  # no-op when requantize=False

for step in 1..denoising_steps:
    with attention_experiment(phase="exec",
                              cache_len=old_cache_len,
                              sparse=True,
                              per_layer_keep=keep,
                              kv_store=kv_store,
                              k_bits=exec_k_bits, v_bits=exec_v_bits,
                              enabled=True):
        logits = model.forward(cur_x, past_key_values=past_key_values,
                               use_cache=True, store_kv=(step == denoising_steps))
    x0 = sample(logits, temperature, top_k, top_p)
    transfer_index = low_confidence_dynamic_pick(...)
    cur_x[transfer_index] = x0[transfer_index]
```

At the last step the block's K/V is committed into DynamicCache via `store_kv=True`. Next block starts, cache grew by `block_length`, keep-set is picked afresh.

## The three configs

| | selector_k_bits | selector_v_bits | exec_k_bits | exec_v_bits | selector_mode |
|---|---|---|---|---|---|
| `fp16_all_kK` | 16 | 16 | 16 | 16 | `all_mean` |
| `fp16_middle_kK` | 16 | 16 | 16 | 16 | `middle` |
| `k4sel_v4_kK` | 4 | 4 | 4 | 4 | `all_mean` |

With `fp16_*` configs, `kv_store` isn't allocated at all — hook works purely with the raw cache tensors and applies only the sparse mask. With `k4sel_v4_kK`, `kv_store` is bound each block and views are rebuilt because the cache grows.

## Attention mask shapes

Dream's `block_diffusion_attention_mask` is a lower-triangular block-causal mask of shape `[B, H=1, T_q, T_k]` boolean. Fast-dLLM's mask is typically 2D/3D additive fp16. `_apply_sparse_mask` in `attention_hook.py` handles both dtypes and 2D/3D/4D shapes via broadcasting.

## Grading

`bench/regrade.py` walks a JSONL and re-grades each row with a lenient extractor:

- `\$70{,}000` → normalized to `70000` (strip `\$`, `{,}`).
- `26.00` normalizes to `26` (trailing `.0+` stripped).
- For MATH-500: try both `normalize_math` (strip `\\left`/`\\right`, lower, no whitespace) and numeric normalization; either match counts.

Original grader (as produced during a run) is under-counting on `\$` amounts and trailing `.00`.

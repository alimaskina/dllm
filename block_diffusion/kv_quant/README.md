# Fast-dLLM v2 KV Cache Quantization

Block-wise scalar int4/int8 quantization for completed KV cache blocks in Fast-dLLM v2 block diffusion generation.

## Files

| File | Description |
|------|-------------|
| `kv_cache_quant.py` | Core quant/dequant (int4/int8, post-RoPE or pre-RoPE keys) |
| `generation_kv_quant.py` | `batch_sample` with `kv_quant_bits` / `keys_pre_rope` hooks |
| `run_gsm8k_kv_quant.py` | GSM8K accuracy eval (baseline vs quantized) |
| `kv_quant_metrics.py` | MSE / cosine error metrics for KV quant analysis |
| `analyze_kv_quant.py` | Collect KV snapshots and report quant error by layer/block |
| `model_utils.py` | Block size helpers for Fast-dLLM v2 |

## Usage

```bash
conda activate fast_dllm
cd kv_quant

# GSM8K n=50: baseline + int8 + int4, pre-RoPE keys
CUDA_VISIBLE_DEVICES=0 python run_gsm8k_kv_quant.py \
  --n 50 --mode both_all --keys-pre-rope --device cuda:0 \
  --out-dir checkpoints/gsm8k_kv_prerope_n50

# Post-RoPE int8 only
python run_gsm8k_kv_quant.py --n 50 --mode both --device cuda:0

# Quant error analysis (MSE, cosine by layer/block/token)
python analyze_kv_quant.py --n 5 --max-new-tokens 512 --device cuda:0
```

## Integration

```python
import types
import generation_kv_quant

model.mdm_sample = types.MethodType(generation_kv_quant.batch_sample, model)

out = model.mdm_sample(
    input_ids, tokenizer=tokenizer,
    block_size=32, small_block_size=8,
    max_new_tokens=2048, min_len=..., seq_len=...,
    threshold=1.0,
    kv_quant_bits=8,       # 4 or 8; 0 = off
    keys_pre_rope=True,    # quantize keys before RoPE
)
```

## Scheme

- **Granularity**: one symmetric scale per completed block (32 tokens) × layer × head, separate for K and V
- **Post-RoPE** (default): quantize keys as stored in cache
- **Pre-RoPE** (`keys_pre_rope=True`): inverse RoPE → quant → dequant → re-apply RoPE
- Values are always quantized in raw (no RoPE) space

## GSM8K n=50 results (seed=1234)

| Config | Accuracy |
|--------|----------|
| Baseline | 78% |
| int8 post-RoPE | 80% |
| int4 post-RoPE | 26% |
| int8 pre-RoPE | 76% |
| int4 pre-RoPE | 32% |

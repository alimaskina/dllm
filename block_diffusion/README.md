# Block diffusion experiments (Fast-dLLM v2)

Анализ **волатильности предиктов токенов по блокам** при блочной диффузии Fast-dLLM v2.

## Модель и параметры GSM8K

- Модель: [Efficient-Large-Model/Fast_dLLM_v2_7B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B)
- Параметры как в [официальном eval_script.sh](https://github.com/NVlabs/Fast-dLLM/blob/main/v2/eval_script.sh):
  - `num_fewshot=0`
  - `threshold=1`
  - `bd_size=32`, `small_block_size=8`
  - `max_new_tokens=2048`
  - chat template (Qwen2.5 instruct)

## Окружение

```bash
conda create -n fast_dllm python=3.10 -y
conda activate fast_dllm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.53.1 accelerate datasets lm_eval tqdm sentencepiece einops
```

## Запуск

```bash
cd block_diffusion
bash run_gsm8k.sh          # 32 примера
bash run_gsm8k.sh smoke    # 4 примера (быстрая проверка)
bash run_gsm8k.sh n128     # 128 примеров
```

По умолчанию GPU `3` (`CUDA_VISIBLE_DEVICES=3`).

## Что измеряется

Для каждого **блока генерации** (32 токена) и каждой **маскированной позиции**:

1. **first_pred** — argmax на первом forward pass, когда позиция ещё `[MASK]`
2. На каждом следующем pass записывается новый argmax, пока позиция замаскирована
3. **changed_vs_first** — менялся ли argmax хотя бы раз до unmask
4. **committed_differs_from_first** — отличается ли финально закоммиченный токен от first_pred

Агрегация в `analyze_volatility.py`:
- средняя доля «нестабильных» позиций **по номеру блока** (block 0, 1, 2, …)
- разбивка **по позиции внутри блока** (0..31)

## Multi-token word unmasking

Same strict lexical filter as `quant_where_to_unmask` (`multitoken_word_filters.is_lexical`).

Traces must include `preds_by_step` / `conf_by_step` (current `generation_traced.py`).
Re-run `run_volatility.py` if analyzing older traces.

```bash
bash run_multitoken.sh checkpoints/sweep_bd/n128_bd32
# or
conda run -n fast_dllm python analyze_multitoken_words.py \
  --traces checkpoints/sweep_bd/n128_bd32/traces.jsonl --report
conda run -n fast_dllm python generate_multitoken_report.py \
  --traces checkpoints/sweep_bd/n128_bd32/traces.jsonl
```

## Артефакты

```
checkpoints/gsm8k_volatility_n32/
  meta.json
  traces.jsonl          # поколение + полный trace
  volatility_summary.json
  volatility_report.md  # таблицы для быстрого просмотра
```

## Файлы

| Файл | Назначение |
|------|------------|
| `generation_traced.py` | instrumented `mdm_sample` с trace |
| `run_volatility.py` | GSM8K + генерация + dump |
| `analyze_volatility.py` | агрегация по блокам |
| `run_gsm8k.sh` | launcher |
| `fast_dllm_attn_capture.py` | SDPA hook для захвата attention |
| `generation_attn_traced.py` | генерация + cross-block attention на step 0 |
| `run_kv_proxy.py` / `run_kv_proxy.sh` | эксперимент one-shot KV proxy |
| `analyze_kv_proxy.py` | Spearman / overlap / recall vs distance |

### One-shot KV allocation proxy test

Проверяет, предсказывает ли attention из B_{i+1} важность токенов B_i для более поздних блоков.

**Setup (default):** `bd_size=32`, `small_block_size=8`, proxy на **inner_step=0** (весь блок в маске, forward на 32 tok).

```bash
bash run_kv_proxy.sh 16 checkpoints/kv_proxy_n16
# → kv_proxy_report.md, kv_proxy_distance_curve.png
```

Saliency: sum attention от всех query-позиций B_j (step 0) к каждому key-токену B_i, mean over heads × layers.

Upstream reference: `../_fast_dllm_upstream/v2/` (sparse clone NVlabs/Fast-dLLM).

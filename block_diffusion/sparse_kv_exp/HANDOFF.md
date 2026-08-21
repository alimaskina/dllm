# Handoff sweep — инструкция для запуска

Пакет для прогона **sparse KV** на двух моделях, 7 задачах и фиксированной сетке конфигов.

## Что прогоняется

| | |
|---|---|
| **LongBench (4)** | `2wikimqa`, `narrativeqa`, `qmsum`, `repobench-p` |
| **Benchmarks (3)** | `gsm8k`, `math500`, `gpqa_diamond` |
| **Примеров на задачу** | 250 (seed=1234; GPQA Diamond — все ~198) |
| **Модели** | `fast_dllm_v2_7b` (основная), `llada2_mini_16b` (экспериментально) |

### Стратегии (17 конфигов)

1. **baseline** — dense FP16, без sparse cache  
2. **middle** × 4 budget tiers  
3. **uniform5** × 4 budget tiers  
4. **extreme** (all_mean selector) × 2 quant × 4 budget tiers:
   - k2v2, k4v4

### Budget (top-k как % old cache)

`2.5%`, `5%`, `10%`, `20%` — per-head, per block:

```
k = max(1, round(old_cache_len × pct / 100))
```

**Итого на модель:** 17 configs × 250 examples × 7 tasks = **29 750 runs**  
_(GPQA Diamond: фактически 17 × 198 × 1 ≈ 3366 runs — датасет меньше 250)_

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone <repo-url> dllm && cd dllm
```

Нужны:
- `block_diffusion/sparse_kv_exp/` — этот пакет
- `_fast_dllm_upstream/v2/` — upstream Fast-dLLM-v2 (generation_functions)

### 2. Conda-окружение (Fast-dLLM)

```bash
conda create -n fast_dllm python=3.10 -y
conda activate fast_dllm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==4.53.1" accelerate datasets lm-eval tqdm sentencepiece einops pyyaml
pip install rouge-score fuzzywuzzy python-Levenshtein
```

Модель скачается с HF при первом запуске:
`Efficient-Large-Model/Fast_dLLM_v2_7B`

### 3. Smoke-тест (1 пример, ~2 мин)

```bash
cd block_diffusion/sparse_kv_exp
chmod +x smoke_handoff.sh launch_handoff.sh
MODEL=fast_dllm_v2_7b DEVICE=cuda:0 bash smoke_handoff.sh
```

Проверить: `results/handoff_smoke/fast_dllm_v2_7b/report.md`

### 4. Полный sweep на N GPU

Один GPU = одна задача (7 задач, можно меньше GPU — задачи round-robin):

```bash
MODEL=fast_dllm_v2_7b \
DEVICES=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
NUM_EXAMPLES=250 \
bash launch_handoff.sh
```

Логи: `results/handoff/fast_dllm_v2_7b/run_*.log`  
Прогресс: `wc -l results/handoff/fast_dllm_v2_7b/results.jsonl` (цель: ~29 750 строк)

### 5. Отчёт

```bash
python run_handoff_report.py --output-dir results/handoff/fast_dllm_v2_7b
```

---

## LLaDA 2.0-mini 16B

```bash
MODEL=llada2_mini_16b CONDA_ENV=llada_quant DEVICES=cuda:0 bash smoke_handoff.sh
```

**Статус:** экспериментально. Sparse KV hooks написаны под block-diffusion loop Fast-dLLM-v2.  
LLaDA2 загружается через HF (`inclusionAI/LLaDA2.0-mini`), но для sparse нужен совместимый `mdm_sample`.  
Если smoke падает — сначала прогоните только `fast_dllm_v2_7b`, LLaDA подключим отдельно.

---

## Ручной запуск одной задачи

LongBench:
```bash
conda activate fast_dllm
cd block_diffusion/sparse_kv_exp

python run_longbench_sweep.py \
  --sweep handoff \
  --model fast_dllm_v2_7b \
  --tasks narrativeqa \
  --num-examples 250 \
  --device cuda:0 \
  --output-dir results/handoff/fast_dllm_v2_7b
```

Benchmark:
```bash
python run_benchmark_sweep.py \
  --sweep handoff \
  --model fast_dllm_v2_7b \
  --tasks gsm8k \
  --num-examples 250 \
  --device cuda:1 \
  --output-dir results/handoff/fast_dllm_v2_7b
```

Resume: повторный запуск пропускает уже записанные `(config, task, example_id)` в `results.jsonl`.

---

## Структура результатов

```
results/handoff/<model>/
  results.jsonl      # одна строка = один run
  sweep_meta.json    # sweep/model/pcts
  analysis.json
  report.md          # таблицы score (coverage) по cache %
  run_*.log          # stdout воркеров
```

Поля в JSONL: `task`, `example_id`, `score`, `avg_topk_coverage`, `prompt_tokens`, `gen_tokens`, `config`, `cost`.

---

## Оценка времени / GPU

- ~29 750 runs × ~30–120 s/run (зависит от длины контекста)  
- 8× A100: порядка **2–5 дней** на модель  
- Можно уменьшить `--num-examples` или `--families` для отладки

---

## Контакты / проблемы

| Проблема | Решение |
|----------|---------|
| OOM на LongBench | уменьшить `max_new_tokens` в `longbench_sweep_configs.py` `_COMMON` |
| `upstream not found` | проверить `_fast_dllm_upstream/v2/` |
| Дубликаты в jsonl | нормально при resume; ключ `(config.name, task, idx)` |
| LLaDA не стартует | использовать `fast_dllm_v2_7b` |

Список конфигов:
```bash
python -c "from handoff_sweep_configs import handoff_config_list; print(len(handoff_config_list()), 'configs')"
python -c "from handoff_sweep_configs import handoff_config_list; [print(c.name) for c in handoff_config_list()]"
```

# Окружение для экспериментов dllm

Краткая инструкция, чтобы поднять то же окружение на другом GPU-сервере.

## Что нужно на сервере

| Компонент | Текущий сервер (референс) |
|-----------|---------------------------|
| GPU | 5× NVIDIA A100-SXM4-80GB |
| Driver | 550.163.01 |
| CUDA (PyTorch) | **12.4** (`cu124`) |
| Conda | Miniconda / Anaconda |
| Git + Git LFS | для клонирования репозитория |
| Hugging Face CLI | `pip install huggingface_hub` → `huggingface-cli login` |

Минимум для 8B-моделей: **1× GPU ≥ 40 GB** (smoke), для полных прогонов квантования — **~80 GB**.

---

## Быстрая установка (рекомендуется)

```bash
git clone git@github.com:alimaskina/dllm.git   # или ваш remote
cd dllm
git lfs pull                                  # если используете LFS-артефакты

# Оба conda-окружения одной командой
bash scripts/setup_environment.sh

# Скачать LLaDA-8B-Base (нужна для quant_*, multi_language, causal_pilot)
huggingface-cli login
bash scripts/setup_environment.sh models
```

Проверка:

```bash
conda activate fast_dllm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

conda activate llada_quant
python -c "import torch, bitsandbytes; print(torch.__version__, torch.cuda.is_available())"
```

---

## Два conda-окружения

В репозитории **два** основных env — у них разные версии `transformers`, их нельзя смешивать в одном venv.

### `fast_dllm` — Fast-dLLM v2, block_diffusion

| Пакет | Версия |
|-------|--------|
| Python | 3.10 |
| torch | 2.6.0+cu124 |
| transformers | 4.53.1 |
| accelerate | 1.14.0 |
| lm_eval | 0.4.12 |
| datasets | 5.0.1 |

**Используется в:** `block_diffusion/` (volatility, KV proxy, multitoken analysis).

```bash
conda activate fast_dllm
cd block_diffusion
bash run_kv_proxy.sh 16 checkpoints/kv_proxy_n16   # GPU 1 по умолчанию
bash run_gsm8k.sh smoke                             # GPU 3 по умолчанию
```

Модель подтягивается из HF Hub: `Efficient-Large-Model/Fast_dLLM_v2_7B`.

### `llada_quant` — LLaDA eval, квантование, multi_language

| Пакет | Версия |
|-------|--------|
| Python | 3.10 |
| torch | 2.5.1+cu124 |
| transformers | 4.46.2 |
| accelerate | 0.34.2 |
| bitsandbytes | 0.44.1 |
| lm_eval | 0.4.8 |
| peft | 0.19.1 |

**Используется в:** `quant_where_to_unmask/`, `quant_unmask/`, `multi_language/`, `multi_language/causal_pilot/`.

```bash
conda activate llada_quant
cd quant_where_to_unmask
bash run_gsm8k.sh fp16 smoke

cd ../multi_language
CUDA_VISIBLE_DEVICES=0 bash causal_pilot/run_causal_pilot.sh probe
```

---

## Ручная установка (если скрипт не подходит)

### fast_dllm

```bash
conda create -n fast_dllm python=3.10 -y
conda activate fast_dllm
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r environment/fast_dllm.txt
```

### llada_quant

```bash
conda create -n llada_quant python=3.10 -y
conda activate llada_quant
pip install torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r environment/llada_quant.txt
```

---

## Модели и пути

Веса **не** лежат в git. Локальная схема:

```
dllm/
  model/
    LLaDA-8B-Base/          # локальная копия или symlink в HF cache
  _fast_dllm_upstream/v2/   # vendored reference NVlabs/Fast-dLLM (уже в репо)
```

### LLaDA-8B-Base (обязательна для большинства экспериментов)

```bash
mkdir -p model
huggingface-cli download GSAI-ML/LLaDA-8B-Base \
  --local-dir model/LLaDA-8B-Base
```

Или symlink на уже скачанный snapshot:

```bash
ln -s ~/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/<hash> \
  model/LLaDA-8B-Base
```

### Fast-dLLM v2 7B

По умолчанию грузится напрямую из Hub при первом запуске. Кэш:

```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
```

### Дополнительные модели (quant_dllm_upstream)

См. `quant_dllm_upstream/README.md` — LLaDA-Instruct, Dream и др. в `../model/`.

---

## Переменные окружения

Добавьте в `~/.bashrc` или экспортируйте перед запуском:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0          # нужная GPU
export REPO_ROOT="$HOME/dllm"          # корень репозитория (опционально)
```

Скрипты `block_diffusion/run_*.sh` используют `conda run -n fast_dllm`.
Скрипты `quant_where_to_unmask/run_*.sh` — `conda run -n llada_quant`.
`multi_language/` и `causal_pilot/` ожидают активированный `llada_quant` (или `conda run -n llada_quant`).

---

## Карта: какой env для какой папки

| Папка | Conda env | GPU (дефолт в скриптах) |
|-------|-----------|-------------------------|
| `block_diffusion/` | `fast_dllm` | 1 (kv_proxy), 3 (gsm8k) |
| `quant_where_to_unmask/` | `llada_quant` | задаётся в run_*.sh |
| `quant_unmask/` | `llada_quant` | 2 |
| `multi_language/` | `llada_quant` | 0–1 |
| `multi_language/causal_pilot/` | `llada_quant` | 0 (train_parallel: 2,3,4) |
| `quant_dllm_upstream/` | `quant-dllm` (отдельный, см. их README) | 80 GB |

---

## Третье окружение (опционально): quant-dllm

Для пайплайна из `quant_dllm_upstream/` (PTQ Quant-dLLM):

```bash
cd quant_dllm_upstream
conda env create -f environment.yml   # env name: quant-dllm, Python 3.11
conda activate quant-dllm
```

Это **отдельный** env с `transformers==4.49.0`; не смешивать с `llada_quant`.

---

## Типичные проблемы

**CUDA mismatch.** PyTorch здесь собран под CUDA 12.4. Если драйвер старый — обновите NVIDIA driver или установите torch под вашу CUDA с [pytorch.org](https://pytorch.org).

**OOM на 8B.** Уменьшите batch / `max_new_tokens`, используйте smoke-режимы (`run_gsm8k.sh smoke`, `run_kv_proxy.sh 8`).

**Hardcoded пути `/home/alimaskina/...`.** В `causal_pilot/run_causal_pilot.sh` и некоторых скриптах quant_* задан абсолютный путь к модели. На новом сервере:

```bash
export INIT_CHECKPOINT="$PWD/model/LLaDA-8B-Base"
bash causal_pilot/run_causal_pilot.sh probe
```

**lm_eval кэш.** Первый запуск GSM8K скачает датасет в `~/.cache/huggingface/datasets`.

---

## Экспорт точного состояния env (для бэкапа)

```bash
conda run -n fast_dllm pip freeze > environment/fast_dllm.freeze.txt
conda run -n llada_quant pip freeze > environment/llada_quant.freeze.txt
```

Freeze-файлы включают nvidia-* wheels и привязаны к конкретной машине; для переноса лучше `environment/*.txt` + `scripts/setup_environment.sh`.

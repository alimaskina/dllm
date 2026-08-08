# Полный отчёт: multi-token word unmasking (WikiText g64)

*Сгенерировано: 2026-07-30 10:54 UTC*

## Содержание

1. [Резюме](#1-резюме)
2. [Данные и методология](#2-данные-и-методология)
3. [Инвентарь слов (Tier A / Tier B)](#3-инвентарь-слов-tier-a--tier-b)
4. [Топология и layout](#4-топология-и-layout)
5. [Порядок unmask](#5-порядок-unmask)
6. [Временная динамика (step_span, gaps)](#6-временная-динамика-step_span-gaps)
7. [Confidence](#7-confidence)
8. [Sibling prediction и co-unmask](#8-sibling-prediction-и-co-unmask)
9. [Разбивка по числу токенов](#9-разбивка-по-числу-токенов)
10. [Sibling vs non-sibling conf](#10-sibling-vs-non-sibling-conf)
11. [Сравнение WikiText vs GSM8K](#11-сравнение-wikitext-vs-gsm8k)
12. [По checkpoint'ам](#12-по-checkpointам)
13. [Примеры](#13-примеры)
14. [Выводы](#14-выводы)

---

## 1. Резюме

Проанализировано **3328** generation traces (WikiText-103, gen_length=64, k=1/step). После loop-фильтра: **3201** traces (96.2%). Найдено **10266** alpha multi-token слов (Tier A) и **6796** lexical (Tier B).

### Как unmask'ятся multi-token слова

- **65.6%** слов с step_span=1 (2-токенные → 2 соседних шага); median span = **1**.
- **89.1%** слов — все токены unmask'ятся подряд по шагам (gap=1).
- Первый unmask: left **47.1%**, right **42.9%**, middle **10.1%**.
- 2-токенные: LR **47.5%**, RL **52.5%**; 3-токенные: **84%** подряд, порядки сильно mixed.

### Sibling prediction (ещё masked токены слова при первом unmask)

- pred==final: **59.8%** (9055 obs)
- все siblings верны (per word): **56.3%**
- co-unmask при conf≥0.8: **100.0%** accuracy (1031 cases, 0 errors)
- first_conf <0.3 → sibling acc падает до **35%**

### Дополнительно: sibling vs non-sibling conf

При первом unmask siblings увереннее остальных masked-позиций (mean 0.327 vs 0.090), но это лишь один из аспектов — подробнее в [§10](#10-sibling-vs-non-sibling-conf).

---

## 2. Данные и методология

| Параметр | Значение |
|----------|----------|
| Модель | GSAI-ML/LLaDA-8B-Base, fp16, temperature=0 |
| Датасет | WikiText-103 (validation + 6× train batches) |
| Prompt | первые 48 токенов строки |
| gen_length / steps / block | 64 / 64 / 64 |
| k per step | 1 (один unmask за шаг) |
| remasking | low_confidence |
| mask_id | 126336 |
| Всего traces | 3328 |
| Loop traces (отброшены) | 127 (3.8%) |
| No-loop traces | 3201 (96.2%) |

**Loop-фильтр:** фраза (ngram≥4) повторяется ≥3 раз в response → trace помечен dirty (типично для g256: `Luxembourg`, `caption;` loops; g64 почти чистый).

**Сегментация слов:** whitespace tokenization (`\S+`), multi-token = ≥2 tokenizer-токена на одно слово.

**Tier A (alpha):** буквенные слова, core≥3 символа (включает `image;`, `present.` и т.п.).

**Tier B (lexical):** strict multi-token words — Tier A минус infobox/task junk, без склеенной пунктуации (`present.`, `height;`, `n't`). См. `multitoken_word_filters.is_lexical`.


---

## 3. Инвентарь слов (Tier A / Tier B)

| Tier | instances | per trace (no-loop) |
|------|-----------|---------------------|
| A (alpha) | 10266 | 3.21 |
| B (lexical) | 6796 | 2.12 |
| Отфильтровано A→B | 3470 (33.8%) |

### Распределение по числу токенов (Tier B)

| ntok | count | % |
|------|-------|---|
| 2 | 5001 | 73.6% |
| 3 | 1422 | 20.9% |
| 4 | 309 | 4.5% |
| 5 | 51 | 0.8% |
| 6 | 8 | 0.1% |
| 8 | 3 | 0.0% |
| 10 | 2 | 0.0% |

### Топ-30 слов (Tier B)

| word | count |
|------|-------|
| `wto` | 107 |
| `pascal` | 38 |
| `lamy` | 35 |
| `Scientology` | 33 |
| `westward` | 30 |
| `northward` | 16 |
| `landfall` | 16 |
| `northwestward` | 16 |
| `Tikal` | 15 |
| `beijing` | 15 |
| `cyclone` | 15 |
| `colonel` | 14 |
| `thursday` | 13 |
| `convoy` | 13 |
| `Scully` | 13 |
| `battalion` | 12 |
| `disbanded` | 12 |
| `torpedo` | 12 |
| `france` | 12 |
| `wolfensohn` | 11 |
| `HMS` | 11 |
| `Mulder` | 11 |
| `tuesday` | 10 |
| `squadron` | 10 |
| `It's` | 10 |
| `turrets` | 10 |
| `typhoon` | 10 |
| `destroyers` | 10 |
| `ECW` | 10 |
| `Rhodesian` | 9 |

---

## 4. Топология

| Метрика | Значение |
|---------|----------|
| Same-step unmask | 0.0% (k=1 → всегда 0) |
| Все токены подряд по шагам | 89.1% |

Позиции токенов в completion **смежные по определению** (слово = непрерывный кусок текста → подряд идущие tokenizer indices). Это не находка.

---

## 5. Порядок unmask

### Позиция первого unmask

| Позиция | count | % |
|----------|-------|---|
| left | 3198 | 47.1% |
| middle | 683 | 10.1% |
| right | 2915 | 42.9% |

### 2 токена (n=5001)

- LR: 51.4%, RL: 48.6%
- consecutive (gap=1): 89.1%

#### Gap-распределение (2 токена)

| gap | count | % |
|-----|-------|---|
| 1 | 4455 | 89.1% |
| 2 | 272 | 5.4% |
| 3 | 103 | 2.1% |
| 4 | 41 | 0.8% |
| 5 | 53 | 1.1% |
| 6 | 20 | 0.4% |
| 7 | 14 | 0.3% |

### 3 токена (n=1422)

- fully consecutive: 90.8%
- monotone_LR: 26.7%, monotone_RL: 29.4%, mixed: 43.9%

| pattern | count | % |
|---------|-------|---|
| LMR | 380 | 26.7% |
| MLR | 311 | 21.9% |
| RML | 250 | 17.6% |
| MRL | 169 | 11.9% |
| RLM | 168 | 11.8% |
| LRM | 144 | 10.1% |

#### Gap-распределение (3 токена)

| gap | count | % |
|-----|-------|---|
| 1 | 2678 | 94.2% |
| 2 | 76 | 2.7% |
| 3 | 30 | 1.1% |
| 4 | 13 | 0.5% |
| 5 | 14 | 0.5% |
| 6 | 11 | 0.4% |

### 4 токена (n=309)

- fully consecutive: 81.9%
- monotone_LR: 10.0%, monotone_RL: 7.8%, mixed: 82.2%

| pattern | count | % |
|---------|-------|---|
| LMMR | 60 | 19.4% |
| MLMR | 56 | 18.1% |
| LMRM | 31 | 10.0% |
| RMML | 24 | 7.8% |
| MMLR | 24 | 7.8% |
| MMRL | 24 | 7.8% |
| MLRM | 22 | 7.1% |
| MRML | 21 | 6.8% |
| RMLM | 19 | 6.1% |
| RLMM | 16 | 5.2% |
| MRLM | 9 | 2.9% |
| LRMM | 3 | 1.0% |

#### Gap-распределение (4 токена)

| gap | count | % |
|-----|-------|---|
| 1 | 864 | 93.2% |
| 2 | 26 | 2.8% |
| 3 | 11 | 1.2% |

### 5 токена (n=51)

- fully consecutive: 90.2%
- monotone_LR: 0.0%, monotone_RL: 0.0%, mixed: 100.0%

| pattern | count | % |
|---------|-------|---|
| MLMMR | 11 | 21.6% |
| MMLMR | 7 | 13.7% |
| MMMRL | 5 | 9.8% |
| MMMLR | 5 | 9.8% |
| LMMMR | 4 | 7.8% |
| RMMML | 3 | 5.9% |
| MMRLM | 3 | 5.9% |
| MRMLM | 3 | 5.9% |
| LMMRM | 2 | 3.9% |
| MMLRM | 1 | 2.0% |
| RLMMM | 1 | 2.0% |
| RMLMM | 1 | 2.0% |

#### Gap-распределение (5 токена)

| gap | count | % |
|-----|-------|---|
| 1 | 198 | 97.1% |
| 2 | 5 | 2.5% |
| 3 | 1 | 0.5% |

### 4+ токена суммарно (n=373)

- step_span mean: 4.36, median: 3
- monotone_LR: 8.3%, monotone_RL: 6.4%, mixed: 85.3%

---

## 6. Временная динамика (step_span, gaps)

**step_span** = `last_unmask_step − first_unmask_step`. Это **не** число шагов unmask!
- span=1 у 2-токенного слова → unmask на шагах T и T+1 → **2 шага**
- span=2 у 3-токенного → минимум 3 шага подряд
- span=0 невозможен при k=1 и ≥2 токенах

| Метрика | Значение |
|---------|----------|
| step_span mean | 1.74 |
| step_span median | 1 |
| step_span p25 / p75 / p90 | 1 / 2 / 3 |
| step_span max | 52 |
| first_step mean / median | 30.4 / 31 |

### Распределение step_span

| span | count | % |
|------|-------|---|
| 1 | 4455 | 65.6% |
| 2 | 1563 | 23.0% |
| 3 | 403 | 5.9% |
| 4 | 135 | 2.0% |
| 5 | 84 | 1.2% |
| 6 | 33 | 0.5% |
| 7 | 26 | 0.4% |
| 8 | 20 | 0.3% |
| 9 | 9 | 0.1% |
| 11 | 7 | 0.1% |
| 12 | 7 | 0.1% |
| 15 | 6 | 0.1% |
| ≥8 | 97 | 1.4% |

**Интерпретация:** long span (≥8) — слово «растянуто» по многим шагам, между токенами unmask'ятся другие позиции. Это ~1.4% случаев.

---

## 7. Confidence

### 7.1 Распределение confidence при unmask

| Какой токен | n | mean | median | p10 | p90 |
|-------------|---|------|--------|-----|-----|
| Первый токен слова | 6796 | 0.476 | 0.404 | 0.161 | 0.242 | 0.688 | 0.953 |
| Последний токен слова | 6796 | 0.925 | 0.996 | 0.718 | 0.966 | 0.999 | 1.000 |
| Все токены всех слов | 15851 | 0.735 | 0.947 | 0.223 | 0.429 | 0.998 | 1.000 |

### 7.2 First conf → качество siblings

| first_conf | words | sibling_acc | all_siblings_ok |
|------------|-------|-------------|------------------|
| <0.3 | 2409 | 34.8% | 29.0% |
| 0.3-0.7 | 2733 | 62.6% | 58.3% |
| 0.7-0.9 | 715 | 88.0% | 86.9% |
| >=0.9 | 939 | 97.4% | 97.1% |

### 7.3 mask_ratio при первом unmask слова

mean=0.525, median=0.516, p10=0.141, p90=0.922

(Доля ещё masked позиций в completion в момент первого unmask токена слова.)

---

## 8. Sibling prediction и co-unmask

В момент первого unmask слова: для каждого **ещё masked** sibling смотрим `completion.predicted_token_id[pos]` vs финальный токен.

| Метрика | Значение |
|---------|----------|
| Sibling observations | 9055 |
| pred == final | **59.8%** |
| все siblings верны (per word) | **56.3%** |

### По sibling confidence

| conf | accuracy | n |
|------|----------|---|
| <0.3 | 42.5% | 5639 |
| 0.3-0.5 | 76.7% | 1465 |
| 0.5-0.7 | 91.9% | 640 |
| 0.7-0.9 | 99.3% | 561 |
| >=0.9 | 100.0% | 750 |

### Co-unmask политики

| threshold | accuracy | n | wrong |
|-----------|----------|---|-------|
| ≥0.5 | 97.1% | 1951 | 56 |
| ≥0.6 | 99.2% | 1593 | 13 |
| ≥0.7 | 99.7% | 1311 | 4 |
| ≥0.75 | 99.9% | 1175 | 1 |
| ≥0.8 | 100.0% | 1031 | 0 |
| ≥0.85 | 100.0% | 892 | 0 |
| ≥0.9 | 100.0% | 750 | 0 |
| ≥0.95 | 100.0% | 529 | 0 |

**Смысл:** если при первом unmask sibling уже имеет высокий conf и правильный pred, можно безопасно unmask'ить его на том же шаге (co-unmask). При conf≥0.8 — 99.8% accuracy.

---

## 9. Разбивка по числу токенов

| ntok | count | % | span_med | first=left | first=right | sib_acc | co≥0.8 acc |
|------|-------|---|----------|------------|-------------|---------|------------|
| 2 | 5001 | 73.6% | 1 | 51.4% | 48.6% | 60.8% | 100.0% (641) |
| 3 | 1422 | 20.9% | 2 | 36.8% | 29.4% | 61.4% | 100.0% (314) |
| 4 | 309 | 4.5% | 3 | 30.4% | 19.1% | 50.8% | 100.0% (69) |
| 5 | 51 | 0.8% | 4 | 13.7% | 11.8% | 61.8% | 100.0% (7) |
| 6 | 8 | 0.1% | 5 | 37.5% | 0.0% | 60.0% | 0.0% (0) |
| 8 | 3 | 0.0% | 8 | 0.0% | 0.0% | 19.0% | 0.0% (0) |
| 10 | 2 | 0.0% | 9 | 50.0% | 0.0% | 27.8% | 0.0% (0) |

---

## 10. Sibling vs non-sibling conf

Дополнительный анализ: насколько увереннее masked-токены **внутри слова** vs **вне слова** в момент первого unmask.

| Группа | n | mean | median | p10 | p90 |
|--------|---|------|--------|-----|-----|
| Sibling (внутри слова) | 9055 | 0.327 | 0.217 | 0.077 | 0.847 |
| Non-sibling (остальные masked) | 212360 | 0.090 | 0.054 | 0.039 | 0.144 |
| Adjacent non-word (±1) | 6720 | 0.259 | 0.159 | 0.061 | 0.688 |

Per-word: sibling mean > non-sibling в **92.6%** словах (6293/6796).

| ntok | sibling mean | non-sibling mean | Δ |
|------|--------------|------------------|---|
| 2 | 0.346 | 0.091 | +0.255 |
| 3 | 0.316 | 0.090 | +0.226 |
| 4 | 0.279 | 0.081 | +0.198 |
| 5 | 0.272 | 0.064 | +0.207 |
| 6 | 0.246 | 0.077 | +0.169 |
| 8 | 0.181 | 0.071 | +0.110 |
| 10 | 0.103 | 0.053 | +0.050 |

## 11. Сравнение WikiText vs GSM8K

| Метрика | WikiText g64 (lexical) | GSM8K fp16 g256 (alpha) |
|---------|------------------------|-------------------------|
| traces | 3328 (3201 no-loop) | 256 |
| word instances | 6796 | 4218 |
| words/trace | 2.12 | 16.48 |
| full word consecutive steps | 89.1% | 48.0% |
| step_span median | 1 | 2 |
| step_span mean | 1.74 | 10.22 |
| first=left | 47.1% | 50.3% |
| first=right | 42.9% | 47.8% |
| first=middle | 10.1% | 1.8% |
| 2tok LR | 47.5% | 49.7% |
| 2tok RL | 52.5% | 50.3% |
| first conf mean | 0.476 | 0.792 |
| all-token conf mean | 0.735 | 0.821 |
| sibling pred==final | 59.8% | 74.9% |
| all siblings ok | 56.3% | 74.7% |
| co-unmask conf≥0.8 | 100.0% | 100.0% |
| sibling conf mean | 0.327 | 0.606 |
| non-sibling conf mean | 0.090 | 0.130 |

GSM8K: более структурированные ответы → выше sibling accuracy и first conf. Порядок unmask на WikiText более симметричен (left≈right).

## 12. По checkpoint'ам

| checkpoint | traces* | words | words/trace | span_med |
|------------|---------|-------|-------------|----------|
| `results_wikitext_fp16_g64_n256` | — | 538 | 2.10 | 1 |
| `results_wikitext_fp16_g64_train_seed43_n512` | — | 1048 | 2.05 | 1 |
| `results_wikitext_fp16_g64_train_seed1043_n512` | — | 1111 | 2.17 | 1 |
| `results_wikitext_fp16_g64_train_seed2043_n512` | — | 1006 | 1.96 | 1 |
| `results_wikitext_fp16_g64_train_seed3043_n512` | — | 1083 | 2.12 | 1 |
| `results_wikitext_fp16_g64_train_seed4043_n512` | — | 1007 | 1.97 | 1 |
| `results_wikitext_fp16_g64_train_seed5043_n512` | — | 1003 | 1.96 | 1 |

*traces per checkpoint: 256 (val) или 512 (train batches)

---

## 13. Примеры

Слова с наибольшим числом wrong sibling predictions:


### `Andriantsimitoviaminiandriana` (10 tok, pattern=LMMMMMMMRM, span=9)

| step | pos | token | conf |
|------|-----|-------|------|
| 14 | 14 | ` And` | 0.311 |
| 15 | 15 | `ri` | 0.843 |
| 16 | 17 | `imit` | 0.566 |
| 17 | 16 | `ants` | 0.915 |
| 18 | 18 | `ov` | 0.654 |
| 19 | 19 | `iam` | 0.547 |
| 20 | 21 | `and` | 0.681 |
| 21 | 20 | `ini` | 0.959 |
| 22 | 23 | `ana` | 0.934 |
| 23 | 22 | `ri` | 0.978 |

Siblings at first unmask:

| pos | pred | conf | final | ok |
|-----|------|------|-------|----|
| 15 | `ri` | 0.181 | `ri` | ✓ |
| 16 | `ri` | 0.049 | `ants` | ✗ |
| 17 | `ri` | 0.087 | `imit` | ✗ |
| 18 | `ri` | 0.066 | `ov` | ✗ |
| 19 | ` ,` | 0.094 | `iam` | ✗ |
| 20 | ` ,` | 0.074 | `ini` | ✗ |
| 21 | ` ,` | 0.058 | `and` | ✗ |
| 22 | `ri` | 0.051 | `ri` | ✓ |
| 23 | `ri` | 0.061 | `ana` | ✗ |

### `Sturzkampfgeschwader` (8 tok, pattern=MRMMMMML, span=8)

| step | pos | token | conf |
|------|-----|-------|------|
| 12 | 18 | `w` | 0.233 |
| 13 | 19 | `ader` | 0.986 |
| 14 | 17 | `ch` | 0.993 |
| 15 | 16 | `ges` | 0.965 |
| 17 | 14 | `amp` | 0.519 |
| 18 | 15 | `f` | 0.999 |
| 19 | 13 | `zk` | 0.827 |
| 20 | 12 | ` Stur` | 0.979 |

Siblings at first unmask:

| pos | pred | conf | final | ok |
|-----|------|------|-------|----|
| 12 | ` the` | 0.142 | ` Stur` | ✗ |
| 13 | ` ` | 0.220 | `zk` | ✗ |
| 14 | ` ` | 0.129 | `amp` | ✗ |
| 15 | `ch` | 0.094 | `f` | ✗ |
| 16 | `w` | 0.131 | `ges` | ✗ |
| 17 | `w` | 0.195 | `ch` | ✗ |
| 19 | ` ` | 0.104 | `ader` | ✗ |

### `Andriantsimitoviaminiand` (8 tok, pattern=MLMMMMRM, span=7)

| step | pos | token | conf |
|------|-----|-------|------|
| 48 | 57 | `ri` | 0.177 |
| 49 | 56 | ` And` | 0.919 |
| 50 | 59 | `imit` | 0.651 |
| 51 | 58 | `ants` | 0.998 |
| 52 | 60 | `ov` | 0.978 |
| 53 | 61 | `iam` | 0.988 |
| 54 | 63 | `and` | 0.976 |
| 55 | 62 | `ini` | 0.998 |

Siblings at first unmask:

| pos | pred | conf | final | ok |
|-----|------|------|-------|----|
| 56 | `ri` | 0.174 | ` And` | ✗ |
| 58 | `ri` | 0.142 | `ants` | ✗ |
| 59 | `ri` | 0.126 | `imit` | ✗ |
| 60 | `ri` | 0.142 | `ov` | ✗ |
| 61 | `ri` | 0.139 | `iam` | ✗ |
| 62 | `ri` | 0.067 | `ini` | ✗ |
| 63 | `ri` | 0.128 | `and` | ✗ |

### `Andriantsimitoviaminiandriana` (10 tok, pattern=MLMMMMMMMR, span=9)

| step | pos | token | conf |
|------|-----|-------|------|
| 28 | 30 | `ri` | 0.447 |
| 29 | 29 | ` And` | 0.987 |
| 30 | 32 | `imit` | 0.714 |
| 31 | 31 | `ants` | 0.876 |
| 32 | 33 | `ov` | 0.495 |
| 33 | 34 | `iam` | 0.598 |
| 34 | 36 | `and` | 0.485 |
| 35 | 35 | `ini` | 0.934 |
| 36 | 37 | `ri` | 0.745 |
| 37 | 38 | `ana` | 0.761 |

Siblings at first unmask:

| pos | pred | conf | final | ok |
|-----|------|------|-------|----|
| 29 | ` And` | 0.325 | ` And` | ✓ |
| 31 | `ri` | 0.054 | `ants` | ✗ |
| 32 | `imit` | 0.285 | `imit` | ✓ |
| 33 | `ri` | 0.060 | `ov` | ✗ |
| 34 | ` .` | 0.101 | `iam` | ✗ |
| 35 | ` .` | 0.082 | `ini` | ✗ |
| 36 | `ri` | 0.079 | `and` | ✗ |
| 37 | `ri` | 0.074 | `ri` | ✓ |
| 38 | `ri` | 0.075 | `ana` | ✗ |

### `Konzentrationslager` (6 tok, pattern=LMMMMR, span=5)

| step | pos | token | conf |
|------|-----|-------|------|
| 39 | 39 | ` K` | 0.201 |
| 40 | 40 | `onz` | 0.823 |
| 41 | 41 | `ent` | 1.000 |
| 42 | 42 | `rations` | 0.996 |
| 43 | 43 | `l` | 0.897 |
| 44 | 44 | `ager` | 0.988 |

Siblings at first unmask:

| pos | pred | conf | final | ok |
|-----|------|------|-------|----|
| 40 | `enk` | 0.082 | `onz` | ✗ |
| 41 | `enk` | 0.182 | `ent` | ✗ |
| 42 | `enk` | 0.083 | `rations` | ✗ |
| 43 | `enk` | 0.084 | `l` | ✗ |
| 44 | ` -` | 0.062 | `ager` | ✗ |

### `Dicraeosaurus` (5 tok, pattern=RMLMM, span=5)

| step | pos | token | conf |
|------|-----|-------|------|
| 46 | 48 | `aurus` | 0.495 |
| 47 | 47 | `os` | 0.417 |
| 49 | 44 | ` D` | 0.152 |
| 50 | 45 | `ic` | 0.595 |
| 51 | 46 | `rae` | 0.978 |

Siblings at first unmask:

| pos | pred | conf | final | ok |
|-----|------|------|-------|----|
| 44 | ` E` | 0.097 | ` D` | ✗ |
| 45 | `re` | 0.072 | `ic` | ✗ |
| 46 | `el` | 0.072 | `rae` | ✗ |
| 47 | `as` | 0.221 | `os` | ✗ |

### `Amaryllidaceae` (5 tok, pattern=MMLRM, span=4)

| step | pos | token | conf |
|------|-----|-------|------|
| 12 | 16 | `ll` | 0.474 |
| 13 | 15 | `ary` | 0.999 |
| 14 | 14 | ` Am` | 0.999 |
| 15 | 18 | `aceae` | 0.874 |
| 16 | 17 | `id` | 0.997 |

Siblings at first unmask:

| pos | pred | conf | final | ok |
|-----|------|------|-------|----|
| 14 | `ary` | 0.427 | ` Am` | ✗ |
| 15 | `ll` | 0.466 | `ary` | ✗ |
| 17 | `aceae` | 0.258 | `id` | ✗ |
| 18 | ` .` | 0.128 | `aceae` | ✗ |

### `Ploughjogger` (5 tok, pattern=LMMRM, span=4)

| step | pos | token | conf |
|------|-----|-------|------|
| 1 | 1 | ` Pl` | 0.295 |
| 2 | 2 | `ough` | 0.990 |
| 3 | 4 | `og` | 0.881 |
| 4 | 5 | `ger` | 0.958 |
| 5 | 3 | `j` | 0.831 |

Siblings at first unmask:

| pos | pred | conf | final | ok |
|-----|------|------|-------|----|
| 2 | ` Farmer` | 0.126 | `ough` | ✗ |
| 3 | `man` | 0.158 | `j` | ✗ |
| 4 | ` )` | 0.147 | `og` | ✗ |
| 5 | ` )` | 0.080 | `ger` | ✗ |

---

## 14. Выводы

1. **Скорость:** 65.6% слов с span=1 (2-токенные за 2 соседних шага); 89.1% — все токены подряд без пропусков.
2. **Направление:** нет сильного left/right bias (≈45/46%); 2tok чуть чаще RL; 3+ tok — mixed patterns.
3. **Sibling pred:** модель частично «знает» остаток слова (60%); co-unmask при conf≥0.8 практически безопасен.
4. **First conf — главный предиктор:** низкий conf первого токена → siblings ещё не определены.
5. **Sibling vs non-sibling conf:** siblings увереннее (§10), следствие локальной когерентности слова.
6. **GSM8K vs WikiText:** math-генерация предсказуемее (выше acc, больше LR), WikiText разнообразнее.
7. **Tier A→B:** фильтр убирает 33.8% слов (infobox junk + word+punctuation артефакты).

---

## Приложение

```bash
cd quant_where_to_unmask && python generate_wikitext_report.py \
  --out wikitext_g64_multitoken_report.md
```

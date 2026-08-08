# Where vs What to unmask — полный отчёт

Анализ remasking в **LLaDA-8B-Base**: где расходятся FP16 и INT4, и как устроен unmask на разных задачах.

**Бенчмарки:**
- **Sudoku 4×4** — FP16 vs INT4, главный stress-test (900 test, traces)
- **WikiText g64** — FP16 only, multi-token слова (3328 traces) → [wikitext_g64_multitoken_report.md](wikitext_g64_multitoken_report.md)
- **GSM8K n256** — FP16 vs INT4, accuracy ~71% vs ~70%, traces 256

---

## Содержание

### Обзор
- [Сводка по всем бенчмаркам](#сводка-по-всем-бенчмаркам)

### Part I — Sudoku 4×4 (FP16 vs INT4)
1. [Резюме Sudoku](#1-резюме-sudoku)
2. [Setup и инфраструктура](#2-setup-и-инфраструктура)
3. [Данные](#3-данные)
4. [Accuracy: few-shot sweep](#4-accuracy-few-shot-sweep)
5. [Пересечение правильных ответов](#5-пересечение-правильных-ответов)
6. [Промпт, маски и формат генерации](#6-промпт-маски-и-формат-генерации)
7. [Что модель заполняет](#7-что-модель-заполняет)
8. [Where vs What (Sudoku)](#8-where-vs-what-sudoku)
9. [Порядок размаскирования](#9-порядок-размаскирования)
10. [Цифры в момент расхождения](#10-цифры-в-момент-расхождения)

### Part II — WikiText g64 (FP16)
11. [WikiText: where / what на уровне слов](#11-wikitext-where--what-на-уровне-слов)

### Part III — GSM8K (FP16 vs INT4)
12. [GSM8K: where / what divergence](#12-gsm8k-where--what-divergence)

### Общее
13. [Сравнение бенчмарков](#13-сравнение-бенчмарков)
14. [Артефакты и скрипты](#14-артефакты-и-скрипты)
15. [Примеры (Sudoku)](#15-примеры-sudoku)
16. [Открытые вопросы](#16-открытые-вопросы)

---

## Сводка по всем бенчмаркам

| | Sudoku 2-shot/900 | GSM8K n256 | WikiText g64 |
|---|-------------------|------------|--------------|
| **Quant** | FP16 vs INT4 | FP16 vs INT4 | FP16 only |
| **Accuracy gap** | **+12.5 pp** | **~0 pp** (71% vs 70%) | N/A (perplexity-style gen) |
| **where match / step** | **54.7%** | **12.5%** | N/A (single model) |
| **P(what \| where same)** | **98.1%** | **41.5%** | sibling pred 57.2% |
| **Первая развилка** | where, step **7** | where, step **3** | left/right ~45% each |
| **Identical unmask order** | 2% | **0%** | 77% consecutive (same model) |
| **Главный вывод** | INT4 ломает **порядок**, не цифры | FP16≈INT4 по ответам, но **разный schedule** | siblings pred==final 57%; co-unmask safe @conf≥0.8 |

**Sudoku** — единственный бенч, где gap в accuracy сопровождается «чистым» where-fork при сохранении what. **GSM8K** — ответы почти одинаковые (72% same answer), но unmask order diverges с step 3 и what тоже часто расходится. **WikiText** — детальный разбор where/what внутри одной модели (multi-token words).

---

## 1. Резюме (Sudoku)

**Главный вывод:** на Sudoku INT4 теряет **7–13 pp** accuracy относительно FP16 (vs ~1–2 pp на GSM8K). Механизм — не «путает цифры», а **расходится remasking schedule** (какую позицию unmask'ить следующей).

| Факт | Значение |
|------|----------|
| Лучший few-shot (900 test) | **2-shot**: FP16 **50.3%**, INT4 **37.8%** |
| Подвыборка 100 test завышает 2-shot | 73%/66% → 50%/38% на 900 |
| Пересечение правильных (2-shot/900) | **308** both, **145** FP16-only, **32** INT4-only |
| P(what same \| where same) | **~98%** |
| Первая развилка | **where** (позиция), median **шаг 7** из 24 |
| На fork-позициях pred цифры совпадает | **99.5%** |
| Полная карта pos→pred совпадает на fork | **~21–26%** |

**One-liner:** Sudoku — stress-test для INT4: 24 greedy remasking шага ломаются на **порядке unmask** (where), а не на **цифрах** (what).

---

## 2. Setup и инфраструктура

| Параметр | Значение |
|----------|----------|
| Model | `GSAI-ML/LLaDA-8B-Base` |
| Quant | FP16 vs INT4 (bitsandbytes) |
| gen_length / steps / block_length | 24 / 24 / 24 |
| remasking | `low_confidence` |
| k per step | 1 |
| mask_id | 126336 |
| Метрика | `exact_match` (Dream protocol) |
| Harness | `eval_llada.py` + `tasks/sudoku4/` |
| Launcher | `run_sudoku.sh` |
| GPUs | 2× (`2gpu`) |

Запуск:

```bash
# 2-shot, 900 test
bash run_sudoku.sh fp16 4x4 2shot large 2gpu
bash run_sudoku.sh int4 4x4 2shot large 2gpu

# 1-shot / 4-shot large
bash run_sudoku.sh fp16 4x4 1shot large 2gpu
bash run_sudoku.sh fp16 4x4 4shot large 2gpu
```

Задачи:

| Task YAML | Few-shot | Test N |
|-----------|----------|--------|
| `sudoku4_8shot` | 8 | 100 |
| `sudoku4_{1,2,4}shot` | 1/2/4 | 100 |
| `sudoku4_{1,2,4}shot_large` | 1/2/4 | 900 |

---

## 3. Данные

### Правильный файл

Dream paper использует **`sudoku_4x4_10.jsonl`**, не `_8`.

| Файл | Проблема |
|------|----------|
| `sudoku_4x4_8.jsonl` | few-shot memorization → ~3% accuracy |
| `sudoku_4x4_10.jsonl` | корректный: 8 few-shot header + 100 test (rows 8–107) |

### Large-сет (900 test)

`tasks/sudoku4/data/sudoku_4x4_combined.jsonl`:

- 908 строк = 8 header (few-shot из `_10`) + **900 уникальных** головоломок
- Источник: все 9 Dream-файлов `sudoku_4x4_{4..12}.jsonl`

### Сравнение с paper

| | Accuracy |
|---|----------|
| Dream paper, LLaDA, 8-shot/100 | ~**46%** |
| Dream 7B (другая модель) | ~81% |
| Наш 8-shot/100 | FP16 44%, INT4 42% |

### Головоломка

- Сетка 4×4: **8 open cells** (zeros) + **8 given clues** на каждый пример
- Распределение open cells в combined: uniform 4–12 zeros, **mean = 8**

---

## 4. Accuracy: few-shot sweep

### 100 test (`sudoku_4x4_10`, rows 8–107)

| Few-shot | FP16 | INT4 | Δ (FP16−INT4) |
|----------|------|------|----------------|
| 8-shot | 44% | 42% | +2 pp |
| 4-shot | 28% | 30% | −2 pp |
| **2-shot** | **73%** | **66%** | **+7 pp** |
| 1-shot | 49% | 35% | +14 pp |

### 900 test (`sudoku_4x4_combined`)

| Few-shot | FP16 | INT4 | Δ (FP16−INT4) |
|----------|------|------|----------------|
| 4-shot | 21.3% | 17.4% | +3.9 pp |
| 1-shot | 33.6% | 24.4% | +9.2 pp |
| **2-shot** | **50.3%** | **37.8%** | **+12.5 pp** |

### Выводы по few-shot

1. **100 test — лёгкая подвыборка.** 2-shot: 73% → 50% при переходе на 900.
2. **Sweet spot = 2-shot** по относительному ranking, но абсолютные цифры на 900 ниже.
3. **4-shot и 8-shot** хуже из-за memorization / длинного промпта.
4. **Paper-like ~46%** ближе к 8-shot/100, не к завышенному 2-shot/100.
5. **INT4 gap растёт** на большом сете: 2-shot 7 pp (100) → 12.5 pp (900).

---

## 5. Пересечение правильных ответов

Анализ exact-match множеств по checkpoint'ам.

| Config | FP16 | INT4 | Оба верны | FP16 only | INT4 only | Jaccard |
|--------|------|------|-----------|-----------|-----------|---------|
| 8-shot, 100 | 44 | 42 | 33 | 11 | 9 | 0.62 |
| 1-shot, 900 | 302 | 220 | 181 | 121 | 39 | 0.53 |
| **2-shot, 900** | **453** | **340** | **308** | **145** | **32** | **0.64** |
| 4-shot, 900 | 192 | 157 | 131 | 61 | 26 | 0.60 |

### Паттерны (2-shot / 900)

```
FP16 correct (453)          INT4 correct (340)
┌──────────────────────┐
│ FP16 only: 145       │
│   ┌──────────────┐   │
│   │ BOTH: 308    │   │  ← 68% FP16-успехов, 91% INT4-успехов
│   └──────────────┘   │
│      INT4 only: 32   │
└──────────────────────┘
```

1. **Overlap/min ≈ 80–91%** — большая часть INT4-успехов есть и у FP16 (308/340 = 90.6%).
2. **Разрыв = FP16-only** — INT4 теряет решения, почти не добавляет (32 INT4-only vs 145 FP16-only).
3. **Оба неправы, но часто одинаково** — 60–75% ошибок дают идентичный неверный ответ.
4. **Нет valid-but-wrong** — 0 случаев «валидное судоку, но не gold». Ошибки = невалидная сетка (копия input, обрыв, мусор).

---

## 6. Промпт, маски и формат генерации

### Структура промпта (2-shot, пример)

```
Fill the positions where the values are 0 in a 4x4 grid with digits 1-4 so that
each column, each row, and each of the four 2x4 subgrids ... Input:
1003
0300
3210
4132

Output:
  1423
2341
3214
4132


Input:
0302
0034
1423
3040

Output:
  4312
2134
1423
3241


Input:
0201
0024
3010
2143
Output:
 
```

- **Инструкция** + **2 few-shot** (Input/Output) + **test Input** + `Output:\n ` (пробел)
- Промпт ≈ **186 токенов**, **без масок**

### Где маски

```
[ prompt (186 tok, фиксирован) ] + [ gen block (24 tok, ВСЕ [MASK] ) ]
  позиции 0..185                    позиции 186..209
```

- **24 маски** на старте (`gen_length=24`, `steps=24`, k=1)
- Промпт **никогда** не маскируется
- К концу **~0 масок** (редко 1 остаётся в хвосте)

### Layout gen-блока (24 токена)

```
rel:  0    1 2 3 4    5    6 7 8 9   10   11 12 13 14  15   16 17 18 19  20  21  22  23
      sp   row1────    \n   row2────   \n   row3─────    \n   row4─────   \n  \n  \n  tail
```

| Слоты | Кол-во | Назначение |
|-------|--------|------------|
| Digit (pos 1–4, 6–9, 11–14, 16–19) | 16 | все 16 клеток сетки |
| Format (sp, `\n`, tail) | 8 | пробел, переносы, хвост |

### Хвост — зачем 24, а не 20

Gold-ответ = **20 токенов**: `' ' + 16 digits + 3 internal '\n'` (без trailing `\n` после row4).

`gen_length=24` — наследие Dream (`max_new_tokens=24`). Лишние **4 слота** (pos 20–23):

- На финале в ~92%: `('\n', '\n', '\n', [MASK])`
- **3–4 шага** из 24 тратятся на хвост, не на судоку
- INT4 на первой развилке **27%** случаев уходит в tail вместо digit-слота

---

## 7. Что модель заполняет

| | |
|---|---|
| Open cells в input | **8** (zeros) |
| Given clues | **8** |
| Digit-слотов в gen | **16** (вся сетка перегенерируется) |
| Unmask на digit-слоты | **16 из 24** шагов |
| Unmask на format | **8 из 24** шагов |
| Clues preserved в output | **94.5%** |
| Output с 16 цифрами | **80.1%** |
| Open cells с правильной цифрой | **62.7%** (из всех open slots) |
| Valid sudoku rate (2-shot FP16) | **50.3%** |

Модель **не заполняет только 8 пустых** — она генерирует полную 16-цифровую сетку + формат.

---

## 8. Where vs What (Sudoku)

Скрипт: `analyze_unmask_divergence.py`

| Понятие | Определение |
|---------|-------------|
| **where** | `unmasked[].pos_comp` — какую позицию снять (`topk(confidence, k=1)`) |
| **what** | `unmasked[].token_id` — какой токен (`argmax(logits)`) |

### 2-shot / 900

| Метрика | ALL | both_ok | fp16_only | both_wrong |
|---------|-----|---------|-----------|------------|
| where match / step | 54.7% | 60.4% | 51.0% | 52.2% |
| what match / step | 53.7% | 60.4% | 48.2% | 51.2% |
| **P(what same \| where same)** | **98.1%** | ~100% | ~94% | ~98% |
| top1 ranking agree | 54.6% | 60.3% | 51.1% | 52.0% |
| first divergence (median step) | 7 | 7 | 8 | 8 |

### Ключевые выводы

1. **Первая развилка — always where, never what.** Во всех fp16_only/int4_only первое расхождение = **разная позиция**, не «та же клетка, другая цифра».
2. **Where — узкое место INT4.** При совпадении позиции цифра совпадает в ~98%.
3. **Tight margins.** На первой развилке `boundary_margin` ≈ **0.03**; в **~80%** < 0.05.
4. **Не рандом.** В 61% шагов «чужая» позиция в собственном top-3 по confidence.
5. **fp16_only** — INT4 уходит в другую ветку на ~шаге 8 и не возвращается.

---

## 9. Порядок размаскирования

Скрипт: `analyze_unmask_order.py`

### Общая статистика (2-shot / 900)

| Метрика | ALL | both_ok | fp16_only |
|---------|-----|---------|-----------|
| Identical 24-step order | 2.0% | 4.2% | 0% |
| Shared prefix length | mean 8.6, med 7 | 8.8 | 8.5 |
| Kendall τ (24 pos) | 0.84 | 0.89 | 0.80 |
| Kendall τ (16 digits) | 0.75 | 0.86 | **0.67** |

### Фаза A: format (шаги 0–4)

**97.3%** обе модели:
```
step 0→4:  sp → nl1 → nl2 → nl3 → nl4
agreement: 100%, 99.4%, 98.4%, 99.0%, 100%
```

### Фаза B: digits + tail (шаги 5–23)

Per-step agreement падает:

| step | agree | phase |
|------|-------|-------|
| 5 | 79.9% | digit/tail |
| 7 | 55.6% | digit |
| 10 | 44.6% | digit |
| 15 | 30.2% | digit |
| 20 | 29.4% | tail |

### Digit-unmask order (k-th digit, k=0..15)

| k | position agree |
|---|----------------|
| 0 (первая цифра) | **81.3%** |
| 4 | 51.4% |
| 8 | 38.9% |
| 15 (последняя) | 41.0% |

Первая digit-ячейка: **row 3** в ~50% случаев (pos 11–14).

### Первая развилка — типы

| Тип | Доля | Смысл |
|-----|------|-------|
| **digit ↔ digit** | **61%** (550) | обе в digit-слоты, но **разные клетки** |
| FP16 digit → INT4 tail | 27% (245) | INT4 берёт `\n` вместо цифры |
| FP16 tail → INT4 digit | 8% (72) | |
| format ↔ format | 2% (15) | |

**digit ↔ digit = разные места, не разные цифры на одной клетке.** Случаев «same cell, different token» на первой развилке: **0**.

Top swaps на первой развилке:

| FP16 | INT4 | count |
|------|------|-------|
| r4c1 | tnl1 (tail) | 51 |
| r3c1 | tnl1 | 34 |
| r3c1 | r3c2 | 18 |

### fp16_only: каскад

| | digit agree |
|---|-------------|
| первые 8 digit-unmask | **55.6%** |
| последние 8 digit-unmask | **18.7%** |

Не ранний хаос — **поздний распад** порядка после fork на ~шаге 8.

---

## 10. Цифры в момент расхождения

На шаге первой развилки сравниваем `predicted_token_id` на ещё masked позициях.

| Метрика | Значение |
|---------|----------|
| FP16-chosen pos: same pred FP16 & INT4 | **99.5%** |
| INT4-chosen pos: same pred (digit-digit) | **99.6%** |
| All masked digit slots: mean fraction same pred | **86.6%** |
| **Full digit map identical** (pos→token) | **20.7%** (digit-digit: 25.6%) |

### Интерпретация

- **На fork-позициях** обе модели почти всегда согласны, **какая цифра** там должна быть.
- Спор — **какую клетку открыть первой** (порядок).
- **Не чистая перестановка** одного плана: полная карта pred совпадает только в ~¼ случаев; на ~13% других masked digit-слотов pred уже различается.

### Два сценария

**A. Только порядок (~26% digit-digit на fork):**
```
Обе: r4c3→'3', r4c4→'4', ...
FP16: сначала r4c4
INT4: сначала r4c3
```

**B. Порядок + частично разные pred (~74%):**
```
Fork-positions: pred совпадает (99.5%)
Другие masked cells: ~13% уже различаются
```

---

## 11. WikiText: where / what на уровне слов

**Полный отчёт:** [wikitext_g64_multitoken_report.md](wikitext_g64_multitoken_report.md)

FP16 only (нет FP16 vs INT4). Анализ remasking на **multi-token словах** — наш второй крупный where/what эксперiment.

### Setup

| Параметр | Значение |
|----------|----------|
| Model | LLaDA-8B-Base, **FP16** |
| Data | WikiText-103, prompt=48 tok, gen=**64**, steps=64, k=1 |
| Traces | **3328** (3201 после loop-фильтра) |
| Слова | 8408 lexical multi-token (Tier B), 2.63 слова/trace |

### Where = порядок unmask внутри слова

| Метрика | Значение |
|---------|----------|
| step_span median | **1** |
| full consecutive (gap=1) | **77.0%** |
| first unmask = left / right / middle | 44.7% / 46.3% / 9.0% |
| 2-tok LR / RL | 47.5% / 52.5% |

Модель чаще unmask'ит с **края** слова, редко из середины.

### What = sibling prediction

При первом unmask слова — `predicted_token_id` на **sibling** позициях (остальные токены слова ещё masked):

| Метрика | Значение |
|---------|----------|
| sibling pred == final | **57.2%** |
| all siblings ok (per word) | **53.2%** |
| co-unmask @ conf≥0.8 | **99.8%** accuracy |
| first_conf < 0.3 → sibling acc | **34%** |

### Sibling vs non-sibling conf

| | mean conf |
|---|-----------|
| Sibling (same word, masked) | **0.323** |
| Non-sibling (other masked) | **0.104** |

Sibling mean > non-sibling в **88%** слов.

### WikiText vs GSM8K (FP16, из того же отчёта)

| | WikiText g64 | GSM8K g256 |
|---|--------------|------------|
| sibling pred==final | 57.2% | **74.9%** |
| first conf mean | 0.512 | **0.792** |

---

## 12. GSM8K: where / what divergence

**FP16 vs INT4**, n=**256**, gen=steps=block=**256**, 5-shot.

### Accuracy

| | FP16 | INT4 |
|---|------|------|
| flexible-extract | 66.8% | 68.8% |
| strict-match | **71.1%** | 69.5% |

Gap ~0 pp (vs Sudoku +12.5 pp).

### Trace analysis (256 paired by doc_id)

| Метрика | GSM8K | Sudoku 2-shot/900 |
|---------|-------|-------------------|
| where match / step | **12.5%** | 54.7% |
| P(what \| where same) | **41.5%** | 98.1% |
| prefix until fork (median) | **3** | 7 |
| identical full order | **0%** | 2% |
| same final answer | **71.9%** | — |

**Интерпретация:**
- Fork на **шаге 3** — почти сразу
- When where совпадает, **what часто нет** (41% vs 98%)
- Но **72% same answer** — 256 шагов дают самокоррекцию

Checkpoints: `checkpoints/results_gsm8k_{fp16,int4}_n256_fast_2gpu/`

---

## 13. Сравнение бенчмарков

| | accuracy gap | where agree | what\|where | steps | fork |
|---|-------------|-------------|-------------|-------|------|
| Sudoku 2-shot/900 | **+12.5 pp** | 54.7% | **98%** | 24 | step 7 |
| GSM8K n256 | ~0 | **12.5%** | **41%** | 256 | step 3 |
| WikiText g64 | N/A | left≈right | sibling 57% | 64 | — |

**Sudoku:** короткий schedule → необратимый where-fork. **GSM8K:** ранний fork, длинная gen спасает accuracy. **WikiText:** what=siblings, where=left/right/middle.

---

## 14. Артефакты и скрипты

| Benchmark | Traces | Отчёт / скрипты |
|-----------|--------|-----------------|
| Sudoku | 900/run, FP16+INT4 | `analyze_unmask_divergence.py`, `analyze_unmask_order.py` |
| WikiText | 3328, FP16 | [wikitext_g64_multitoken_report.md](wikitext_g64_multitoken_report.md), `generate_wikitext_report.py` |
| GSM8K | 256, FP16+INT4 | `checkpoints/results_gsm8k_*_n256_fast_2gpu/` |

```bash
python compare_fp16_int4_sudoku4.py
python analyze_unmask_divergence.py
python analyze_unmask_order.py
python generate_wikitext_report.py
```

---

## 15. Примеры (Sudoku)

### Identical order (both_ok #22)

```
FP16/INT4: sp nl1 nl2 nl3 nl4 r3c1 r4c1 r1c2 r1c1 r4c2 r4c3 r1c4 r1c3 r3c4 r3c3 r2c4 r4c4 r3c2 r2c1 r2c2 r2c3 tnl1 tnl2 tnl3
```

### fp16_only #6 (fork@8)

```
FP16: ... r3c4 [r2c4] ...
INT4: ... r3c4 [r4c1] ...
```

### Order differs, digit map agrees (#0)

```
Fork: FP16→r4c4, INT4→r4c3 — digit pred map identical
```

---

## 16. Открытые вопросы

### Sudoku
- [ ] 8-shot на 900
- [ ] gen_length=20
- [ ] Remasking sweep

### GSM8K / WikiText
- [ ] GSM8K — полный divergence report (outcome categories)
- [ ] WikiText INT4 traces
- [ ] Unified analyze script для любого checkpoint pair

---

*Последнее обновление: 2026-07-14. Sudoku 2-shot/900. WikiText 3328 traces. GSM8K n256 FP16 vs INT4.*

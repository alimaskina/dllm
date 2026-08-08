# Hidden dynamics: момент открытия каждого токена слова

Слов: **250**, событий открытия: **540**, checkpoint: `checkpoints/results_wikitext_fp16_g64_n256`
**Фильтр:** strict lexical (`is_lexical`, без word+punctuation)

## Методология

Для каждого multi-token слова сортируем токены по времени unmask (step, pos).
Для **каждого** открытия отдельно:
1. Восстанавливаем sequence `x` **на конце этого step**
2. Forward → hidden states (последний слой)
3. Считаем метрики **только в этот момент** (без усреднения по другим шагам)

**open_idx=1** — момент открытия первого токена слова
**open_idx=2** — момент открытия второго
**open_idx=3** — третьего (если есть)

---

## 1. Cosine similarity hidden'ов (на конкретном step открытия)

Контроль **честный**: сравниваем `masked sibling ↔ other masked` (оба ещё `[MASK]` на этом шаге).
НЕ сравниваем opened↔other — там REAL vs MASK, это разные режимы.

| Момент | n | within pairs | opened↔masked sib | masked sib↔other | opened↔open | Δ(opened↔sib − sib↔other) |
|--------|---|--------------|-------------------|------------------|-------------|------------------------------|
| открыт 1-й | 250 | 0.748 | 0.748 | 0.740 | — | +0.008 |
| открыт 2-й | 250 | 0.480 | 0.609 | 0.609 | 0.473 | +0.001 |
| открыт 3-й | 33 | 0.549 | 0.642 | 0.597 | 0.513 | +0.044 |
| открыт 4-й | 7 | 0.617 | — | — | 0.594 | — |

**Как читать:**
- Всё на **одном step** — момент открытия k-го токена слова
- `opened↔masked sib` — hidden только что открытого (REAL) vs hidden sibling ещё `[MASK]`
- `masked sib↔other` — **контроль**: hidden sibling `[MASK]` vs hidden другой `[MASK]` позиции (вне слова)
- `Δ(opened↔sib − sib↔other)` — насколько opened ближе к sibling, чем sibling к random masked
- `opened↔open` — REAL vs REAL (уже открытые куски слова)

---

## 2. Logit lens с hidden только что открытого токена

Проецируем hidden **открытого на этом шаге** токена → предсказываем still-masked siblings.

| Момент | n | sib top-1 | sib top-10 | median rank | native top1 на sib-позиции |
|--------|---|-----------|------------|-------------|----------------------------|
| открыт 1-й | 250 | 0.0% | 27.5% | 47 | 64.4% |
| открыт 2-й | 250 | 0.0% | 12.1% | 72 | 95.5% |
| открыт 3-й | 33 | 0.0% | 14.3% | 204 | 85.7% |
| открыт 4-й | 7 | — | — | — | — |

- **native top1 на sib-позиции** — logit lens с hidden masked sibling (как модель обычно предсказывает)

---

## 3. Контекст момента

| Момент | mean mask_ratio | mean opened conf | mean masked sibs left |
|--------|-----------------|------------------|-----------------------|
| открыт 1-й | 0.528 | 0.493 | 1.160 |
| открыт 2-й | 0.509 | 0.926 | 0.160 |
| открыт 3-й | 0.530 | 0.946 | 0.212 |
| открыт 4-й | 0.451 | 0.996 | 0.000 |

---

## 4. Разбивка по длине слова (1-е открытие)

| ntok | n | opened↔masked sib | masked sib↔other | Δ paired |
|------|---|-------------------|------------------|----------|
| 2 | 217 | 0.738 | 0.735 | +0.003 |
| 3 | 26 | 0.789 | 0.768 | +0.021 |
| 4 | 7 | 0.918 | 0.810 | +0.108 |

## 5. Примеры по моментам

### Открытие 1-го токена — лучшие (низкий rank sibling)

- `Scientologists`: открыли `ologists` (conf=0.38), lens_top1=`ologists`, sibling ` Scient` rank=2
- `battalion`: открыли `alion` (conf=0.37), lens_top1=`alion`, sibling ` batt` rank=2
- `australian`: открыли `ustralian` (conf=0.57), lens_top1=`ustralian`, sibling ` a` rank=2

### Открытие 1-го токена — худшие

- `Musicians`: открыли ` Music`, sibling `ians` rank=6773
- `Peavey`: открыли ` Pe`, sibling `ave` rank=10158
- `Peavey`: открыли ` Pe`, sibling `ave` rank=10017
### Открытие 2-го токена — лучшие (низкий rank sibling)

- `progeria`: открыли `ger` (conf=1.00), lens_top1=`ger`, sibling ` pro` rank=5
- `vietnam`: открыли ` v` (conf=1.00), lens_top1=` v`, sibling `iet` rank=5
- `turrets`: открыли `ts` (conf=1.00), lens_top1=`ts`, sibling ` t` rank=7

### Открытие 2-го токена — худшие

- `Scientology`: открыли `ology`, sibling `` rank=—
- `Toland`: открыли ` Tol`, sibling `` rank=—
- `Godrich`: открыли ` God`, sibling `` rank=—
### Открытие 3-го токена — лучшие (низкий rank sibling)

- `Shikamaru`: открыли `ik` (conf=1.00), lens_top1=`ik`, sibling ` Sh` rank=10
- `wolfensohn`: открыли `hn` (conf=1.00), lens_top1=`hn`, sibling `en` rank=15
- `Shikamaru`: открыли `amar` (conf=1.00), lens_top1=`amar`, sibling `u` rank=21

### Открытие 3-го токена — худшие

- `Pocono`: открыли `ocon`, sibling `` rank=—
- `vietnam`: открыли `iet`, sibling `` rank=—
- `Mengele`: открыли `enge`, sibling `` rank=—

---

## 6. Динамика: как меняется при каждом следующем открытии

- opened↔masked sib: 1-е **0.748** → 2-е **0.609**
- masked sib↔other (контроль): 1-е **0.740** → 2-е **0.609**
- Δ paired: 1-е **+0.008** → 2-е **+0.001**
- logit lens sib top-10: 1-е **27.5%** → 2-е **12.1%**
- 3-е открытие: within-word cos **0.549**, n=33

---

## 7. Финальное состояние (все токены слова REAL)

После unmask **последнего** токена слова: cos между hidden всех позиций (REAL↔REAL).

| Метрика | Value |
|---------|-------|
| mean cos(within word, all REAL) | 0.479 |
| median | 0.480 |
| logit lens: top1 sibling-токена с hidden каждой позиции | 0.0% |

Сравнение: 1-е открытие within mean **0.748** → финал **0.479**.


**Интерпретация динамики:** при каждом следующем открытии в слове остаётся меньше masked siblings, контекст слова обогащается открытыми токенами — ожидаем рост similarity и улучшение предсказания оставшихся кусков.

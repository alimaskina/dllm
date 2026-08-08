# Multilingual Diffusion PoC — полный отчёт первого прогона

**Дата:** 2026-08-07 (основной прогон); follow-up batch v2/v3 — 2026-08-08  
**Корпус:** OPUS-100 validation (до 5000 предложений / язык)  
**Языки:** EN, DE, RU, TR, FI, ZH, KO  
**GPU:** 8× A100 80GB (для прогона использовались свободные 0, 1; 2, 5, 6, 7 были заняты)

План экспериментов: [`multilingual_diffusion_feasibility.md`](../multilingual_diffusion_feasibility.md)

---

## 1. Контекст: какую гипотезу проверяли

Центральная цепочка из feasibility-плана:

```
фрагментация → objective mismatch → inference OOD states → errors
```

**Идея:** языки вроде RU/FI/DE режут слова на больше BPE-токенов (k = 3, 4, 5…). При *independent token masking* на обучении модель чаще видит **частично раскрытые** слова (один subtoken уже открыт, sibling ещё в маске), а состояния «всё multi-token слово целиком в маске при низком шуме» — редки (вероятность t^k).

При inference **confidence/margin decoding** может оставлять целые multi-token «острова» unresolved дольше, чем training corruption предсказывает — редкие OOD-состояния, на которых модель мало училась.

PoC проверял links **#1–#4** плюс sanity check на языковую компетентность.

---

## 2. Инфраструктура

В `multi_language/` собран минимальный пайплайн:

| Скрипт | Звено | Что делает |
|--------|-------|------------|
| `tokenization_audit.py` | #1 | P(k=1), P(k=2), P(k=3), P(k≥4) по языкам |
| `training_objective_audit.py` | #2 | Аналитика t^k по эмпирическому распределению k |
| `oracle_trajectory_random.py` | #3 | Negative control: random reveal order |
| `oracle_trajectory_model.py` | #3b | Oracle trajectories + confidence scheduling |
| `free_decoding_trajectory.py` | #3c | Free decoding: model argmax + prompt generation |
| `policy_sweep.py` | #6 | Oracle mismatch: confidence / margin / random / L2R |
| `weighted_exposure_audit.py` | #1×#6 | Weighted OOD exposure: f_k × P_traj + counterfactuals |
| `prompt_gen_eval.py` | #3→errors | Prompt gen: tok_acc, char edit vs mismatch/exposure |
| `codebook_segmentation_audit.py` | #5 | DLM vs AR on controlled k-token codes (+ few-shot, oracle) |
| `fast_dllm_trajectory.py` | #6 | Native block MDM oracle trajectories for Fast-dLLM |
| `fast_dllm_policy_sweep.py` | #6 | Policy sweep via block MDM adapter |
| `run_next_experiments.sh` | infra | Batch: codebook + Dream/Fast-dLLM policy sweep |
| `run_batch_v3.sh` | infra | Batch: codebook v3 + Fast-dLLM native + short prompt gen |
| `nll_oracle_audit.py` | #4 | Gold NLL на oracle trajectories |
| `trajectory_utils.py` | infra | Shared trajectory / NLL helpers |
| `language_competence_mcq.py` | sanity | MCQ: модель знает язык или нет |
| `model_loader.py` | infra | Загрузка LLaDA под transformers 5.8 (патч remote code) |
| `word_utils.py`, `data_sources.py` | infra | Сегментация слов, загрузка OPUS-100 |
| `run_poc.sh` | infra | Оркестрация всех шагов |

**Токенizers:**
- **LLaDA** — `GSAI-ML/LLaDA-8B-Base` (vocab 126k)
- **Qwen2.5** — `Qwen/Qwen2.5-7B` (vocab 151k; тот же family, что Dream / Fast-dLLM)

**Артефакты:** `multi_language/results/*.json`, логи `*.log`

**Детальная процедура каждого эксперимента (скрипты, команды, шаги):** см. **§2b**.

**Перезапуск:** `bash multi_language/run_poc.sh`

---

## 2b. Детальное описание процедуры по каждому эксперименту

Ниже — что именно делалось в коде и при запуске: скрипты, команды, фильтры, гиперпараметры, артефакты. Порядок соответствует хронологии прогона PoC + follow-up batch (v2/v3).

### Эксп. 1 — Tokenization audit (link #1)

**Скрипт:** `tokenization_audit.py`  
**Команда:** `python tokenization_audit.py --tokenizer all_unique --max-samples 5000 --out results/tokenization_audit.json`  
**GPU:** CPU

**Процедура:**
1. Загрузить OPUS-100 validation (`data_sources.load_opus_texts`) — до 5000 предложений на язык (EN, DE, RU, TR, FI, ZH, KO).
2. Для каждого языка сегментировать «слова» через `word_utils.segment_words`:
   - EN/DE/RU/TR/FI — regex `\S+` (whitespace tokenization);
   - ZH — один Unicode-иероглиф = одно слово;
   - KO — один слог Hangul (U+AC00–U+D7A3) = одно слово.
3. Для каждого слова посчитать **k** = `len(tokenizer.encode(word, add_special_tokens=False))`, минимум 1.
4. Агрегировать: P(k=1), P(k=2), P(k=3), P(k≥4), mean/median k, strata по длине слова в символах (1–3, 4–6, 7–10, 11+).
5. Прогнать для tokenizer presets: **LLaDA**, **Qwen2.5-7B**, **Dream**, **Fast-dLLM** (`TOKENIZER_PRESETS`).

**Выход:** `results/tokenization_audit.json`

---

### Эксп. 2 — Training objective audit (link #2)

**Скрипт:** `training_objective_audit.py`  
**Команда:** `python training_objective_audit.py --tokenizer llada|qwen --max-samples 5000 --out results/objective_audit_{tok}.json`  
**GPU:** CPU

**Процедура:**
1. Взять эмпирическое распределение k по словам из эксп. 1 (`word_k_distribution`).
2. Для сетки mask ratio t ∈ {0.1, 0.2, …, 0.9} вычислить аналитически:
   - P(all k siblings masked) = **t^k** (weighted by f_k);
   - P(at least one sibling visible) = **1 − t^k**.
3. Сравнить языки относительно EN @ t=0.3.

**Выход:** `objective_audit_llada.json`, `objective_audit_qwen.json`

---

### Эксп. 3a — Oracle random trajectory (sanity check)

**Скрипт:** `oracle_trajectory_random.py`  
**Команда:** `python oracle_trajectory_random.py --tokenizer llada --max-samples 500 --out results/oracle_random_llada.json`  
**GPU:** CPU (симуляция без модели)

**Процедура:**
1. Токенизировать предложения, найти multi-token слова (k≥2).
2. Симулировать reverse trajectory: на каждом шаге случайно выбирать позиции для reveal (uniform over masked).
3. Записывать triplets `(k, t, whole_word_unresolved)` на каждом шаге.
4. Агрегировать P_traj(k,t) и Mismatch = P_traj / t^k.

**Ожидание:** Mismatch ≈ 1.0 — подтверждает, что метрика не даёт ложный сигнал при random policy.

**Выход:** `oracle_random_llada.json`

---

### Эксп. 3b — Oracle confidence trajectory (link #3, основной)

**Скрипт:** `oracle_trajectory_model.py` (+ shared `trajectory_utils.confidence_trajectory`)  
**Команда:** `CUDA_VISIBLE_DEVICES=1 python oracle_trajectory_model.py --model llada --max-samples 100 --remasking low_confidence --device cuda:0 --out results/oracle_confidence_llada.json`  
**GPU:** 1

**Модель:** LLaDA-8B-Base, mask_id=126336, bf16/fp16 через `model_loader.load_llada` (патч remote code для transformers 5.8).

**Процедура на одно предложение:**
1. `tokenize_sentence(text, lang, tokenizer)` → token ids + список позиций каждого слова.
2. Оставить только multi-token слова (k≥2) для наблюдений.
3. Фильтр: 8 ≤ len(ids) ≤ 128, хотя бы одно multi-token слово.
4. Инициализация: x = all [MASK], masked_set = все позиции.
5. **32 шага** denoising (`get_num_transfer_tokens` из `quant_where_to_unmask/generate.py` — сколько токенов открыть на шаг).
6. На каждом шаге **до reveal**:
   - t = |masked| / n;
   - для каждого multi-token слова записать (k, t, unresolved=все sibling positions ∈ masked_set).
7. Forward LLaDA → logits → confidence:
   - `low_confidence`: conf = softmax prob of argmax token;
   - `topk_margin`: top1 − top2 prob.
8. Выбрать top-k_reveal позиций с max confidence среди masked.
9. **Oracle reveal:** x[pos] = gold[pos] (не argmax модели).
10. Повторить до конца steps или пока masked_set пуст.

**Агрегация:** `aggregate_obs` → P_traj(k,t); `compute_mismatch` → P_traj / t^k по bins t ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 0.9}.

**Языки:** EN, RU, DE, FI, ZH (~68–72 предложений / язык после фильтра).

**Выход:** `oracle_confidence_llada.json`

---

### Эксп. 3c — Free decoding trajectory

**Скрипт:** `free_decoding_trajectory.py`  
**Команда:** `CUDA_VISIBLE_DEVICES=1 python free_decoding_trajectory.py --setup both --max-samples 80 --device cuda:0 --out results/free_decoding_llada.json`  
**GPU:** 1

**Две постановки:**

| Setup | Что делаем |
|-------|------------|
| **full** | То же, что 3b, но `oracle=False`: reveal **model argmax**; ошибки накапливаются. Дополнительно tok_acc vs gold. |
| **prompt** | 25% gold prefix (`prefix_frac=0.25`), остаток через `generate()` из `quant_where_to_unmask/generate.py` с `record_trace=True`. Trajectory obs из trace unmask events (`observations_from_generate_trace`). |

**Фильтры prompt:** gen_length кратно block_length, min gen_length ≥ 8.

**Выход:** `free_decoding_llada.json` (оба setup + tok_acc)

---

### Эксп. 5c — Policy sweep (link #6, LLaDA)

**Скрипт:** `policy_sweep.py`  
**Команда (типичная):** `python policy_sweep.py --model llada --policies low_confidence,topk_margin,random,l2r --max-samples 60 --device cuda:0 --out results/policy_sweep_llada.json`

**Процедура:** повтор эксп. 3b для каждой policy:
- `low_confidence`, `topk_margin`, `random`, `l2r` (left-to-right: conf = −position index);
- oracle=True, 32 steps, те же фильтры предложений;
- EN/RU/DE/FI.

**Выход:** `policy_sweep_llada.json`

---

### Эксп. 5d — Weighted exposure audit

**Скрипт:** `weighted_exposure_audit.py`  
**Команда:** `python weighted_exposure_audit.py --policy-json results/policy_sweep_llada.json --out results/weighted_exposure_llada.json`  
**GPU:** CPU (post-hoc аналитика)

**Процедура:**
1. Загрузить f_k из OPUS + LLaDA tokenizer (`word_k_distribution`, до 5000 sent/lang).
2. Загрузить P_traj(k,t) из policy sweep (policy=`low_confidence`, langs EN/RU/DE/FI).
3. **Pooled P_traj:** усреднить P_traj по 4 языкам (убрать шум per-lang trajectories).
4. Вычислить @ t=0.3:
   - **Corpus exposure:** E = Σ_{k≥2} f_k · P_traj(k,t);
   - **Token-weighted:** E_tok = Σ_{k≥2} (k·f_k/E[k]) · P_traj(k,t);
   - **Training baseline:** Σ f_k · t^k;
   - **Excess:** E − training baseline.
5. **Counterfactuals:**
   - E(f_RU, P_pool) / E(f_EN, P_pool) — чистый fragmentation factor;
   - E(f_RU, P_RU) / E(f_RU, P_EN) — lang gap при фиксированном f.

**Выход:** `weighted_exposure_llada.json`

---

### Эксп. 5e — Prompt generation eval (link #3→errors)

**Скрипт:** `prompt_gen_eval.py`  
**Команды:**
```bash
# Полный OPUS
python prompt_gen_eval.py --max-samples 80 --device cuda:0 \
  --out results/prompt_gen_eval_llada.json

# Короткие тексты (follow-up)
python prompt_gen_eval.py --max-samples 500 --min-tokens 12 --max-tokens 48 \
  --out results/prompt_gen_eval_llada_short.json
```

**Процедура на предложение:**
1. Фильтр длины (short: 12–48 subtokens после tokenize).
2. 25% gold prefix, `generate()` на remainder (LLaDA, low_confidence, 32 steps, mask_id=126336).
3. Метрики: tok_acc, normalized Levenshtein char edit, exact match.
4. Параллельно собрать trajectory obs → mismatch(k,t) как в prompt setup 3c.
5. Корреляция Pearson: token-weighted exposure (§5d) vs char_edit / tok_acc по 4 языкам.

**Выход:** `prompt_gen_eval_llada.json`, `prompt_gen_eval_llada_short.json`

---

### Эксп. 4 — Gold NLL on oracle trajectories (link #4)

**Скрипт:** `nll_oracle_audit.py` (+ `nll_analysis.py` для strata)  
**Команда:** `CUDA_VISIBLE_DEVICES=0 python nll_oracle_audit.py --max-samples 80 --device cuda:0 --out results/nll_oracle_llada.json`

**Процедура:**
1. Oracle confidence trajectory (`confidence_trajectory`, oracle=True, collect_nll=True).
2. **Перед каждым reveal** для каждой ещё masked позиции:
   - NLL = −log p(gold_token | context);
   - метка `whole_word_unresolved` если все sibling subtokens слова ещё в маске (k≥2).
3. Strata через `nll_analysis.enrich_records`:
   - by (k, t) bins;
   - by char length bucket;
   - by word frequency quartile (log-freq из OPUS corpus).
4. Сравнить mean NLL: whole-unresolved vs partial (хотя бы один sibling открыт).

**Выход:** `nll_oracle_llada.json` (aggregate + by_k_t strata)

---

### Эксп. 5 — Language competence MCQ (sanity)

**Скрипт:** `language_competence_mcq.py`  
**Команда:** `CUDA_VISIBLE_DEVICES=0 python language_competence_mcq.py --model qwen15 --n-per-lang 50 --device cuda:0 --out results/competence_qwen15_mcq.json`

**Процедура:**
1. Из OPUS-100: 4 перевода одной исходной фразы (1 gold + 3 distractors из того же корпуса).
2. Prompt: «Which translation is most fluent? A/B/C/D».
3. Qwen2.5-1.5B greedy decode одной буквы.
4. ~50 примеров / язык, 7 языков.

**Выход:** `competence_qwen15_mcq.json`

---

### Эксп. 7 — Dream policy sweep (репликация #3/#6)

**Скрипт:** `policy_sweep.py` + расширенный `model_loader.py`  
**Команда (batch):** `CUDA_VISIBLE_DEVICES=3 python policy_sweep.py --model dream --policies low_confidence,topk_margin,random --max-samples 60 --device cuda:0 --out results/policy_sweep_dream.json`

**Особенности загрузки Dream:**
- `load_dlm("dream")` → AutoModel + trust_remote_code;
- Патчи: `ROPE_INIT_FUNCTIONS['default']`, `DreamGenerationConfig.validate`, `all_tied_weights_keys`;
- **AR-shift logits:** `logits = cat([logits[:,:1], logits[:,:-1]], dim=1)` (`ar_shift=True`);
- mask_id = **151666**.

**Процедура:** идентична эксп. 5c, модель Dream-v0-Base-7B (Qwen tokenizer family).

**Выход:** `policy_sweep_dream.json`

---

### Эксп. 8a — Fast-dLLM policy sweep (full-sequence, неудачная репликация)

**Скрипт:** `policy_sweep.py`  
**Команда:** `CUDA_VISIBLE_DEVICES=5 python policy_sweep.py --model fast_dllm --policies low_confidence,topk_margin,random --max-samples 60 --out results/policy_sweep_fast_dllm.json`

**Модель:** Fast-dLLM v2 7B, mask_id=**151665**, loader=`AutoModelForCausalLM`, **без** AR-shift.

**Проблема:** использовался тот же full-sequence `confidence_trajectory`, что для LLaDA. Fast-dLLM обучен на **block MDM** — forward на полной последовательности не отражает inference → mismatch ≈ 1, inconclusive.

**Выход:** `policy_sweep_fast_dllm.json`

---

### Эксп. 8b — Fast-dLLM native block MDM adapter

**Скрипты:** `fast_dllm_trajectory.py` (новый), `fast_dllm_policy_sweep.py`  
**Команда:** `CUDA_VISIBLE_DEVICES=1 python fast_dllm_policy_sweep.py --max-samples 60 --block-size 128 --device cuda:0 --out results/policy_sweep_fast_dllm_native.json`

**Процедура (PoC single-block MDM):**
1. Предложение len 8–128 subtokens, pad до block_size=128 mask-токенами.
2. Старт: **все позиции [MASK]** (151665).
3. Цикл до unmask всех n content positions:
   - forward `model(input_ids=x[:,:128], block_size=128, use_cache=False)`;
   - AR-shift logits;
   - confidence schedule (low_confidence / random / topk_margin);
   - unmask **1 позицию** с max confidence (gradual reveal);
   - oracle reveal: x[pos] = gold[pos].
4. Запись (k,t,unresolved) для multi-token слов на каждом шаге.
5. **Багfix:** не трогать `model.model.bd_size` — ломает последующие forward passes.

**Ограничение:** 1 tok/step → основная масса obs при t≈0.9+, не t=0.3; magnitude ниже LLaDA/Dream.

**Выход:** `policy_sweep_fast_dllm_native.json`

---

### Эксп. 9 — Codebook DLM vs AR (link #5)

**Скрипт:** `codebook_segmentation_audit.py`  
**Команды:**
```bash
# v2 zero-shot
CUDA_VISIBLE_DEVICES=0 python codebook_segmentation_audit.py \
  --dlm dream --ar qwen --n-trials 200 --out results/codebook_segmentation_dream_v2.json

# v3 few-shot + oracle + AR teacher
CUDA_VISIBLE_DEVICES=0 python codebook_segmentation_audit.py \
  --dlm dream --ar qwen --n-trials 200 --fewshot 2 --compare-oracle \
  --out results/codebook_segmentation_dream_v3.json
```

**Подготовка кодов:**
1. Из OPUS EN собрать слова, которые tokenize ровно в k subtokens (k=1,2,4) под **Dream/Qwen tokenizer**.
2. Дедупликация по tuple(token ids).

**Задача на trial:**
1. Sample random code (list of k token ids).
2. **Prompt:** few-shot v3 — 2 in-context примера «Record the code:\n{decoded word}\n…» + target code appended.
3. **DLM (Dream):** полностью замаскировать code span, denoise 32 steps low_confidence (`confidence_trajectory`, ar_shift=True). Free mode: reveal argmax; oracle mode: reveal gold.
4. **AR greedy (Qwen2.5-7B):** autoregressive generate k tokens после prompt.
5. **AR teacher-forced (v3):** для каждого code token — predict given gold prefix (upper bound).
6. Метрика: per-token accuracy code span.

**Выход:** `codebook_segmentation_dream_v2.json`, `codebook_segmentation_dream_v3.json`

---

### Инфраструктура загрузки моделей (`model_loader.py`)

Создан единый loader для three DLM presets:

| preset | model_id | mask_id | ar_shift | loader |
|--------|----------|---------|----------|--------|
| llada | GSAI-ML/LLaDA-8B-Base | 126336 | no | custom LLaDA |
| dream | Dream-org/Dream-v0-Base-7B | 151666 | **yes** | AutoModel |
| fast_dllm | Efficient-Large-Model/Fast_dLLM_v2_7B | 151665 | no | CausalLM |

Патчи для transformers 5.8 / remote code: RoPE default, tied weights, Dream generation config, LLaDA tie_weights kwargs.

---

### Batch-оркестрация

| Скрипт | Что запускает |
|--------|---------------|
| `run_poc.sh` | Steps 1–4 первого прогона (tokenization → objective → oracle → free → NLL → MCQ) |
| `run_next_experiments.sh` | Codebook v2 + Dream sweep + Fast-dLLM full-seq sweep (GPU 0, 3, 5) |
| `run_batch_v3.sh` | Codebook v3 + Fast-dLLM native + prompt gen short (GPU 0, 1) |

**GPU в follow-up:** 0 — codebook; 1 — Fast-dLLM native → prompt short (цепочка); 3 — Dream sweep; 5 — Fast-dLLM full-seq.

---

### Метод

Для каждого «слова» в тексте считаем **k** = число subtoken'ов под tokenizer модели (без special tokens).

**Сегментация слов (PoC-уровень):**
- EN, DE, RU, TR, FI — whitespace (`\S+`)
- ZH — один иероглиф = одно слово
- KO — один слог Hangul = одно слово

Корпус: OPUS-100 validation, до 5000 предложений на язык.

### Результаты — LLaDA tokenizer

| lang | n_words | k=1 | k=2 | k=3 | k≥4 | mean_k |
|------|---------|-----|-----|-----|-----|--------|
| en   | 27397   | 65% | 23% | 7%  | 4%  | 1.59   |
| de   | 23279   | 32% | 28% | 18% | 22% | 2.59   |
| ru   | 23635   | 16% | 13% | 15% | **56%** | **3.95** |
| tr   | 10887   | 14% | 24% | 25% | 37% | 3.28   |
| fi   | 13358   | 13% | 23% | 25% | 39% | 3.28   |
| zh   | 65627   | ~100% | 0.2% | — | — | 1.00 |
| ko   | 26658   | 48% | 37% | 16% | 0%  | 1.68   |

### Результаты — Qwen2.5 tokenizer (Dream / Fast-dLLM family)

| lang | k=1 | k≥4 | mean_k |
|------|-----|-----|--------|
| en   | 66% | 4%  | 1.59   |
| de   | 35% | 17% | 2.40   |
| ru   | 22% | 32% | 2.97   |
| tr   | 15% | 28% | 2.88   |
| fi   | 13% | 38% | 3.25   |
| zh   | ~100% | 0% | 1.00 |
| ko   | **99%** | 0% | 1.02 |

### Выводы по link #1

- **GO для DE/RU/TR/FI:** фрагментация существенно выше, чем у EN. RU на LLaDA tokenizer — экстремальный случай: **56% слов имеют k≥4**, median k = 4.
- На Qwen tokenizer картина мягче (RU k≥4 ≈ 32%), но тренд тот же.
- **Оговорки по ZH/KO:** текущая сегментация занижает фрагментацию. ZH почти всегда k=1 (нужен jieba или словарная сегментация). KO на Qwen tokenizer ≈ 99% k=1 (syllable-level слова + Qwen BPE).

**Статус: GO** — для fusional/agglutinative языков различия реальны и большие.

---

## 4. Эксперимент 2 — training objective (link #2)

### Метод

Для стандартного **independent token masking** при mask ratio t:

- P(все k sibling subtokens замаскированы) = **t^k**
- P(хотя бы один sibling виден) = **1 − t^k**

Агрегируем по эмпирическому распределению k в каждом языке (из эксп. 1).

### Результаты при t = 0.3 (низкий шум)

| lang | P(all siblings masked) | vs EN |
|------|------------------------|-------|
| en   | 0.219                  | 1.00× |
| de   | 0.127                  | 0.58× |
| ru   | 0.066                  | **0.30×** |
| tr   | 0.072                  | 0.33× |
| fi   | 0.067                  | 0.31× |
| zh   | 0.299                  | 1.37× |
| ko   | 0.180                  | 0.82× |

### Интерпретация

**Кажущийся парадox:** fragmented языки получают **меньше** fully-unresolved whole-word states на training, потому что при большом k вероятность t^k быстро падает.

**Но одновременно:** P(хотя бы один sibling виден) = 1 − t^k **выше** для RU/FI/TR (~93% vs ~78% у EN). Training curriculum другой:
- модель **чаще** доучивает subtoken при частично открытом слове;
- **реже** видит всё слово целиком в маске при низком шуме.

Mismatch на inference (link #3) — **отдельный механизм**: decoding policy может создавать whole-word holes **чаще**, чем t^k, даже если на training они были редки.

**Статус: GO** — objective действительно по-разному «видит» fragmented языки.

---

## 5. Эксперимент 3 — train–inference mismatch (link #3) ⭐

### Метод — oracle reverse trajectories

1. Берём gold-предложение, токенизируем, находим multi-token слова (k ≥ 2).
2. Стартуем с полной маски, на каждом шаге scheduler выбирает позиции для reveal.
3. В **oracle** режиме подставляем **gold token** — ошибки не накапливаются.
4. На каждом глобальном mask ratio **t** записываем: остаётся ли целое k-token слово полностью в маске.
5. Сравниваем с training expectation:

   **Mismatch(k, t) = P_traj(whole word unresolved | k, t) / P_train(whole word unresolved | k, t)**

   где P_train = t^k.

**Модель:** LLaDA-8B-Base, policy = `low_confidence`, 32 steps, ~68–72 предложения / язык (EN, RU, DE, FI, ZH).

### Negative control — random scheduler

`oracle_trajectory_random.py`: при случайном порядке reveal mismatch ≈ **1.0** (симуляция согласована с t^k). Sanity check пройден.

### Результаты — confidence decoding

**Mismatch(k, t) при t = 0.3:**

| lang | k=2 | k=3 | k=4 |
|------|-----|-----|-----|
| en   | 3.5× | **11.7×** | **42.9×** |
| ru   | 2.8× | 9.2×  | 27.8× |
| de   | 2.6× | 11.6× | 29.1× |
| fi   | 3.2× | 9.9×  | 19.7× |

**Пример (EN, k=4, t=0.3):** P_traj ≈ 0.35, P_train = 0.3⁴ ≈ 0.008 → слово остаётся fully masked в **~43× чаще**, чем на training.

При более высоком t mismatch снижается (t=0.7, k=4: ~2.5–2.8×), но остаётся > 1.

### Выводы по link #3

1. **Confidence-decoding систематически создаёт «unresolved lexical islands»** — даже на EN, не только на fragmented языках.
2. Эффект **сильно растёт с k**: k=4 на порядки хуже k=2.
3. При **фиксированном k** mismatch **сопоставим across languages** — доминирует **policy × k**, а не policy × language.
4. **Языковой penalty** идёт через **бóльшую долю слов с большим k** (RU: 56% k≥4 на LLaDA tok), а не через отдельный language-specific decoding prior.

**Статус: GO** — ключевое звено гипотезы подтверждено.

### Примечание: что делает oracle и зачем он нужен

Oracle **не проверяет**, может ли модель воспроизвести предложение. Модель используется только для **выбора, какие позиции открыть** (confidence); **что** записывается — всегда gold. Так мы изолируем **decoding policy** от накопления ошибок.

На шаге 0 (all-mask) приор всё равно есть: веса LLaDA + logits/confidence на каждой позиции. С шага 1 модель видит gold-контекст.

---

## 5b. Эксперимент 3c — free decoding (без oracle)

### Зачем

Показать, что mismatch **не артеfact oracle**, а свойство реального decoding. Две постановки:

| Setup | Описание |
|-------|----------|
| **full** | Всё предложение в маске → reveal **model argmax** (ошибки накапливаются) |
| **prompt** | 25% gold prefix + `generate()` на остаток (ближе к «свободной генерации») |

Скрипт: `free_decoding_trajectory.py`. Та же метрика mismatch(k,t).

### Результаты — full free decoding (EN, t=0.3)

| lang | k=2 | k=3 | k=4 | tok_acc |
|------|-----|-----|-----|---------|
| en   | 3.6× | 10.2× | 13.4× | ~0% |
| ru   | 2.3× | 6.5× | **29.7×** | ~0% |
| de   | 2.7× | 10.5× | 36.0× | ~0% |
| fi   | 2.9× | 8.0× | 20.1× | ~0% |

### Результаты — prompt generation (25% prefix, EN, t=0.3)

| lang | k=2 | k=3 | k=4 | tok_acc |
|------|-----|-----|-----|---------|
| en   | 3.3× | 6.9× | **41.2×** | ~3.6% |
| ru   | 2.1× | 5.1× | 36.7× | ~1.9% |
| de   | 2.8× | 9.8× | 29.0× | ~1.4% |
| fi   | 2.1× | 6.9× | 11.8× | ~1.9% |

### Выводы по 3c

1. **Mismatch сохраняется на free decoding** — для k≥3 часто **сопоставим или выше**, чем oracle (особенно k=4 в prompt setup).
2. **tok_acc ≈ 0–4%** на OPUS non-EN — модель плохо генерирует эти тексты, но **trajectory mismatch измеряется до финала** и не требует правильной генерации.
3. Prompt setup ближе к inference: часть контекста видна, mismatch для k=4 у EN **41×** — сильнее, чем oracle (43× vs comparable).

**Статус: GO** — policy-induced mismatch воспроизводится без gold injection.

---

## 5c. Эксперимент 6 — policy sweep (feasibility §6)

### Метод

Oracle trajectories, те же предложения; 4 policies: `low_confidence`, `topk_margin`, `random`, `l2r`.  
Скрипт: `policy_sweep.py`. EN/RU/DE/FI, ~67–72 sent/lang.

### Cross-lang mean Mismatch @ t=0.3

| policy | k=2 | k=3 | k=4 |
|--------|-----|-----|-----|
| **low_confidence** | 3.0× | **10.6×** | **29.9×** |
| **topk_margin** | 2.8× | 9.6× | 26.0× |
| **l2r** | 2.9× | 11.0× | **37.3×** |
| **random** | 1.2× | 1.1× | ~0.1× |

### Выводы

1. **Random ≈ 1** (k=2,3) — sanity control пройден; mismatch **не** артеfact метрики.
2. **Confidence и margin** дают сильный mismatch, масштабируется с k — эффект **policy-specific**, не языковый.
3. **L2R ещё хуже** для k=4 (37× vs 30×) — фиксированный порядок тоже создаёт lexical islands.
4. **Policy × k** доминирует; **Policy × Language** — нет (cross-lang mean одинакового порядка для каждой policy).

**Статус: GO** — link #6 закрыт на PoC-уровне.

---

## 5d. Weighted exposure — fragmentation × policy (связка #1 × #6)

### Зачем

Policy sweep показал: **Mismatch(k,t) при фиксированном k** — одинаков по языкам. Но в RU **bad states встречаются чаще**, потому что слова длиннее (56% k≥4 vs 4% у EN). Нужна метрика, которая **не смешивает** два эффекта.

### Метод

Скрипт: `weighted_exposure_audit.py`. Артефакт: `weighted_exposure_llada.json`.

**Определения** (policy = `low_confidence`, P_traj **pooled** по EN/RU/DE/FI — убираем шум траекторий):

| Метрика | Формула | Интерпретация |
|---------|---------|---------------|
| **Corpus word-slot** | `E = Σ_{k≥2} f_k(L) · P_traj(k,t)` | Случайное слово из корпуса в whole-unresolved state |
| **Token-weighted** | `E_tok = Σ_{k≥2} (k·f_k/E[k]) · P_traj(k,t)` | Случайный subtoken в bad state |
| **Excess** | `E − Σ w_k · t^k` | OOD сверх training objective |

**Counterfactual:** `E(f_RU, P_pool) / E(f_EN, P_pool)` — чистый fragmentation factor при одной policy.

### Результаты @ t=0.3

**Corpus word-slot exposure** (главная aggregate-метрика):

| lang | P(k≥2) | P(k≥4) | E_traj | E_train | excess | vs EN |
|------|--------|--------|--------|---------|--------|-------|
| en   | 34.5%  | 4.3%   | 0.093  | 0.023   | 0.070  | 1.00× |
| ru   | 83.9%  | 56.0%  | 0.216  | 0.018   | 0.198  | **2.33×** |
| de   | 68.0%  | 21.9%  | 0.180  | 0.031   | 0.149  | 1.94× |
| fi   | 87.4%  | 38.8%  | 0.230  | 0.030   | 0.201  | 2.49× |

**Token-weighted @ t=0.3:**

| lang | E_traj | vs EN |
|------|--------|-------|
| en   | 0.146  | 1.00× |
| ru   | 0.236  | **1.61×** |
| de   | 0.217  | 1.49× |
| fi   | 0.246  | 1.68× |

**Counterfactual RU vs EN:**
- Fragmentation factor (corpus, pooled P_traj): **2.33×**
- Fragmentation factor (token-weighted): **1.61×**
- P_traj lang gap @ same f_RU: **1.00×** — траектории по языкам не объясняют разницу

### Выводы — два уровня, без противоречия

1. **При фиксированном k** (§5c): mismatch одинаков → policy × k, не language bias.
2. **Aggregate exposure** (§5d): RU **2.3×** чаще попадает в bad states на уровне слов, **1.6×** на уровне subtokens — из‑за `f_k` (больше multi-token слов, особенно k≥4).
3. Counterfactual подтверждает: **весь языковой gap = fragmentation**, не различие decoding path.
4. Excess exposure у RU тоже выше (0.20 vs 0.07) — не только чаще, но и **сильнее** отрыв от training distribution.

**Статус: GO** — механизм «RU хуже не потому что другая policy, а потому что чаще в опасных k» количественно подтверждён.

---

## 5e. Prompt generation eval — exposure → errors

### Метод

25% gold prefix + `generate()` на остаток (как §3c prompt).  
Скрипт: `prompt_gen_eval.py`. Метрики: **tok_acc**, **normalized char edit distance**, exact-match rate.  
Корреляция с **token-weighted exposure** (§5d) по языкам.

### Результаты

| lang | n_sent | tok_acc | char_edit | exact_match |
|------|--------|---------|-----------|-------------|
| en   | 53     | 3.6%    | 0.836     | 0%          |
| ru   | 67     | 1.9%    | **0.879** | 0%          |
| de   | 52     | 1.4%    | 0.860     | 0%          |
| fi   | 47     | 1.9%    | 0.875     | 0%          |

**Корреляция exposure vs errors (4 langs):**
- exposure vs char_edit: **r = +0.97**
- exposure vs tok_acc: **r = −0.87**

### Выводы

1. Генерация на OPUS по-прежнему **плохая** (tok_acc 1–4%), но **RU/FI/DE хуже EN** по char edit — в направлении, предсказанном exposure.
2. Сильная корреляция exposure ↔ errors на 4 языках — aggregate fragmentation penalty **связана с реальными ошибками**, не только с trajectory metric.
3. Exact match = 0% везде — нужны более простые тексты для абсолютных метрик; относительное сравнение langs всё равно информативно.

**Статус: GO (preliminary)** — link mismatch/exposure → generation errors подтверждён направленно.

### 5e-bis. Short texts (≤48 tok, `prompt_gen_eval_llada_short.json`)

| lang | n_sent | tok_acc | char_edit |
|------|--------|---------|-----------|
| en   | 258    | 4.1%    | 0.828     |
| ru   | 291    | 3.4%    | 0.884     |
| de   | 289    | 2.5%    | 0.869     |
| fi   | 258    | 1.5%    | 0.888     |

Exposure vs char_edit: **r = +0.998** (4 langs). Абсолютный tok_acc почти не вырос vs полный OPUS — фильтр дал больше n, не проще задачу.

---

## 7. Репликация — Dream policy sweep

### Метод

Тот же pipeline (`policy_sweep.py`), модель **Dream-v0-Base-7B** (Qwen tokenizer family).  
Патч `model_loader.py`: RoPE `default`, Dream generation config, AR-shift logits.

### Cross-lang mean Mismatch @ t=0.3

| policy | k=2 | k=3 | k=4 |
|--------|-----|-----|-----|
| **low_confidence** | 2.5× | **9.3×** | **33.3×** |
| **topk_margin** | 2.5× | 9.9× | 34.2× |
| **random** | 1.2× | 0.8× | ~0.1× |

### Вывод

**Механизм реплицируется на Dream** — те же порядки величин, что LLaDA (§5c): random≈1, confidence/margin >> 1, масштаб с k.

**Статус: GO** — link #3/#6 не артеfact LLaDA.

---

## 8. Репликация — Fast-dLLM policy sweep

### 8a. Full-sequence forward (упрощение)

Fast-dLLM v2 7B, oracle trajectory через `model(x).logits` без block MDM.

| policy | k=2 | k=3 | k=4 |
|--------|-----|-----|-----|
| low_confidence | ~1.0× | 1.3× | 0.8× |
| random | 1.2× | 1.2× | 1.8× |

**Статус: inconclusive** — full-sequence forward не отражает block decoding.

### 8b. Native block MDM adapter (`fast_dllm_trajectory.py`)

Single-block MDM: полностью замаскированное предложение (≤128 tok), AR-shift logits, low_confidence unmask по 1 tok/step.  
Скрипт: `fast_dllm_policy_sweep.py` → `policy_sweep_fast_dllm_native.json`.

**Mismatch @ t=0.9** (основная масса obs при 1-tok/step; t=0.3 разрежен):

| policy | k=3 | k=4 | k=8 |
|--------|-----|-----|-----|
| **low_confidence** (all langs) | **1.37×** | **1.52×** | **2.32×** |
| random (cross-lang @ t=0.3) | 0.7–1.7× | 0.5–1.6× | — |

Low_confidence **масштабируется с k** (1.37 → 1.52 → … → 5.4 @ k=16) — тот же qualitative pattern, что Dream/LLaDA. Random ≈ 1 на EN, но **RU/DE/FI показывают k=3–4 mismatch @ t=0.3** (0.7–1.7×).

### Вывод

Native block adapter **восстанавливает mismatch-сигнал** для Fast-dLLM. Magnitude ниже Dream (9× @ k=3), но направление и k-scaling совпадают.

**Статус: GO (partial)** — link #3/#6 реплицируется; нужна калибровка t-bins (multi-token unmask per step) для сопоставимости с LLaDA @ t=0.3.

---

## 9. Codebook DLM vs AR (#5)

### 9a. v2 — zero-shot OOD prompt

Dream vs Qwen2.5-7B, OPUS EN words, prompt `Record the code:`.

| k | DLM tok_acc | AR tok_acc | AR/DLM |
|---|-------------|------------|--------|
| 2 | 0.08% | 0.5% | 6.0× |
| 4 | 0.31% | 1.4% | 4.4× |

AR слегка лучше — **нет сигнала diffusion penalty** на zero-shot OOD.

### 9b. v3 — few-shot (2 examples) + oracle upper bound

Скрипт расширен: `--fewshot 2 --compare-oracle`, AR teacher-forced upper bound.  
→ `codebook_segmentation_dream_v3.json`

| k | DLM free | DLM oracle | AR greedy | AR teacher |
|---|----------|------------|-----------|------------|
| 1 | **12.5%** | 4.3% | 0% | 0% |
| 2 | **9.4%** | 2.7% | 0.8% | 2.8% |
| 4 | 2.9% | 4.1% | 0.8% | **15.6%** |

**Few-shot поднимает DLM** (k=1: 0→12.5%, k=2: 0.08→9.4%). На k=2 **DLM free > AR greedy** (9.4% vs 0.8%). На k=4 **AR teacher-forced** (15.6%) >> DLM (2.9%) — diffusion всё ещё хуже upper bound AR при длинных кодах.

Oracle DLM **не выше free** на k=1–2 (вероятно, короткие коды + OOD prompt; oracle trajectory ≠ task accuracy).

**Статус: inconclusive** — нет чистого diffusion-specific penalty, но **DLM может обгонять greedy AR** при in-context codes; для #5 нужен in-vocab task с matched training signal.

---

## 6. Эксперимент 4 — gold NLL на oracle trajectories (link #4)

### Метод

На oracle confidence trajectories (3b), **перед каждым reveal**, для каждой ещё masked позиции считаем gold NLL. Сравниваем:
- **whole-word-unresolved** — все sibling subtokens слова ещё в маске (k≥2);
- **partial** — хотя бы один sibling уже открыт.

Скрипт: `nll_oracle_audit.py`. **Перезапуск** с strata по (k, t), char length, frequency.

### Результаты — aggregate

| lang | NLL whole-unresolved | NLL partial (k≥2) | ratio | low-t whole (t≤0.35) |
|------|---------------------|-------------------|-------|----------------------|
| en   | 7.15 | 3.54 | **2.02×** | 5.84 |
| ru   | 5.86 | 3.79 | **1.55×** | 4.83 |
| de   | 7.01 | 4.29 | **1.63×** | 5.68 |
| fi   | 7.18 | 5.21 | **1.38×** | 6.14 |

### Результаты — controlled strata @ t=0.3

| k | EN ratio | RU ratio | DE ratio | FI ratio | cross-lang mean |
|---|----------|----------|----------|----------|-----------------|
| 3 | **2.39×** | 1.50× | 1.51× | 1.23× | **1.66×** |
| 4 | 1.14× | 1.57× | 1.65× | 1.67× | **1.51×** |

Gap **сохраняется внутри фиксированных k и t** — это state effect, не просто «RU модель слабее».  
На EN k=4 @ t=0.3 ratio слабее (1.14×) — мало whole-unresolved observations (n=128).

### Выводы

1. **Whole-word-unresolved states systematically harder** — NLL на 1.4–2× выше aggregate; **1.5–2.4× внутри k=3 @ t=0.3**.
2. Эффект **сильнее на EN/DE** в aggregate (partial baseline ниже); в controlled k,t strata gap **есть у всех langs**.
3. На low-t (где mismatch максимален) gap сохраняется.

**Статус: GO** — underrepresented states не только частые, но и **objectively harder**, в т.ч. при контроле k/t.

---

## 7. Эксперимент 5 — language competence (sanity check)

### Метод

MCQ из OPUS-100: 4 варианта перевода, вопрос «какой fluent»; ответ одной буквой A/B/C/D.  
**Модель:** Qwen2.5-1.5B (AR baseline), ~50 примеров / язык.

### Результаты

| lang | accuracy |
|------|----------|
| en   | 28%      |
| ru   | 30%      |
| de   | 32%      |
| tr   | 28%      |
| fi   | 36%      |
| zh   | 32%      |
| ko   | 32%      |

(~25% = random guess)

### Вывод

Задача **плохо дискриминирует** competence: distractors из того же OPUS-корпуса слишком похожи. Для отделения «модель не знает язык» от «denoising плохой» нужен нормальный benchmark (Belebele, MGSM, MMLU-multilingual). На PoC **не блокирует** — links #1–#3 уже дают сильный сигнал.

---

## 8. Общий вердict — go / no-go

| Link | Статус | Суть |
|------|--------|------|
| **#1** fragmentation | **GO** | RU/FI/TR/DE сильно более фрагментированы, чем EN |
| **#2** objective | **GO** | Другой sibling-exposure на training; t^k vs язык |
| **#3** inference mismatch | **GO** | 10–40× mismatch; **подтверждено на free decoding (3c)** |
| **#4** harmful (NLL) | **GO** | Whole-unresolved NLL 1.5–2.4× выше partial; держится в strata k,t |
| **#3→errors** | **GO (prelim.)** | Prompt gen: exposure ↔ char_edit r=0.97 |
| **#5** DLM vs AR | **inconclusive** | Few-shot: DLM > AR greedy @ k=2; AR teacher >> DLM @ k=4 |
| **#6** policy × lang | **GO** | Random≈1; confidence/margin >> 1; **Dream + Fast-dLLM native** |

### Главный вывод

**Гипотеза живая.** Цепочка **1 → 2 → 3 → 4** держится на первом прогоне.

Механизм: **policy × fragmentation (k)** → OOD mask states → higher NLL. Multilingual penalty — через **бóльшую долю k≥3 слов** в RU/FI/DE.

---

## 9. Ограничения

1. **ZH/KO** — сегментация слов грубая; audit для этих языков нужно пересчитать (jieba для ZH).
2. **Competence MCQ** — toy-задача; нужен proper benchmark + сравнение LLaDA vs Qwen.
3. **Один DLM, одна policy** — только LLaDA + `low_confidence`.
4. **Free decoding tok_acc ~0%** на non-EN OPUS — модель плохо генерирует эти тексты; mismatch всё равно валиден, но нужны тексты ближе к training distribution.
5. **NLL link #4** — без контроля frequency, char length, global t stratification.
6. **Корпус OPUS** — короткие переводческие фразы.
7. **LLaDA loading** — патч `model_loader.py` для transformers 5.8.

---

## 10. Следующие шаги (приоритет)

1. ~~**Fast-dLLM native trajectory**~~ — **сделано** (`fast_dllm_trajectory.py`); улучшить t-bin coverage (multi-unmask/step).
2. **Codebook #5 v3+** — in-vocab codes, LLaDA same-vocab baseline, больше few-shot.
3. ~~**Prompt gen short texts**~~ — **сделано** (`prompt_gen_eval_llada_short.json`).
4. **Causal training** — lexical/group masking (§7), если #5 держится.
5. **ZH audit** с jieba; **competence** на Belebele / MGSM.

---

## 11. Файлы результатов

| Файл | Содержание |
|------|------------|
| `tokenization_audit.json` | P(k) по языкам и tokenizer'ам |
| `objective_audit_llada.json` | t^k агрегаты, LLaDA tok |
| `objective_audit_qwen.json` | t^k агрегаты, Qwen tok |
| `oracle_random_llada.json` | Random scheduler sanity check |
| `oracle_confidence_llada.json` | Confidence oracle, mismatch по lang/k/t |
| `free_decoding_llada.json` | Free decoding: full + prompt setups |
| `policy_sweep_llada.json` | Mismatch по 4 policies × lang/k/t |
| `weighted_exposure_llada.json` | Weighted OOD exposure + counterfactuals |
| `prompt_gen_eval_llada.json` | Prompt gen errors vs exposure |
| `codebook_segmentation_dream_v2.json` | Dream vs Qwen codebook zero-shot |
| `codebook_segmentation_dream_v3.json` | Dream vs Qwen few-shot + oracle + AR teacher |
| `policy_sweep_dream.json` | Dream policy sweep |
| `policy_sweep_fast_dllm.json` | Fast-dLLM full-seq sweep (inconclusive) |
| `policy_sweep_fast_dllm_native.json` | Fast-dLLM block MDM native sweep |
| `prompt_gen_eval_llada_short.json` | Prompt gen on short OPUS (≤48 tok) |
| `nll_oracle_llada.json` | Gold NLL + by_k_t strata |
| `competence_qwen15_mcq.json` | MCQ accuracy по языкам |

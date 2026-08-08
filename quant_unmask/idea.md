
# Order Drift: quantization may preserve _what_ but break _where_ in diffusion LMs

## Ключевая идея

В LLaDA-style masked diffusion decoding есть два разных решения:

1. **What-to-unmask** — какой токен поставить в masked-позицию.
    
2. **Where-to-unmask** — какую masked-позицию раскрыть сейчас.
    

В LLaDA позиции раскрываются по confidence ranking.

Для каждой masked-позиции:

$$  
\hat{x}_i = \arg\max_v p(v \mid x_t, i)  
$$

Confidence этого предсказания:

$$  
c_i = p(\hat{x}_i \mid x_t, i)  
$$

Затем модель раскрывает top-k позиций:

$$  
S_t = \operatorname{TopK}_{i \in \text{masked}}(c_i, k)  
$$

Low-bit quantization может сохранить token top-1:

 $$  
\arg\max_v p^Q(v \mid x_t, i)
$$
$$
\arg\max_v p^{FP}(v \mid x_t, i)  
$$

но изменить ranking позиций:

$$  
\operatorname{TopK}_i c_i^Q  
\neq  
\operatorname{TopK}_i c_i^{FP}  
$$

То есть модель может всё ещё знать, **что** написать, но раскрывать токены по другой **where/order** траектории.

---

## Почему это правдоподобно

Quantization noise может быть малым относительно **token margin**:

$$  
z_{i,1} - z_{i,2}  
$$

поэтому token top-1 остаётся стабильным.

Но та же ошибка может быть большой относительно **position margin**:

$$  
c_{(k)} - c_{(k+1)}  
$$

поэтому top-k set позиций меняется.

Это особенно вероятно, когда у многих masked-позиций близкие confidence scores, например на early или middle denoising steps.

---

## На что опираемся

- **LLaDA decoding**: позиции выбираются через `topk(confidence)` среди masked-позиций.
    
- **Where-to-Unmask**: показывает, что where-to-unmask — это отдельное важное решение, которое сильно влияет на качество.
    
- **DLM quantization papers**: изучают low-bit quantization для diffusion LMs, но в основном через logits, activations, calibration и downstream accuracy.
    
- **Gap**: существующие quantization evaluations явно не проверяют, сохраняет ли quantization cross-position confidence ranking, который управляет decoding.
    

---

## Paper gap

Существующие работы в основном спрашивают:

> Does the quantized DLM preserve token predictions, logits, activations, or downstream accuracy?

Но LLaDA-style decoding требует ещё одного условия:

> Does the quantized DLM preserve the ranking of masked positions by confidence?

Предлагаемый hidden failure mode:

> **Order drift**: quantization-induced divergence of the unmasking trajectory, not reducible to token top-1 errors.

---

## Модели для проверки

### Основные модели

- **LLaDA-8B / LLaDA-8B-Instruct**
    
    - самый чистый объект;
        
    - есть official generation code;
        
    - position selection явно реализован через confidence-based top-k.
        
- **Dream-7B**
    
    - вторая diffusion LM;
        
    - нужна, чтобы показать, что эффект не только LLaDA-specific.
        

### Опциональные модели

- **Dream-Coder-7B**
    
    - code generation;
        
    - порядок раскрытия может быть особенно важен для синтаксиса и структуры.
        

---

## Quantized checkpoints

### LLaDA-8B

- [qubitron/LLaDA-8B-Quantized](https://huggingface.co/qubitron/LLaDA-8B-Quantized)  
    INT8 / INT4 weight-only; удобно для первого smoke test.
- [qubitron/LLaDA-8B-Quantized — files](https://huggingface.co/qubitron/LLaDA-8B-Quantized/tree/main)  
    файлы `llada_int8_quantized.pt`, `llada_int4_quantized.pt`.
- [FunAGI/LLaDA-8B-Instruct-gptqmodel-4bit](https://huggingface.co/FunAGI/LLaDA-8B-Instruct-gptqmodel-4bit)  
    GPTQ 4-bit.
- [mrdmnd/llada-8b-instruct-4bit-gptq](https://huggingface.co/mrdmnd/llada-8b-instruct-4bit-gptq)  
    community GPTQ 4-bit.

### Dream-7B

- [Rainnighttram/Dream-v0-Instruct-7B-4bit](https://huggingface.co/Rainnighttram/Dream-v0-Instruct-7B-4bit)  
    4-bit quantized Dream-v0-Instruct-7B.
- [Rainnighttram/Dream-7B-bnb-4bit](https://huggingface.co/Rainnighttram/Dream-7B-bnb-4bit)  
    старый bnb-4bit вариант; на странице есть warning про performance issues.
- [bartowski/Dream-org_Dream-v0-Instruct-7B-GGUF](https://huggingface.co/bartowski/Dream-org_Dream-v0-Instruct-7B-GGUF)  
    GGUF quants; полезно для внешней проверки, но неудобно для логирования internals.

### LLaDA2.0

- [mlx-community/LLaDA2.0-mini-4bit](https://huggingface.co/mlx-community/LLaDA2.0-mini-4bit)
- [mlx-community/LLaDA2.0-mini-6bit](https://huggingface.co/mlx-community/LLaDA2.0-mini-6bit)
- [mlx-community/LLaDA2.0-mini-8bit](https://huggingface.co/mlx-community/LLaDA2.0-mini-8bit)

MLX удобно для Mac/inference, но для статьи лучше PyTorch-compatible модели, потому что нужно логировать decoding internals.

---

## Quantization settings

Начать с:

- FP16 / BF16 baseline;
    
- INT8 / W8;
    
- INT4 / W4;
    
- GPTQ-4bit, если доступно;
    
- bitsandbytes 4-bit, если доступно.
    

Для чистых paper results желательно также сделать controlled quantization из одного и того же FP checkpoint:

- W8;
    
- W4;
    
- W4A8;
    
- W4A4, если технически возможно.
    

Community quantized checkpoints полезны для smoke tests, но controlled quantization лучше для causal interpretation.

---

## Benchmarks

Можно следовать setup из Where-to-Unmask:

|Benchmark|Почему полезен|
|---|---|
|**GSM8K**|reasoning chains; порядок может влиять на последующий контекст|
|**MATH**|более длинный и сложный reasoning|
|**Sudoku 99**|highly order-sensitive constraint solving|
|**StrategyQA**|commonsense reasoning|

Их controlled setup удобен, потому что изолирует order decision:

- fixed completion length;
    
- fully masked completion;
    
- reveal one position per step;
    
- greedy token choice;
    
- меняется только where/order decision.
    

Это хорошо подходит для измерения quantization-induced order drift.

---

## Experiments

### 1. Fixed-state fidelity

Запустить FP и quantized models на одних и тех же masked states.

Измерять:

- token top-1 agreement;
    
- confidence error;
    
- Spearman/Kendall rank correlation over positions;
    
- top-1 position agreement;
    
- top-k position overlap.
    

Ожидаемый результат:

> Token top-1 agreement высокий, но position agreement заметно ниже.

---

### 2. Free-running trajectory drift

Запустить FP и quantized decoding независимо.

Логировать на каждом шаге:

- masked sequence;
    
- predicted top token per position;
    
- confidence per position;
    
- selected positions;
    
- revealed tokens.
    

Измерять:

- first divergence step;
    
- trajectory edit distance;
    
- cumulative top-k overlap;
    
- final answer accuracy;
    
- output edit distance.
    

Ожидаемый результат:

> Маленькие confidence perturbations накапливаются и приводят к разным decoding trajectories.

---

### 3. What/where decomposition

Запустить четыре decoding modes:

|Mode|Where|What|Purpose|
|---|---|---|---|
|**FP/FP**|FP|FP|full-precision baseline|
|**Q/Q**|Q|Q|normal quantized decoding|
|**FP/Q**|FP|Q|token degradation only|
|**Q/FP**|Q|FP|position/order degradation only|

Сильный результат:

$$  
FP/Q \approx FP/FP  
$$

но

$$  
Q/FP \ll FP/FP  
$$

Это показало бы, что degradation идёт в основном через **where/order drift**, а не через token prediction drift.

---

### 4. Early/mid/late perturbation

Использовать FP decoding, но заменять where decisions на quantized where decisions только в одном сегменте:

- 0–10%;
    
- 10–20%;
    
- ...
    
- 90–100%.
    

Ожидаемый результат:

> Early quantization-induced order drift вредит сильнее, чем late drift.

Это согласуется с выводом Where-to-Unmask, что ранние position decisions особенно важны.

---

### 5. Margin analysis

Для каждого decoding step считать token margin:

$$  
M_i^{token} = z_{i,1} - z_{i,2}  
$$

и position boundary margin:

$$  
M_t^{pos} = c_{(k)} - c_{(k+1)}  
$$

Проверить:

> Order drift возникает там, где position margins маленькие, даже если token margins большие.

Это даёт механистическое объяснение эффекта.

---

### 6. Varying (k), steps, and block length

В LLaDA (k) — это число позиций, раскрываемых за один шаг.

Варьировать:

- (k = 1): full order stability;
    
- (k > 1): top-k boundary stability;
    
- `block_length = gen_length`: global order;
    
- `block_length < gen_length`: local order внутри semi-AR blocks.
    

Ожидаемый результат:

> При большем (k) важнее top-k boundary; маленькие boundary margins усиливают order drift.

---

## Возможный mitigation

### Order-aware calibration

Во время PTQ calibration сохранять не только logits, но и FP position ranking.

Loss:

# $$  
\mathcal{L}

\mathcal{L}_{logit}  
+  
\lambda \mathcal{L}_{rank}  
$$

Boundary-focused rank loss:

# $$  
\mathcal{L}_{rank}

\sum_{i \in S_k, j \notin S_k}  
\max(0, \gamma - (c_i^Q - c_j^Q))  
$$

Цель:

> Preserve which positions enter the unmask set.

---

### Order-sensitive mixed precision

1. Quantize one layer/block.
    
2. Measure drop in top-k position overlap.
    
3. Keep order-sensitive modules in 8-bit/BF16.
    
4. Quantize the rest more aggressively.
    

---

## Главный expected claim

Low-bit diffusion LMs нужно оценивать не только по тому, сохраняют ли они token predictions, но и по тому, сохраняют ли они confidence ranking over masked positions.

Иначе quantized model может сохранять **what** it wants to write, но менять **where** it writes first, вызывая скрытую trajectory-level degradation.

---

## One-sentence pitch

> A quantized diffusion LM may still know **what** to generate, but reveal it in the wrong **order**.
# Hidden states внутри multi-token слова + logit lens

Samples: 250 strict lexical words from `checkpoints/results_wikitext_fp16_g64_n256`
Layer: -1 (last hidden before ff_out)
**Фильтр:** `is_lexical` (без word+punctuation)

## 1. Похожи ли hidden'ы токенов одного слова?

Cosine similarity hidden states в момент первого unmask (после unmask первого токена).

| Метрика | Value |
|---------|-------|
| mean cos(within word pairs) | 0.748 |
| median cos(within) | 0.758 |
| mean cos(first_unmasked vs other masked) | 0.718 |
| median cos(cross) | 0.749 |
| mean Δ(within − cross) | +0.030 |
| % words where within > cross | 60.8% |

**Вывод:** да, но **слабо**: within-word cos выше на Δ≈0.03 (61% слов).

## 2. Logit lens с hidden первого unmask-токена

Берём `hidden[first_unmasked_pos]`, проецируем через `model.transformer.ff_out` → logits. Проверяем, попадают ли **финальные** sibling-токены в top-k.

| Метрика | Value |
|---------|-------|
| sibling in top-1 | 0.0% |
| sibling in top-10 | 27.5% |
| sibling in top-50 | 51.0% |
| median rank of sibling token | 47 |
| trace sibling pred==final (baseline) | 65.0% |
| logit lens top1 at sibling **position** hidden | 64.4% |

**Интерпретация:** logit lens с позиции первого unmask — это НЕ то же самое, что pred на masked sibling позиции. Сравниваем оба.

### Примеры (top sibling rank / hits)

- `Scientologists` (2tok): lens_top1=`ologists`, sibling ` Scient` rank=2 top10=True
- `battalion` (2tok): lens_top1=`alion`, sibling ` batt` rank=2 top10=True
- `australian` (2tok): lens_top1=`ustralian`, sibling ` a` rank=2 top10=True
- `it's` (2tok): lens_top1=`'s`, sibling ` it` rank=2 top10=True
- `Radcliffe` (2tok): lens_top1=` Rad`, sibling `cliffe` rank=2 top10=True

### Примеры (худшие)

- `Revlon` (2tok): lens_top1=` Rev`, sibling `lon` rank=3644
- `BCOF` (3tok): lens_top1=` B`, sibling `CO` rank=3030
- `Musicians` (2tok): lens_top1=` Music`, sibling `ians` rank=6773
- `Peavey` (3tok): lens_top1=` Pe`, sibling `ave` rank=10158
- `Peavey` (3tok): lens_top1=` Pe`, sibling `ave` rank=10017

## 3. Краткий ответ

1. **Hidden similarity:** within-word cos ≈ 0.75 vs cross ≈ 0.72 (Δ=+0.03).
2. **Logit lens с первого unmask:** sibling token в top-1 только 0.0%, в top-10 27.5% — **слово целиком из одного hidden не восстанавливается**.
3. На самой masked sibling-позиции logit lens top1 = 64.4% (≈ trace baseline 65.0%) — модель предсказывает sibling там, где он стоит, не с первого токена.

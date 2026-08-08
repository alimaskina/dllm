# Dream vs LLaDA: multitoken / word representation

Same WikiText prompts; Dream n=64 new traces; LLaDA n=64 from `results_wikitext_fp16_g64_n256`.

## Unmask order

| | Dream | LLaDA |
|--|------:|------:|
| 2-tok gap=1 | 92% | 91% |
| same-step | 0% | 0% |
| LR / RL | 49 / 51 | 56 / 44 |
| sibling pred==final | 61% | 61% |

**Одинаково:** нет AR left→right unmask; halves открываются почти consecutive, без co-unmask.

## Attention: others → left vs right (`all_open`)

Prefer **right%**:

| layer-ish | Dream | LLaDA |
|-----------|------:|------:|
| early (~7) | **82%** | 71% |
| mid (~14/21) | **74–91%** | 61–64% |
| late (~27/31) | 69% | 57% |

Dream **сильнее** right-bias (особенно L21: 91%). First vs second unmasked ≈ 50/50 у обеих.

## Logit lens sibling (`all_open`, mean rank, 0=top1)

| | Dream L27 | LLaDA L31 |
|--|----------:|----------:|
| L self / R self | 956 / 1910 | **1.8 / 0.0** |
| L→R sib / R→L sib | **85** / 1204 | 356 / **54** |

- LLaDA late: self почти идеально читается; sibling лучше **из right → left** (ближе к AR last-token).
- Dream late: self плохо через lm_head без AR-shift; sibling лучше **из left → right** (обратно AR).

## Hidden drift after open

Обе модели: **jump ≫ post-open drift**.

| | Dream jump/old L2 | LLaDA jump/old L2 |
|--|------------------:|------------------:|
| early | ~14–44× | ~12–55× |
| late | ~4–6× | ~4–5× |

Already-open ≈ still-MASK (old/mask ≲ 1). Слово не «дособирается» в открытом токене.

## Sibling activation patching (`between`)

Corrupt: opened → MASK. Patch clean residual at opened pos @ layer L.  
Metric: logit правильного sibling на ещё MASK позиции. recovery% = (patch−corrupt)/(clean−corrupt).

| layer | Dream recovery | LLaDA recovery |
|------:|---------------:|---------------:|
| 0 | **95%** | **92%** |
| 7 | 73% | 83% |
| 14 | 60% | 76% |
| 21 | 43% | 45% |
| late (27/31) | 47% | **0%** |

Clean≫corrupt у обеих (opened token **causally** помогает sibling).  
Инфа в opened residual **ранних/mid** слоёв; к last layer LLaDA patch бесполезен (уже переписано / не в том слоте).

## Где живёт представление слова?

1. **Не** единый last-token object как в чистом AR — unmask order и (у Dream) lens этому противоречат.
2. **Есть** causal зависимость: открытый subword несёт сигнал для sibling (patching 90%+ @ L0).
3. Сигнал **ранний** в residual opened позиции; late layer ≠ хранилище.
4. Снаружи чаще **читают right** (attn), особенно Dream — это point of readout, не обязательно storage.
5. После commit hidden **застывает** — storage не дрейфует как growing summary.

Файлы: `dream_wikitext_multitoken_suite.md`, `llada_wikitext_multitoken_suite.md`, `dream_sibling_patching.md`, `llada_sibling_patching.md`, `dream_hidden_drift_after_open_all_layers.md`.

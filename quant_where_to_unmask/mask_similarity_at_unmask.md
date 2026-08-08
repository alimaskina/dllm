# Mask↔mask hidden similarity при unmask

Traces: **32**, событий: **2048**, checkpoint: `checkpoints/results_wikitext_fp16_g64_n256`, layer: **-1**

## Методология

На step `s` перед unmask позиции `p`:
- **cross_mask** = cos(h[p], h[q]) для других `[MASK]` в completion
- **ctrl_pair** = средний cos между парами **других** still-masked (без p)
- **jump** = cos(h[p] before → h[p] after unmask) — как в hidden jump
- **sib_mask** = cross только к sibling `[MASK]` в том же multi-token слове

Вопрос: насколько unmask-позиция похожа на **другие маски** до открытия, и меняется ли это после commit.

---

## 1. Похожесть на другие маски **до** unmask

| Метрика | value |
|---------|-------|
| mean cross_mask cos | 0.691 |
| mean ctrl_pair cos | 0.885 |
| Δ(cross − ctrl) | -0.193 |
| mean cross_mask cos max | 0.795 |
| mean sib_mask cos | 0.736 (n=81) |

- **6.8%** событий: cross_mask **выше** ctrl_pair на >0.01

---

## 2. Self jump vs cross similarity

| Метрика | value |
|---------|-------|
| mean jump cos (self) | 0.810 |
| mean cross_mask cos (before) | 0.691 |
| mean (jump − cross) | +0.119 |

Если jump ≪ cross ⇒ unmask меняет hidden **сильнее**, чем типичное сходство между масками.

---

## 3. После unmask: связь с оставшимися масками

| Метрика | value |
|---------|-------|
| mean cos к remaining masks (after) | 0.678 |
| Δ(after − before) cross | -0.013 |
| mean cross_drift (same q, s→s+1) | -0.013 |

---

## 4. Multi-token слова vs остальное

| subset | n | cross before | Δ cross−ctrl | sib cos | jump−cross |
|--------|---|--------------|--------------|---------|------------|
| multi-token | 150 | 0.630 | -0.261 | 0.736 | +0.148 |
| other | 1898 | 0.696 | -0.188 | — | +0.117 |

---

## 5. По mask_ratio

| bucket | n | cross before | Δ cross−ctrl | jump−cross |
|--------|---|--------------|--------------|------------|
| >0.9 | 224 | 0.702 | -0.261 | +0.090 |
| 0.5-0.9 | 832 | 0.705 | -0.228 | +0.108 |
| <0.5 | 992 | 0.676 | -0.146 | +0.136 |

---

## 6. Выводы

- Unmask-позиция **дальше** от других масок, чем типичная mask pair (Δ=-0.193).
- Self jump cos на **+0.119** ниже/выше cross_mask — unmask меняет позицию сопоставимо с «кластером масок».

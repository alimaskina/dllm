# Hidden jump при unmask: траектория и скачок MASK→REAL

Traces: **64**, unmask-событий: **4096**, checkpoint: `checkpoints/results_wikitext_fp16_g64_n256`, layer: **-1**

## Методология

Trace фиксирует состояние **до** commit unmask (см. `generate.py`).
Для unmask на step `s`, pos `p`:
- **h_before** = forward на sequence trace step `s` (позиция ещё `[MASK]`)
- **h_after** = forward на sequence trace step `s+1` (позиция уже REAL)
- **jump** = переход `h[s] → h[s+1]` на той же позиции
- **control** = те же метрики для позиций, оставшихся `[MASK]` и на `s`, и на `s+1`
- **traj** = средний step-to-step сдвиг hidden на этой позиции **пока она masked** (steps `0..s-1`)

---

## 1. Главный результат: есть ли скачок при unmask?

| Метрика | unmask jump | control (still masked) | Δ / ratio |
|---------|-------------|------------------------|-----------|
| mean cos(step→step+1) | 0.812 | 0.982 | -0.170 |
| mean L2(step→step+1) | 115.635 | 28.956 | ratio **5.145×** |
| median cos | 0.832 | 0.982 | — |
| median L2 | 116.951 | — | — |
| p10 cos | 0.689 | — | — |
| p90 cos | 0.913 | — | — |

- **97.7%** событий: L2 jump > **1.5×** control (n=4032)
- **92.8%** событий: cos jump **ниже** средней masked-траектории на >0.05

**Интерпретация:** если unmask = обычный шаг diffusion, jump ≈ control и ≈ traj. Большой ratio или падение cos vs traj ⇒ сдвиг траектории в момент commit.

---

## 2. Jump vs masked-траектория (одна позиция до открытия)

| Метрика | while masked (traj) | at unmask (jump) | jump / traj |
|---------|---------------------|------------------|-------------|
| mean cos | 0.980 | 0.812 | — |
| mean L2 | 31.701 | 115.635 | **4.199×** |
| mean Δcos (jump − traj) | — | -0.168 | — |

Средняя длина masked-траектории: **31.5** steps (0 для step=0 unmask).

---

## 3. Разбивка по step unmask

| step bucket | n | jump cos | ctrl cos | Δ cos | jump L2 | ratio L2 | jump/traj L2 |
|-------------|---|----------|----------|-------|---------|----------|--------------|
| step=0 | 64 | 0.818 | 0.992 | -0.174 | 116.739 | 5.553 | — |
| step 1-7 | 448 | 0.800 | 0.990 | -0.190 | 122.381 | 5.528 | 2.321 |
| step 8-31 | 1536 | 0.812 | 0.988 | -0.176 | 117.409 | 5.281 | 3.796 |
| step 32-63 | 2048 | 0.814 | 0.975 | -0.160 | 112.793 | 4.940 | 4.912 |

---

## 4. Multi-token слова vs остальные позиции

| subset | n | jump cos | ctrl cos | Δ cos | ratio L2 | jump/traj L2 |
|--------|---|----------|----------|-------|----------|--------------|
| multi-token word | 293 | 0.798 | 0.982 | -0.184 | 5.727 | 4.310 |
| other positions | 3803 | 0.813 | 0.982 | -0.169 | 5.099 | 4.191 |

---

## 5. По mask_ratio в момент unmask

| mask_ratio bucket | n | jump cos | ratio L2 | jump/traj L2 |
|-------------------|---|----------|----------|--------------|
| >0.9 | 448 | 0.803 | 5.548 | 2.263 |
| 0.5-0.9 | 1664 | 0.812 | 5.278 | 3.783 |
| <0.5 | 1984 | 0.814 | 4.936 | 4.923 |

---

## 6. Примеры: largest / smallest jump

### Top L2 jump

- step=25 `
` conf=0.15: cos=0.251, L2=265.1, ratio=6.271×, traj_L2=27.049
- step=16 `
` conf=0.18: cos=0.250, L2=265.0, ratio=6.107×, traj_L2=27.289
- step=25 `
` conf=0.16: cos=0.243, L2=261.3, ratio=2.776×, traj_L2=33.835
- step=30 `
` conf=0.40: cos=0.340, L2=259.9, ratio=4.619×, traj_L2=24.965
- step=6 `
` conf=0.15: cos=0.392, L2=259.7, ratio=5.552×, traj_L2=69.901
- step=7 `
` conf=0.16: cos=0.311, L2=256.9, ratio=9.111×, traj_L2=57.155
- step=11 `
` conf=0.40: cos=0.330, L2=256.4, ratio=7.017×, traj_L2=40.375
- step=3 `.` conf=0.34: cos=0.221, L2=256.0, ratio=6.770×, traj_L2=66.509

### Lowest cos jump (strongest direction change)

- step=40 `2`: cos=0.096, L2=202.1, Δcos_vs_ctrl=-0.903
- step=63 `9`: cos=0.154, L2=172.8, Δcos_vs_ctrl=—
- step=19 `
`: cos=0.220, L2=246.8, Δcos_vs_ctrl=-0.757
- step=45 `
`: cos=0.221, L2=240.2, Δcos_vs_ctrl=-0.771
- step=3 `.`: cos=0.221, L2=256.0, Δcos_vs_ctrl=-0.763
- step=42 `
`: cos=0.222, L2=233.8, Δcos_vs_ctrl=-0.756
- step=13 `.`: cos=0.232, L2=226.1, Δcos_vs_ctrl=-0.758
- step=36 `
`: cos=0.234, L2=230.5, Δcos_vs_ctrl=-0.763

---

## 7. Выводы

- Unmask-переход **крупнее** типичного masked step: ratio vs control ≈ **5.15×**, vs masked traj ≈ **4.20×**.
- Средний Δcos(unmask − control) = **-0.170**.
- Средний Δcos(jump − masked traj) = **-0.168**.
- Ранние unmask (step=0): n=64, ratio L2=5.553×; поздние (step≥32): n=2048, ratio L2=4.940×.

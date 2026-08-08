# Hidden drift: jump vs still-MASK vs already-OPEN

Traces: **32**, layer: **-1**

На каждом переходе step `s → s+1` для каждой completion-позиции:
- **jump** — MASK→REAL на этом step
- **still MASK** — осталась MASK
- **already OPEN** — уже REAL на `s` и на `s+1`
- **fresh OPEN** — открыли на `s-1`, сейчас второй step как REAL
- **old OPEN** — REAL уже ≥2 steps

| group | mean cos | mean L2 | n |
|-------|---------:|--------:|--:|
| jump (MASK→REAL) | 0.810 | 116.7 | 2016 |
| still MASK | 0.987 | 25.7 | 64512 |
| already OPEN | 0.972 | 30.3 | 62496 |
| fresh OPEN (1 step after unmask) | 0.955 | 46.4 | 1984 |
| old OPEN (≥2 steps) | 0.972 | 29.7 | 60512 |

ratio L2 jump / already OPEN = **3.86×**

ratio L2 already OPEN / still MASK = **1.18×**


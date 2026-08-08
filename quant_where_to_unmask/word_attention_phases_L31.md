# Attention и multi-token слова по фазам unmask

Traces: **48**, snapshots: **307**, layers: **[31]**

## Фазы

| Фаза | Описание |
|------|----------|
| `before_first` | До первого unmask слова — все его токены `[MASK]` |
| `between` | После ≥1 REAL, до полного слова — каждый промежуточный step |
| `all_open` | Все токены слова REAL |

## Query groups

- **other_masked** — другие `[MASK]` в completion (не это слово)
- **other_open** — другие REAL в completion
- **word_masked** — `[MASK]` внутри слова
- **word_open** — REAL внутри слова

Метрика: mean Σ attn(query→targets) усреднён по queries в группе.

---

## Фаза: before_first

### Outgoing: куда смотрят → на слово (open vs masked)

| query group | → word_open | → word_masked | open/(open+masked) | n |
|-------------|-------------|---------------|---------------------|---|
| other_masked | — | 0.0162 | — | 96 |
| other_open | — | 0.0160 | — | 96 |
| word_masked | — | 0.1445 | — | 96 |

### Incoming: кто смотрит на word_open / word_masked

**Target: REAL токены слова**

| from | mean incoming mass | n |
|------|-------------------|---|
| other_masked | — | 0 |
| other_open | — | 0 |
| word_masked | — | 0 |
| word_open | — | 0 |

**Target: MASK токены слова**

| from | mean incoming mass | n |
|------|-------------------|---|
| other_masked | 0.2307 | 96 |
| other_open | 0.1564 | 96 |
| word_masked | 0.1445 | 96 |
| word_open | — | 0 |

### other_masked → позиция в слове (left / mid / right)

| target role | mean mass | n |
|-------------|-----------|---|
| left | 0.0071 | 96 |
| mid | 0.0065 | 16 |
| right | 0.0080 | 96 |

---

## Фаза: between

### Outgoing: куда смотрят → на слово (open vs masked)

| query group | → word_open | → word_masked | open/(open+masked) | n |
|-------------|-------------|---------------|---------------------|---|
| other_masked | 0.0148 | 0.0076 | 0.6613 | 115 |
| other_open | 0.0088 | 0.0072 | 0.5485 | 115 |
| word_masked | 0.0603 | 0.1286 | 0.3191 | 115 |
| word_open | 0.1344 | 0.0406 | 0.7678 | 115 |

### Incoming: кто смотрит на word_open / word_masked

**Target: REAL токены слова**

| from | mean incoming mass | n |
|------|-------------------|---|
| other_masked | 0.4747 | 115 |
| other_open | 0.1862 | 115 |
| word_masked | 0.0614 | 115 |
| word_open | 0.1344 | 115 |

**Target: MASK токены слова**

| from | mean incoming mass | n |
|------|-------------------|---|
| other_masked | 0.2314 | 115 |
| other_open | 0.1459 | 115 |
| word_masked | 0.1286 | 115 |
| word_open | 0.0416 | 115 |

### other_masked → позиция в слове (left / mid / right)

| target role | mean mass | n |
|-------------|-----------|---|
| left | 0.0098 | 115 |
| mid | 0.0095 | 35 |
| right | 0.0097 | 115 |

---

## Фаза: all_open

### Outgoing: куда смотрят → на слово (open vs masked)

| query group | → word_open | → word_masked | open/(open+masked) | n |
|-------------|-------------|---------------|---------------------|---|
| other_masked | 0.0235 | — | — | 96 |
| other_open | 0.0164 | — | — | 96 |
| word_open | 0.1823 | — | — | 96 |

### Incoming: кто смотрит на word_open / word_masked

**Target: REAL токены слова**

| from | mean incoming mass | n |
|------|-------------------|---|
| other_masked | 0.3892 | 96 |
| other_open | 0.1829 | 96 |
| word_masked | — | 0 |
| word_open | 0.1823 | 96 |

**Target: MASK токены слова**

| from | mean incoming mass | n |
|------|-------------------|---|
| other_masked | — | 0 |
| other_open | — | 0 |
| word_masked | — | 0 |
| word_open | — | 0 |

### other_masked → позиция в слове (left / mid / right)

| target role | mean mass | n |
|-------------|-----------|---|
| left | 0.0106 | 96 |
| mid | 0.0136 | 16 |
| right | 0.0107 | 96 |

---

## Сводка: other_masked предпочитает open или masked siblings?

| phase | → open | → masked | open share |
|-------|--------|----------|------------|
| before_first | — | 0.0162 | — |
| between | 0.0148 | 0.0076 | 0.6613 |
| all_open | 0.0235 | — | — |

## Сводка: other_open → word

| phase | → word_open | → word_masked | open share |
|-------|-------------|---------------|------------|
| before_first | — | 0.0160 | — |
| between | 0.0088 | 0.0072 | 0.5485 |
| all_open | 0.0164 | — | — |

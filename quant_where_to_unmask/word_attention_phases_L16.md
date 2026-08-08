# Attention и multi-token слова по фазам unmask

Traces: **48**, snapshots: **307**, layers: **[16]**

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
| other_masked | — | 0.0187 | — | 96 |
| other_open | — | 0.0183 | — | 96 |
| word_masked | — | 0.0645 | — | 96 |

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
| other_masked | 0.2451 | 96 |
| other_open | 0.1898 | 96 |
| word_masked | 0.0645 | 96 |
| word_open | — | 0 |

### other_masked → позиция в слове (left / mid / right)

| target role | mean mass | n |
|-------------|-----------|---|
| left | 0.0064 | 96 |
| mid | 0.0318 | 16 |
| right | 0.0069 | 96 |

---

## Фаза: between

### Outgoing: куда смотрят → на слово (open vs masked)

| query group | → word_open | → word_masked | open/(open+masked) | n |
|-------------|-------------|---------------|---------------------|---|
| other_masked | 0.0089 | 0.0096 | 0.4815 | 115 |
| other_open | 0.0093 | 0.0103 | 0.4743 | 115 |
| word_masked | 0.0412 | 0.0571 | 0.4192 | 115 |
| word_open | 0.0550 | 0.0392 | 0.5840 | 115 |

### Incoming: кто смотрит на word_open / word_masked

**Target: REAL токены слова**

| from | mean incoming mass | n |
|------|-------------------|---|
| other_masked | 0.2255 | 115 |
| other_open | 0.2112 | 115 |
| word_masked | 0.0414 | 115 |
| word_open | 0.0550 | 115 |

**Target: MASK токены слова**

| from | mean incoming mass | n |
|------|-------------------|---|
| other_masked | 0.2805 | 115 |
| other_open | 0.2066 | 115 |
| word_masked | 0.0571 | 115 |
| word_open | 0.0400 | 115 |

### other_masked → позиция в слове (left / mid / right)

| target role | mean mass | n |
|-------------|-----------|---|
| left | 0.0081 | 115 |
| mid | 0.0051 | 35 |
| right | 0.0088 | 115 |

---

## Фаза: all_open

### Outgoing: куда смотрят → на слово (open vs masked)

| query group | → word_open | → word_masked | open/(open+masked) | n |
|-------------|-------------|---------------|---------------------|---|
| other_masked | 0.0193 | — | — | 96 |
| other_open | 0.0208 | — | — | 96 |
| word_open | 0.0951 | — | — | 96 |

### Incoming: кто смотрит на word_open / word_masked

**Target: REAL токены слова**

| from | mean incoming mass | n |
|------|-------------------|---|
| other_masked | 0.2736 | 96 |
| other_open | 0.2371 | 96 |
| word_masked | — | 0 |
| word_open | 0.0951 | 96 |

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
| left | 0.0079 | 96 |
| mid | 0.0050 | 16 |
| right | 0.0105 | 96 |

---

## Сводка: other_masked предпочитает open или masked siblings?

| phase | → open | → masked | open share |
|-------|--------|----------|------------|
| before_first | — | 0.0187 | — |
| between | 0.0089 | 0.0096 | 0.4815 |
| all_open | 0.0193 | — | — |

## Сводка: other_open → word

| phase | → word_open | → word_masked | open share |
|-------|-------------|---------------|------------|
| before_first | — | 0.0183 | — |
| between | 0.0093 | 0.0103 | 0.4743 |
| all_open | 0.0208 | — | — |

# Prefer left vs right subword: кто смотрит?

Traces: **64**, 2-tok lexical words: **109**

Для каждой query-группы: mean attn(queries → left) vs → right.
**left_share** = left / (left + right); >0.5 → предпочитают **левый** subword.

## Фаза `before_first`

### Layer 16

| query group | → left | → right | **left_share** | n |
|-------------|-------:|--------:|---------------:|--:|
| other_masked | 0.0067 | 0.0073 | **0.4951** | 109 |
| other_open | 0.0068 | 0.0061 | **0.5285** | 109 |
| word_masked | 0.0341 | 0.0239 | **0.5842** | 109 |

### Layer 31

| query group | → left | → right | **left_share** | n |
|-------------|-------:|--------:|---------------:|--:|
| other_masked | 0.0079 | 0.0090 | **0.4768** | 109 |
| other_open | 0.0085 | 0.0062 | **0.5703** | 109 |
| word_masked | 0.0881 | 0.0499 | **0.6439** | 109 |

## Фаза `between`

### Layer 16

| query group | → left | → right | **left_share** | n |
|-------------|-------:|--------:|---------------:|--:|
| other_masked | 0.0089 | 0.0101 | **0.4527** | 109 |
| other_open | 0.0087 | 0.0096 | **0.4725** | 109 |
| word_open | 0.0482 | 0.0417 | **0.5528** | 109 |
| word_masked | 0.0492 | 0.0418 | **0.5445** | 109 |

### Layer 31

| query group | → left | → right | **left_share** | n |
|-------------|-------:|--------:|---------------:|--:|
| other_masked | 0.0111 | 0.0110 | **0.4804** | 109 |
| other_open | 0.0076 | 0.0073 | **0.5094** | 109 |
| word_open | 0.0951 | 0.0756 | **0.5751** | 109 |
| word_masked | 0.1076 | 0.0733 | **0.5882** | 109 |

## Фаза `all_open`

### Layer 16

| query group | → left | → right | **left_share** | n |
|-------------|-------:|--------:|---------------:|--:|
| other_masked | 0.0081 | 0.0114 | **0.4174** | 109 |
| other_open | 0.0080 | 0.0111 | **0.4364** | 109 |
| word_open | 0.0453 | 0.0464 | **0.4925** | 109 |

### Layer 31

| query group | → left | → right | **left_share** | n |
|-------------|-------:|--------:|---------------:|--:|
| other_masked | 0.0113 | 0.0113 | **0.4851** | 109 |
| other_open | 0.0073 | 0.0077 | **0.4904** | 109 |
| word_open | 0.0993 | 0.0795 | **0.5522** | 109 |


# 2-tok words, фаза `all_open`: per-token attention внутри слова

Traces: **64**, words: **109**, layers: **[16, 31]**

Для каждого токена слова (left / right по позиции):
- **self** = attn[q→q]
- **sibling** = attn[q→other subword]
- **sib%** = sibling / (sibling + self)

> 0.5 → больше mass на sibling, чем на self

## Layer 16

| role | self | sibling | **sib%** | n |
|------|-----:|--------:|---------:|--:|
| left | 0.0509 | 0.0355 | **0.4036** | 109 |
| right | 0.0573 | 0.0397 | **0.4055** | 109 |

Δ sib% (right − left) = **+0.0019**

## Layer 31

| role | self | sibling | **sib%** | n |
|------|-----:|--------:|---------:|--:|
| left | 0.1382 | 0.0352 | **0.2129** | 109 |
| right | 0.1239 | 0.0604 | **0.3329** | 109 |

Δ sib% (right − left) = **+0.1200**


# Sibling activation patching (dream)

Model: **Dream-org/Dream-v0-Base-7B**, words: **80**, phase=`between`

Corrupt = opened subword → MASK. Patch = restore clean residual at opened pos at layer L.
Metric: logit of **correct sibling token** at still-masked position.

| layer | clean | corrupt | patched | Δ(patch−corrupt) | recovery% |
|------:|------:|--------:|--------:|-----------------:|----------:|
| 0 | 20.14 | 12.00 | 19.69 | **+7.69** | **94.5%** |
| 7 | 20.14 | 12.00 | 17.96 | **+5.95** | **73.2%** |
| 14 | 20.14 | 12.00 | 16.90 | **+4.90** | **60.2%** |
| 21 | 20.14 | 12.00 | 15.47 | **+3.46** | **42.6%** |
| 27 | 20.14 | 12.00 | 15.79 | **+3.79** | **46.6%** |

recovery% = (patched−corrupt)/(clean−corrupt). ~100% ⇒ that layer's opened residual carries the sibling-relevant info.

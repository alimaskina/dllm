# Sibling activation patching (llada)

Model: **GSAI-ML/LLaDA-8B-Base**, words: **80**, phase=`between`

Corrupt = opened subword → MASK. Patch = restore clean residual at opened pos at layer L.
Metric: logit of **correct sibling token** at still-masked position.

| layer | clean | corrupt | patched | Δ(patch−corrupt) | recovery% |
|------:|------:|--------:|--------:|-----------------:|----------:|
| 0 | 22.96 | 17.09 | 22.49 | **+5.40** | **92.1%** |
| 7 | 22.96 | 17.09 | 21.99 | **+4.89** | **83.4%** |
| 14 | 22.96 | 17.09 | 21.55 | **+4.45** | **75.9%** |
| 21 | 22.96 | 17.09 | 19.74 | **+2.65** | **45.2%** |
| 31 | 22.96 | 17.09 | 17.09 | **+0.00** | **0.0%** |

recovery% = (patched−corrupt)/(clean−corrupt). ~100% ⇒ that layer's opened residual carries the sibling-relevant info.

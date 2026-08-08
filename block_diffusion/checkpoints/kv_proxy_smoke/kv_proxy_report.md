# One-shot KV allocation proxy test

Source blocks analyzed: **13**
Cross-block attention pairs: **49**

## Next-block proxy (B_{i+1} → B_i) vs future oracle

| Metric | Mean |
|--------|------|
| Spearman ρ vs oracle | 1.000 |
| Top-10% overlap | 1.000 |
| Top-20% overlap | 1.000 |
| Top-30% overlap | 1.000 |
| Recall@10% oracle | 1.000 |
| Recall@20% oracle | 1.000 |
| Recall@30% oracle | 1.000 |

## By distance (observer block j - source block i)

| d | n | Spearman | Ovlp@10% | Ovlp@20% | Ovlp@30% | Recall@10% | Recall@20% | Recall@30% |
|---|---|----------|----------|----------|----------|------------|------------|------------|
| 1 | 13 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 11 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 9 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 6 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 7 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation

- **High recall@d=1** with stable metrics as d grows → one-shot allocation after B_{i+1} is viable.
- **Rapid decay** of Spearman/recall with distance → rankings shift; periodic re-allocation needed.

![distance curve](kv_proxy_smoke/kv_proxy_distance_curve.png)
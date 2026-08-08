# Causal pilot — selective training exposure (LLaDA continued pretraining)

Проверяет: высокий NLL на **whole-unresolved multi-token states** связан с недостаточным training exposure или это intrinsic difficulty.

## Три compute-matched режима

| Mode | Corruption |
|------|------------|
| **IID** | Independent token masking (baseline LLaDA) |
| **WORD** | ~25% примеров: случайное слово с k≥3 fully masked |
| **SPAN** | ~25% примеров: contiguous span k≥3, **не** совпадающий с одним словом |

**Compute matching:** сначала IID mask → фиксируем число masked M; при intervention unit принудительно fully masked, затем rebalance (unmask/mask позиции **вне** unit) чтобы сохранить ровно M.

## Pipeline

```bash
cd multi_language
bash causal_pilot/run_causal_pilot.sh probe   # held-out probe set
bash causal_pilot/run_causal_pilot.sh base    # NLL base model
bash causal_pilot/run_causal_pilot.sh train   # 3 runs (IID, WORD, SPAN)
bash causal_pilot/run_causal_pilot.sh eval    # probe NLL + compare
# или всё сразу:
MAX_STEPS=800 LR=1e-5 bash causal_pilot/run_causal_pilot.sh all
```

## Probe eval buckets

- **k:** 2, 3, 4+
- **t:** ≈0.2, 0.3, 0.4 (oracle trajectory, low_confidence)
- **whole** vs **partial**

Главные метрики: `NLL_IID - NLL_WORD`, `NLL_IID - NLL_SPAN` по bucket → `results/causal_pilot_comparison.json`.

## Интерпретация

- **WORD > SPAN > IID** на whole k3/k4+ @ low t → lexical fragmentation signal
- **WORD ≈ SPAN > IID** → general local-fragmentation, тоже GO
- **WORD ≈ SPAN ≈ IID** → causal exposure hypothesis слабая

## Файлы

| File | Role |
|------|------|
| `corruption.py` | IID / WORD / SPAN + rebalance |
| `train.py` | Continued pretraining loop |
| `probe_set.py` | Fix held-out probe before training |
| `eval_probe.py` | Gold NLL buckets |
| `compare_runs.py` | Delta tables |

**Не включено:** downstream benchmarks, multilingual eval (по design).

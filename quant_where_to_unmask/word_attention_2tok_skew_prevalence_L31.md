# Сколько 2-tok слов с перекосом attention от «чужих» токенов

Traces: **64**, layer: **31**

Per-word share = attn(other→subword0) / (attn→subword0 + attn→subword1).
0.5 = симметрия; перекос = share заметно отличается от 0.5.

## Фаза `before_first`

### По позиции (subword0 vs subword1) (n=109)

- mean share: **0.515** (0.5 = symmetric)
- mean |share−0.5|: **0.052**
- median |share−0.5|: **0.044**
- |share−0.5| > 0.05: **45.0%** слов
- |share−0.5| > 0.10: **9.2%** слов
- |share−0.5| > 0.15: **0.9%** слов
- |share−0.5| > 0.20: **0.0%** слов

## Фаза `between`

### По позиции (subword0 vs subword1) (n=109)

- mean share: **0.499** (0.5 = symmetric)
- mean |share−0.5|: **0.123**
- median |share−0.5|: **0.105**
- |share−0.5| > 0.05: **73.4%** слов
- |share−0.5| > 0.10: **50.5%** слов
- |share−0.5| > 0.15: **30.3%** слов
- |share−0.5| > 0.20: **22.9%** слов

### По состоянию (open vs masked subword) (n=109)

- mean share: **0.600** (0.5 = symmetric)
- mean |share−0.5|: **0.123**
- median |share−0.5|: **0.105**
- |share−0.5| > 0.05: **73.4%** слов
- |share−0.5| > 0.10: **50.5%** слов
- |share−0.5| > 0.15: **30.3%** слов
- |share−0.5| > 0.20: **22.9%** слов

## Фаза `all_open`

### По позиции (subword0 vs subword1) (n=109)

- mean share: **0.493** (0.5 = symmetric)
- mean |share−0.5|: **0.090**
- median |share−0.5|: **0.081**
- |share−0.5| > 0.05: **67.0%** слов
- |share−0.5| > 0.10: **40.4%** слов
- |share−0.5| > 0.15: **16.5%** слов
- |share−0.5| > 0.20: **4.6%** слов

## Примеры сильного перекоса (|pos_share−0.5| > 0.15)

- `Sailor` (between): |Δ|=0.36
- `Blythe` (between): |Δ|=0.36
- `Sailor` (between): |Δ|=0.34
- `McCarty` (between): |Δ|=0.34
- `Ethridge` (between): |Δ|=0.33
- `Tikal` (between): |Δ|=0.31
- `Sailor` (all_open): |Δ|=0.31
- `Tikal` (between): |Δ|=0.30
- `Savige` (between): |Δ|=0.30
- `Ethridge` (between): |Δ|=0.30
- `IJN` (between): |Δ|=0.29
- `Tikal` (all_open): |Δ|=0.29
- `Headlam` (between): |Δ|=0.27
- `Blythe` (all_open): |Δ|=0.26
- `AIF` (between): |Δ|=0.26
